import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import math
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    """
    def __init__(self, smoothing=0.1, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        计算标签平滑损失
        
        Args:
            pred: [batch_size, seq_len, vocab_size]
            target: [batch_size, seq_len]
            
        Returns:
            loss: 标签平滑损失值
        """
        
        pred = F.log_softmax(pred, dim=-1)
        
        # 检查输入维度
        if pred.dim() == 3:
            # [batch_size, seq_len, vocab_size]
            batch_size, seq_len, vocab_size = pred.size()
        else:
            # [batch_size*seq_len, vocab_size]
            batch_size, vocab_size = pred.size()
            
        # 创建平滑标签
        with torch.no_grad():
            # 创建one-hot编码
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            
            # 根据pred的维度调整scatter操作
            if pred.dim() == 3:
                true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
                # 处理padding位置
                mask = (target == self.ignore_index).unsqueeze(-1)
            else:
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
                # 处理padding位置
                mask = (target == self.ignore_index).unsqueeze(-1)
                
            true_dist.masked_fill_(mask, 0.0)
            
        return torch.sum(-true_dist * pred, dim=-1).mean()

# def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=None, scheduler=None):
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=None, scheduler=None, is_master=True):
    """
    训练一个epoch
    
    Args:
        model: Transformer模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        clip_grad: 梯度裁剪阈值，None表示不使用梯度裁剪
        scheduler: 学习率调度器，None表示不使用学习率调度
        
    Returns:
        epoch_loss: 当前epoch的平均损失
    """
    model.train()
    epoch_loss = 0
    
    optimizer.zero_grad()
    
    accumulation_steps = 2
    # for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", disable=not is_master):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        # 前向传播
        output, _, _, _ = model(src, tgt)
        
        # 计算损失（忽略padding）
        loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                         tgt[:, 1:].contiguous().view(-1))
        
        # loss = loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # if (i + 1) % accumulation_steps == 0:
        # 梯度裁剪（如果启用）
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # 更新参数
        optimizer.step()
        
        # optimizer.zero_grad()
        
        # 更新学习率（如果使用学习率调度器）
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()

    # return epoch_loss / len(dataloader)
    # 分布式下聚合所有进程的loss以计算全局平均
    if dist.is_initialized():
        loss_sum = torch.tensor(epoch_loss, device=device, dtype=torch.float32)
        count = torch.tensor(float(len(dataloader)), device=device, dtype=torch.float32)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return (loss_sum / count).item()
    else:
        return epoch_loss / len(dataloader)

# def evaluate(model, dataloader, criterion, device):
def evaluate(model, dataloader, criterion, device, is_master=True):
    """
    在验证集上评估模型
    
    Args:
        model: Transformer模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        val_loss: 验证集上的平均损失
    """
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        # for batch in tqdm(dataloader, desc="Evaluating"):
        for batch in tqdm(dataloader, desc="Evaluating", disable=not is_master):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # 前向传播
            output, _, _, _ = model(src, tgt)
            
            # print(output)
            # 计算损失（忽略padding）
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                             tgt[:, 1:].contiguous().view(-1))
            
            val_loss += loss.item()
    
    # return val_loss / len(dataloader)
    # 分布式下聚合所有进程的loss以计算全局平均
    if dist.is_initialized():
        loss_sum = torch.tensor(val_loss, device=device, dtype=torch.float32)
        count = torch.tensor(float(len(dataloader)), device=device, dtype=torch.float32)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return (loss_sum / count).item()
    else:
        return val_loss / len(dataloader)

# def train(model, train_loader, val_loader, optimizer, criterion, device, 
#           n_epochs=10, clip_grad=None, scheduler=None, patience=5, 
#           save_path=None, log_interval=1, use_adamw=False):
def train(model, train_loader, val_loader, optimizer, criterion, device,
          n_epochs=10, clip_grad=None, scheduler=None, patience=5,
          save_path=None, log_interval=1, use_adamw=False, is_master=True, tgt_vocab=None):
    """
    训练模型
    
    Args:
        model: Transformer模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        n_epochs: 训练轮数
        clip_grad: 梯度裁剪阈值，None表示不使用梯度裁剪
        scheduler: 学习率调度器，None表示不使用学习率调度
        patience: 早停耐心值，当验证损失连续多少轮未改善时停止训练
        save_path: 模型保存路径，None表示不保存模型
        log_interval: 日志打印间隔
        use_adamw: 是否使用AdamW优化器
        
    Returns:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        best_model: 最佳模型状态字典
    """
    # 如果使用AdamW，替换优化器
    if use_adamw and not isinstance(optimizer, optim.AdamW):
        print("Switching to AdamW optimizer")
        optimizer = optim.AdamW(model.parameters(), lr=optimizer.param_groups[0]['lr'])
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in tqdm(range(n_epochs), total=n_epochs, desc="Epochs"):
        start_time = time.time()
        
        # DDP随机采样器需要在每个epoch设置不同的seed
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            
        # 训练一个epoch
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad, scheduler)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad, scheduler, is_master=is_master)
        train_losses.append(train_loss)
        
        # 在验证集上评估
        # val_loss = evaluate(model, val_loader, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device, is_master=is_master)
        val_losses.append(val_loss)
        # 计算BLEU分数
        # bleu = calculate_bleu(model, val_loader, device, tgt_vocab, is_master=is_master)
        # if is_master:
        #     print(f"Validation BLEU Score: {bleu * 100:.2f}")

        # 打印日志
        # if (epoch + 1) % log_interval == 0:
        if is_master and ((epoch + 1) % log_interval == 0):
            end_time = time.time()
            print(f"Epoch {epoch+1}/{n_epochs} | Time: {end_time - start_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {
                'epoch': epoch + 1,
                # 'model_state_dict': model.state_dict(),
                'model_state_dict': getattr(model, 'module', model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            patience_counter = 0
            
            # if save_path is not None:
            if is_master and save_path is not None:
                torch.save(best_model, save_path)
                print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            
        # 早停
        if patience is not None and patience_counter >= patience:
            # print(f"Early stopping after {epoch+1} epochs")
            if is_master:
                print(f"Early stopping after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, best_model

# 学习率调度器函数
def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型，可选值：'cosine', 'linear', 'step', 'plateau'
        **kwargs: 调度器的其他参数
        
    Returns:
        scheduler: 学习率调度器
    """
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 10),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'linear':
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.1),
            total_iters=kwargs.get('total_iters', 10)
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            verbose=kwargs.get('verbose', False)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def generate_square_subsequent_mask(sz: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    为序列生成一个上三角的-inf掩码矩阵。
    这用于防止解码器在自注意力机制中关注未来的token。

    参数:
        sz: 目标序列的长度。
        device: 张量所在的设备。

    返回:
        一个形状为 [sz, sz] 的张量，其中上三角部分（不含对角线）为 -inf，
        其余部分为 0.0。
    """
    # 创建一个形状为 (sz, sz) 的矩阵，对角线及以上部分为1，其余为0
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    
    # 将值为1的部分替换为-inf，值为0的部分保持不变。
    # 在PyTorch的注意力模块中，-inf会被加到注意力分数上，
    # 经过softmax后，这些位置的权重会变为0，从而实现遮蔽效果。
    # PyTorch的MultiHeadAttention层会自动处理bool类型的mask，
    # 但返回float类型的-inf mask是更通用的做法。
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask

from torchtext.data.metrics import bleu_score

def calculate_bleu(model, data_loader, device, tgt_vocab, is_master=True):
    model.eval()
    predictions = []
    references = []
    
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    sos_idx = tgt_vocab.get("<sos>", -1)
    eos_idx = tgt_vocab.get("<eos>", -1)

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Calculating BLEU", disable=not is_master):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device) # [batch_size, tgt_len]

            # 使用 beam search 生成预测
            # 注意：beam_search_decode 通常一次处理一句，需要循环
            for i in range(src.size(0)):
                src_sentence = src[i].unsqueeze(0)
                pred_ids = model.module.beam_search_decode(src_sentence, beam_width=1, max_len=100) # 如果用了DDP/DP,用.module访问
                
                # --- 解码ID为Token ---
                pred_tokens = [inv_tgt_vocab.get(t_id.item(), "<unk>") for t_id in pred_ids if t_id.item() not in [sos_idx, eos_idx]]
                ref_tokens = [inv_tgt_vocab.get(t_id.item(), "<unk>") for t_id in tgt[i] if t_id.item() not in [sos_idx, eos_idx, model.module.pad_idx]]
                
                predictions.append(pred_tokens)
                references.append([ref_tokens]) # Sacrebleu 需要 [ref1, ref2, ...] 的格式

    return bleu_score(predictions, references)