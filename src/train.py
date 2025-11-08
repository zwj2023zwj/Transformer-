import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from tqdm import tqdm
import random
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.utils.data_utils import load_iwslt2017_data, create_iwslt_dataloaders
from src.utils.train_utils import LabelSmoothingLoss, train, get_scheduler
from src.utils.visualization import plot_training_curves, save_training_stats, create_results_directory

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Train Transformer on IWSLT2017')
    # 数据参数
    parser.add_argument('--dataset', type=str, default='iwslt', choices=['iwslt'], help='Dataset to use iwslt')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory for IWSLT dataset')
    parser.add_argument('--language_pair', type=str, default='de-en', choices=['de-en', 'en-de'], help='Language pair for translation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (for debugging)')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples in toy dataset')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size for toy dataset')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--is_pos_encoding', type=bool, default=True, help='Use positional encoding')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--use_adamw', action='store_true', help='Use AdamW optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'linear', 'step', 'plateau'], 
                        help='Learning rate scheduler type')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='../results', help='Directory to save results')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
    
    # 并行/分布式参数
    parser.add_argument('--ddp', action='store_true', help='Use DistributedDataParallel for multi-GPU')
    parser.add_argument('--devices', type=str, default=None, help='Comma-separated GPU ids for DataParallel, e.g. "0,1"')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DDP (set by torchrun)')
    parser.add_argument('--dist_backend', type=str, default=None, choices=['nccl', 'gloo'], help='DDP backend (default: nccl on Linux, gloo on Windows)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查CUDA可用性
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # 设备与并行初始化
    is_master = True
    if args.ddp:
        backend = args.dist_backend or ('nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo')
        dist.init_process_group(backend=backend, init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        if torch.cuda.is_available():
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        is_master = (dist.get_rank() == 0)
        if is_master:
            print(f"DDP initialized | backend: {backend} | world_size: {dist.get_world_size()} | local_rank: {local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
    # 创建结果目录
    # result_dir = create_results_directory(args.save_dir)
    # print(f"Results will be saved to {result_dir}")
    
    # 创建结果目录（仅主进程创建/打印）
    result_dir = None
    if is_master:
        result_dir = create_results_directory(args.save_dir)
        print(f"Results will be saved to {result_dir}")
        
    print(f"Loading IWSLT2017 dataset ({args.language_pair})...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, src_vocab, tgt_vocab = load_iwslt2017_data(
        data_dir=args.data_dir,
        language_pair=args.language_pair,
        max_samples=args.max_samples
    )
    
    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    print(f"Test samples: {len(test_src)}")
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    train_loader, val_loader = create_iwslt_dataloaders(
        train_src, train_tgt, val_src, val_tgt,
        src_vocab, tgt_vocab,
        batch_size=args.batch_size,
        max_len=args.max_len,
        distributed=args.ddp
    )
    
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    
    # 创建模型
    print("Creating Transformer model...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        is_pos_encoding=args.is_pos_encoding
    ).to(device)
    
    # 多卡封装
    if args.ddp:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, output_device=device.index if device.type == 'cuda' else None)
        else:
            model = DDP(model)
    elif args.devices and torch.cuda.is_available():
        device_ids = [int(i) for i in args.devices.split(',') if i.strip()]
        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids)
            
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建优化器
    if args.use_adamw:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Using AdamW optimizer")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("Using Adam optimizer")
    
    # (如果使用DDP，len(train_loader) 在每个进程中都是一样的)
    num_training_steps = len(train_loader) * args.epochs
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    # if is_master:
    #     print(f"Using linear scheduler with {args.warmup_steps} warmup steps over a total of {num_training_steps} training steps.")
        
    # 创建学习率调度器
    scheduler = None
    if args.scheduler:
        if args.scheduler == 'cosine':
            scheduler = get_scheduler(optimizer, 'cosine', T_max=args.epochs)
        elif args.scheduler == 'linear':
            scheduler = get_scheduler(optimizer, 'linear', total_iters=args.epochs)
        elif args.scheduler == 'step':
            scheduler = get_scheduler(optimizer, 'step', step_size=args.epochs // 3)
        elif args.scheduler == 'plateau':
            scheduler = get_scheduler(optimizer, 'plateau', patience=2)
        print(f"Using {args.scheduler} learning rate scheduler")
    
    # 创建损失函数
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    # criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.get("<pad>", -1), label_smoothing=0.1)
    print(f"Using label smoothing with factor {args.label_smoothing}")
    
    # 训练模型
    # print("Starting training...")
    if is_master:
        print("Starting training...")
    train_losses, val_losses, best_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=args.epochs,
        clip_grad=args.clip_grad,
        scheduler=scheduler,
        patience=args.patience,
        # save_path=os.path.join(result_dir, 'best_model.pt'),
        save_path=(os.path.join(result_dir, 'best_model.pt') if is_master and result_dir is not None else None),
        log_interval=args.log_interval,
        use_adamw=args.use_adamw,
        is_master=is_master,
        tgt_vocab=tgt_vocab
    )
    
    # # 保存训练统计数据
    # save_training_stats(
    #     train_losses=train_losses,
    #     val_losses=val_losses,
    #     save_path=os.path.join(result_dir, 'training_stats.csv')
    # )
    
    # # 绘制训练曲线
    # plot_training_curves(
    #     train_losses=train_losses,
    #     val_losses=val_losses,
    #     save_path=os.path.join(result_dir, 'training_curves.png'),
    #     show=False
    # )
    
    # print(f"Training completed. Best model saved at epoch {best_model['epoch']} with validation loss {best_model['val_loss']:.4f}")
    # print(f"Results saved to {result_dir}")
    if is_master and result_dir is not None:
        save_training_stats(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=os.path.join(result_dir, 'training_stats.csv')
        )
        
        # 绘制训练曲线
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=os.path.join(result_dir, 'training_curves.png'),
            show=False
        )
        if best_model is not None:
            print(f"Training completed. Best model saved at epoch {best_model['epoch']} with validation loss {best_model['val_loss']:.4f}")
        print(f"Results saved to {result_dir}")

    # 结束DDP进程组
    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()
        
if __name__ == '__main__':
    main()