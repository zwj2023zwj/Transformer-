import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .layers import create_masks

class Transformer(nn.Module):
    """
    完整的Transformer模型
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, max_seq_len=5000, dropout=0.1, 
                 pad_idx=0, sos_idx=2, eos_idx=3, is_pos_encoding=True):
        super(Transformer, self).__init__()
        
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        
        # 编码器
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx, 
            is_pos_encoding=is_pos_encoding
        )
        
        # 解码器
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
    def forward(self, src, tgt):
        """
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
            enc_attentions: 编码器自注意力权重
            dec_self_attentions: 解码器自注意力权重
            dec_cross_attentions: 解码器交叉注意力权重
        """
        # 创建mask
        enc_mask, dec_mask, dec_enc_mask = create_masks(src, tgt[:, :-1], self.pad_idx)
        
        # print(src)
        # print(tgt)
        # print(enc_mask)
        # print(dec_mask)
        # print(dec_enc_mask)
        
        # 编码器前向传播
        enc_output, enc_attentions = self.encoder(src, enc_mask)
        
        # 解码器前向传播（移除最后一个token，因为我们预测下一个token）
        output, (dec_self_attentions, dec_cross_attentions) = self.decoder(
            tgt[:, :-1], enc_output, dec_mask, dec_enc_mask
        )
        
        return output, enc_attentions, dec_self_attentions, dec_cross_attentions
    
    def encode(self, src):
        """
        仅执行编码过程
        
        Args:
            src: [batch_size, src_len]
            
        Returns:
            enc_output: [batch_size, src_len, d_model]
            enc_mask: [batch_size, 1, 1, src_len]
        """
        # 创建编码器mask
        enc_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 编码器前向传播
        enc_output, _ = self.encoder(src, enc_mask)
        
        return enc_output, enc_mask
    
    # def decode(self, tgt, enc_output, enc_mask):
    #     """
    #     给定编码器输出，执行解码过程
        
    #     Args:
    #         tgt: [batch_size, tgt_len]
    #         enc_output: [batch_size, src_len, d_model]
    #         enc_mask: [batch_size, 1, 1, src_len]
            
    #     Returns:
    #         output: [batch_size, tgt_len, tgt_vocab_size]
    #     """
    #     # 创建解码器mask
    #     tgt_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
    #     seq_len = tgt.size(1)
    #     look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(tgt.device)
    #     dec_mask = torch.logical_and(tgt_mask, ~look_ahead_mask.unsqueeze(0))
        
    #     # 解码器前向传播
    #     output, _ = self.decoder(tgt, enc_output, dec_mask, enc_mask)
        
    #     return output
    def decode(self, tgt, enc_output, enc_mask):
        """
        给定编码器输出，执行解码过程 (已修复)
        
        Args:
            tgt: [batch_size, tgt_len]
            enc_output: [batch_size, src_len, d_model]
            enc_mask: [batch_size, 1, 1, src_len]
            
        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        # --- 修复：确保掩码创建逻辑与训练时一致 ---
        # 1. Padding mask (True for non-pad positions)
        dec_padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 2. Look-ahead mask (下三角矩阵, True for past and present)
        tgt_len = tgt.size(1)
        look_ahead_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        # 3. 组合成最终的解码器自注意力掩码
        dec_mask = dec_padding_mask & look_ahead_mask
        
        # --- 修复结束 ---
        
        # 解码器前向传播
        # 注意：这里的 cross_mask (交叉注意力掩码) 应该使用 enc_mask
        output, _ = self.decoder(tgt, enc_output, dec_mask, enc_mask)
        
        return output
    
    def generate(self, src, max_len=100):
        """
        生成序列
        
        Args:
            src: [batch_size, src_len]
            max_len: 生成序列的最大长度
            
        Returns:
            outputs: [batch_size, max_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # 编码
        enc_output, enc_mask = self.encode(src)
        
        # 初始化目标序列
        tgt = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        
        # 逐步生成
        for i in range(max_len - 1):
            # 解码当前序列
            output = self.decode(tgt, enc_output, enc_mask)
            
            # 获取下一个token的预测
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # 将预测添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # 如果所有序列都生成了EOS，则提前停止
            if (next_token == self.eos_idx).all():
                break
                
        return tgt
    
    # 在你的 Transformer 类 (transformer.py) 中添加这个新方法

    # def beam_search_decode(self, src, beam_width=5, max_len=100):
    #     """
    #     使用束搜索进行解码 (batch_size=1)
        
    #     Args:
    #         src: 源语言句子张量 [1, src_len]
    #         beam_width: 束宽 (k)
    #         max_len: 生成句子的最大长度
            
    #     Returns:
    #         Tensor: 生成的最佳目标语言句子ID序列
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         device = src.device
            
    #         # 1. 编码器只需要计算一次
    #         enc_output, enc_mask = self.encode(src)
            
    #         # 2. 初始化束 (beams)
    #         # 每个 beam 包含 (序列, 分数)
    #         # 初始时，只有一个 beam，包含 <sos>，分数为 0
    #         beams = [(torch.full((1, 1), self.sos_idx, dtype=torch.long, device=device), 0.0)]
            
    #         # 3. 循环进行解码
    #         for _ in range(max_len - 1):
    #             new_beams = []
    #             all_done = True # 检查是否所有 beam 都已结束
                
    #             for seq, score in beams:
    #                 # 如果当前 beam 已经以 <eos> 结尾，则直接保留
    #                 if seq[0, -1].item() == self.eos_idx:
    #                     new_beams.append((seq, score))
    #                     continue
                    
    #                 all_done = False # 只要有一个 beam 没结束，就继续
                    
    #                 # 使用 decode 方法获取下一步的概率分布
    #                 # 注意：这里我们只关心最后一个词的输出
    #                 output = self.decode(seq, enc_output, enc_mask)
    #                 next_token_logits = output[:, -1, :] # [1, vocab_size]
                    
    #                 # 使用 log_softmax 获取对数概率，更稳定
    #                 next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
    #                 # 获取 top-k 个最可能的下一个词
    #                 top_k_log_probs, top_k_ids = torch.topk(next_token_log_probs, k=beam_width, dim=-1)
                    
    #                 # 将这 k 个词扩展到当前的 beam 中
    #                 for i in range(beam_width):
    #                     next_id = top_k_ids[0, i].unsqueeze(0).unsqueeze(0) # [1, 1]
    #                     log_prob = top_k_log_probs[0, i].item()
                        
    #                     new_seq = torch.cat([seq, next_id], dim=1)
    #                     new_score = score + log_prob
    #                     new_beams.append((new_seq, new_score))

    #             if all_done:
    #                 break
                
    #             # 4. 从所有候选的新 beam 中，选出 top-k 个
    #             # 我们根据分数进行排序
    #             new_beams.sort(key=lambda x: x[1], reverse=True)
    #             beams = new_beams[:beam_width]

    #         # 5. 选择最佳序列 (可选：使用长度惩罚)
    #         # 这里我们简单地选择分数最高的
    #         best_seq, best_score = beams[0]
            
    #         return best_seq.squeeze(0) # 移除 batch 维度
    def beam_search_decode(self, src, beam_width=5, max_len=100, length_penalty=0.6):
        """
        使用束搜索进行解码 (batch_size=1)
        
        Args:
            src: 源语言句子张量 [1, src_len]
            beam_width: 束宽 (k)
            max_len: 生成句子的最大长度
            length_penalty (float): 长度惩罚因子 alpha. 0.0 表示无惩罚.
            
        Returns:
            Tensor: 生成的最佳目标语言句子ID序列 [seq_len]
        """
        self.eval()
        with torch.no_grad():
            device = src.device
            
            # 1. 编码器只需要计算一次
            enc_output, enc_mask = self.encode(src)
            
            # 2. 初始化 beams
            # beams 存储的是 (序列, 分数) 的元组列表
            beams = [(torch.full((1, 1), self.sos_idx, dtype=torch.long, device=device), 0.0)]
            completed_beams = []

            # 3. 循环进行解码
            for _ in range(max_len - 1):
                new_beams = []
                
                # 如果所有 beam 都已经完成，则提前退出
                if not beams:
                    break
                    
                for seq, score in beams:
                    # 使用 decode 方法获取下一步的概率分布
                    output = self.decode(seq, enc_output, enc_mask)
                    next_token_logits = output[:, -1, :] # [1, vocab_size]
                    
                    # 使用 log_softmax 获取对数概率
                    next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # 获取 top-k 个最可能的下一个词
                    top_k_log_probs, top_k_ids = torch.topk(next_token_log_probs, k=beam_width, dim=-1)
                    
                    # 将这 k 个词扩展到当前的 beam 中
                    for i in range(beam_width):
                        next_id = top_k_ids[0, i].unsqueeze(0).unsqueeze(0) # [1, 1]
                        log_prob = top_k_log_probs[0, i].item()
                        
                        new_seq = torch.cat([seq, next_id], dim=1)
                        new_score = score + log_prob
                        
                        # 如果遇到 <eos>，则将该序列移至 "已完成" 列表
                        if next_id.item() == self.eos_idx:
                            # 应用长度惩罚
                            final_score = new_score / (new_seq.size(1) ** length_penalty)
                            completed_beams.append((new_seq, final_score))
                        else:
                            # 否则，加入到下一次迭代的候选列表中
                            new_beams.append((new_seq, new_score))

                # 4. 从所有候选的新 beam 中，选出 top-k 个
                if not new_beams:
                    break
                
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

            # 5. 如果有完成的序列，从中选择分数最高的
            if completed_beams:
                completed_beams.sort(key=lambda x: x[1], reverse=True)
                best_seq, _ = completed_beams[0]
            else:
                # 如果没有序列正常结束（比如达到max_len），则从当前仍在进行的beams中选择最好的
                best_seq, _ = beams[0]
            
            return best_seq.squeeze(0) # 移除 batch 维度