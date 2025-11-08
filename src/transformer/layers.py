import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.init as init

class PositionalEncoding(nn.Module):
    """
    位置编码
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区（不作为模型参数）
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        # 添加位置编码到输入
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(nn.Module):
    """
    Position-wise前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        根据激活函数选择合适的初始化方法。
        """
        # 第一个线性层，后面接ReLU，使用Kaiming初始化
        # a=0 表示ReLU, mode='fan_in' 是默认值且常用
        init.kaiming_uniform_(self.linear1.weight, a=0, nonlinearity='relu')
        if self.linear1.bias is not None:
            init.constant_(self.linear1.bias, 0)

        # 第二个线性层，不直接接激活函数，使用Xavier初始化
        init.xavier_uniform_(self.linear2.weight)
        if self.linear2.bias is not None:
            init.constant_(self.linear2.bias, 0)
            
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model]
        """
        residual = x
        
        # 两层前馈网络，中间使用ReLU激活函数
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        # 残差连接和层归一化
        # x = self.layer_norm(residual + x)
        
        return x

# def create_padding_mask(seq, pad_idx=0):
#     """
#     创建padding mask
    
#     Args:
#         seq: [batch_size, seq_len]
#         pad_idx: padding的索引值
        
#     Returns:
#         mask: [batch_size, 1, 1, seq_len]
#     """
#     # 创建mask，将padding位置标记为0，非padding位置标记为1
#     mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
#     return mask

# def create_look_ahead_mask(seq_len):
#     """
#     创建前瞻mask（用于解码器中的自注意力机制）
    
#     Args:
#         seq_len: 序列长度
        
#     Returns:
#         mask: [seq_len, seq_len]
#     """
#     # 创建上三角矩阵（包括对角线）
#     mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
#     # 将上三角部分设为0，其余部分设为1
#     mask = (1 - mask).bool()
#     return mask

# def create_masks(src, tgt, pad_idx=0):
#     """
#     创建编码器和解码器所需的所有mask
    
#     Args:
#         src: [batch_size, src_len]
#         tgt: [batch_size, tgt_len]
#         pad_idx: padding的索引值
        
#     Returns:
#         enc_mask: [batch_size, 1, 1, src_len]
#         dec_mask: [batch_size, 1, tgt_len, tgt_len]
#         dec_enc_mask: [batch_size, 1, tgt_len, src_len]
#     """
#     # 编码器的padding mask
#     enc_mask = create_padding_mask(src, pad_idx)
    
#     # 解码器的padding mask
#     dec_padding_mask = create_padding_mask(tgt, pad_idx)
    
#     # 解码器的前瞻mask
#     batch_size, tgt_len = tgt.size()
#     look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)
    
#     # 组合前瞻mask和padding mask
#     dec_mask = torch.logical_and(dec_padding_mask, look_ahead_mask.unsqueeze(0))
    
#     # 解码器-编码器的padding mask
#     dec_enc_mask = create_padding_mask(src, pad_idx)
    
#     return enc_mask, dec_mask, dec_enc_mask

def create_padding_mask(seq, pad_idx=0):
    """创建padding mask (保持不变)"""
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(seq_len):
    """创建前瞻mask (逻辑可以简化，但您原来的也没错)"""
    # 使用 torch.tril 更直接
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask

def create_masks(src, tgt, pad_idx=0):
    """
    创建编码器和解码器所需的所有mask (已修复)
    
    Args:
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        pad_idx: padding的索引值
        
    Returns:
        enc_mask: [batch_size, 1, 1, src_len]
        dec_mask: [batch_size, 1, tgt_len, tgt_len]
        dec_enc_mask: [batch_size, 1, 1, src_len] # <--- 注意力！dec_enc_mask 的形状也应调整
    """
    # 编码器的padding mask (保持不变)
    # Shape: [batch_size, 1, 1, src_len]
    enc_mask = create_padding_mask(src, pad_idx)
    
    # --- ✨ BUG修复区域开始 ✨ ---

    # 1. 创建目标语言的填充掩码，但形状要调整为 [batch_size, 1, 1, tgt_len]
    #    这样它关注的是被注意的词(key)是否是padding
    dec_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2) # Shape: [batch_size, 1, 1, tgt_len]

    # 2. 创建前瞻掩码
    tgt_len = tgt.size(1)
    look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device) # Shape: [tgt_len, tgt_len]

    # 3. 组合两者。PyTorch的广播机制会自动处理
    #    (dec_padding_mask) [B, 1, 1, L] & (look_ahead_mask) [L, L] -> [B, 1, L, L]
    dec_mask = dec_padding_mask & look_ahead_mask
    
    # --- ✨ BUG修复区域结束 ✨ ---

    # 解码器-编码器的交叉注意力掩码
    # 这个掩码的 Query 来自 tgt，Key 来自 src。所以形状应该是 [batch_size, 1, 1, src_len]
    # 它和 enc_mask 完全一样
    dec_enc_mask = create_padding_mask(src, pad_idx)
    
    return enc_mask, dec_mask, dec_enc_mask