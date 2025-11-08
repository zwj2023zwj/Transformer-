import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward

# class EncoderLayer(nn.Module):
#     """
#     Transformer编码器层
#     """
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super(EncoderLayer, self).__init__()
        
#         # 多头自注意力机制
#         self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
#         # 前馈神经网络
#         self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: [batch_size, seq_len, d_model]
#             mask: [batch_size, 1, 1, seq_len]
            
#         Returns:
#             x: [batch_size, seq_len, d_model]
#             attention: [batch_size, n_heads, seq_len, seq_len]
#         """
#         # 自注意力层
#         x_attn, attention = self.self_attention(x, x, x, mask)
        
#         # 前馈神经网络
#         x = self.feed_forward(x_attn)
        
#         return x, attention

class EncoderLayer(nn.Module):
    """
    Transformer编码器层 (这是修改后的新版本)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # 两个工位 (和原来一样)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 为每个工位准备一套独立的质检工具
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 准备一些 Dropout 工具
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        这是新的、正确的流水线流程
        """
        # 1. 第一个工位：自注意力
        residual = x  # 保存加工前的零件
        x_attn, attention = self.self_attention(x, x, x, mask)
        # 质检：把加工后的零件(x_attn)和原始零件(residual)加起来，再做归一化
        x = self.norm1(residual + self.dropout1(x_attn))
        
        # 2. 第二个工位：前馈网络
        residual = x  # 保存加工前的零件
        x_ffn = self.feed_forward(x)
        # 质检
        x = self.norm2(residual + self.dropout2(x_ffn))
        
        return x, attention
    
class Encoder(nn.Module):
    """
    Transformer编码器
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, is_pos_encoding=True, 
                 max_seq_len=5000, dropout=0.1, pad_idx=0):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = nn.Parameter(self._get_positional_encoding(max_seq_len, d_model), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        self.is_pos_encoding = is_pos_encoding
        
        # 创建n_layers个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def _get_positional_encoding(self, max_seq_len, d_model):
        """
        生成位置编码
        """
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len]
            mask: [batch_size, 1, 1, seq_len]
            
        Returns:
            x: [batch_size, seq_len, d_model]
            attentions: 包含每层注意力权重的列表
        """
        seq_len = x.size(1)
        
        # 词嵌入和位置编码
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        if self.is_pos_encoding:
            x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过每个编码器层
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, mask)
            attentions.append(attention)
            
        return x, attentions