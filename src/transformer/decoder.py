import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward

# class DecoderLayer(nn.Module):
#     """
#     Transformer解码器层
#     """
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super(DecoderLayer, self).__init__()
        
#         # 多头自注意力机制
#         self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
#         # 多头交叉注意力机制
#         self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
#         # 前馈神经网络
#         self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
#     def forward(self, x, enc_output, self_mask=None, cross_mask=None):
#         """
#         Args:
#             x: [batch_size, tgt_len, d_model]
#             enc_output: [batch_size, src_len, d_model]
#             self_mask: [batch_size, 1, tgt_len, tgt_len]
#             cross_mask: [batch_size, 1, tgt_len, src_len]
            
#         Returns:
#             x: [batch_size, tgt_len, d_model]
#             self_attention: [batch_size, n_heads, tgt_len, tgt_len]
#             cross_attention: [batch_size, n_heads, tgt_len, src_len]
#         """
#         # 自注意力层
#         x_self, self_attention = self.self_attention(x, x, x, self_mask)
        
#         # 交叉注意力层
#         x_cross, cross_attention = self.cross_attention(x_self, enc_output, enc_output, cross_mask)
        
#         # 前馈神经网络
#         x = self.feed_forward(x_cross)
        
#         return x, self_attention, cross_attention

class DecoderLayer(nn.Module):
    """
    Transformer解码器层 (这是修改后的新版本)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 三个工位 (和原来一样)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 为每个工位准备一套独立的质检工具
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 准备一些 Dropout 工具
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        # --- 这是新的、正确的流水线流程 ---
        
        # 1. 第一个工位：自注意力
        residual = x  # 保存加工前的零件
        x_attn, self_attention = self.self_attention(x, x, x, self_mask)
        # 质检：把加工后的零件(x_attn)和原始零件(residual)加起来，再做归一化
        x = self.norm1(residual + self.dropout1(x_attn))
        
        # 2. 第二个工位：交叉注意力
        residual = x  # 保存加工前的零件
        x_cross, cross_attention = self.cross_attention(x, enc_output, enc_output, cross_mask)
        # 质检
        x = self.norm2(residual + self.dropout2(x_cross))
        
        # 3. 第三个工位：前馈网络
        residual = x  # 保存加工前的零件
        x_ffn = self.feed_forward(x)
        # 质检
        x = self.norm3(residual + self.dropout3(x_ffn))
        
        return x, self_attention, cross_attention
    
class Decoder(nn.Module):
    """
    Transformer解码器
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, 
                 max_seq_len=5000, dropout=0.1, pad_idx=0):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = nn.Parameter(self._get_positional_encoding(max_seq_len, d_model), requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        
        # 创建n_layers个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
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
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: [batch_size, tgt_len]
            enc_output: [batch_size, src_len, d_model]
            self_mask: [batch_size, 1, tgt_len, tgt_len]
            cross_mask: [batch_size, 1, tgt_len, src_len]
            
        Returns:
            output: [batch_size, tgt_len, vocab_size]
            attentions: 包含每层自注意力和交叉注意力权重的元组列表
        """
        seq_len = x.size(1)
        
        # 词嵌入和位置编码
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 通过每个解码器层
        self_attentions, cross_attentions = [], []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, self_mask, cross_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
            
        # 最终线性层
        output = self.fc_out(x)
        
        return output, (self_attentions, cross_attentions)