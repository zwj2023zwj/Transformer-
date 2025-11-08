import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    """
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        计算注意力权重并应用于value
        
        Args:
            query: [batch_size, n_heads, seq_len, d_k]
            key: [batch_size, n_heads, seq_len, d_k]
            value: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # 计算注意力分数
        # [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力权重到value
        output = torch.matmul(attention, value)
        
        return output, attention

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self._initialize_weights()
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model)
    
    def _initialize_weights(self):
        """
        使用Xavier Uniform初始化权重，将偏置初始化为0。
        """
        # 对W_q, W_k, W_v, W_o进行初始化
        for linear_layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            # 使用Xavier Uniform初始化权重
            init.xavier_uniform_(linear_layer.weight)
            # 如果有偏置项，则初始化为0
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0)
                
    def forward(self, query, key, value, mask=None):
        """
        多头注意力计算
        
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention: [batch_size, n_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        residual = query
        
        # 线性投影并分割成多头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # 应用注意力机制
        output, attention = self.attention(q, k, v, mask)
        
        # 合并多头
        # [batch_size, n_heads, seq_len, d_v] -> [batch_size, seq_len, n_heads, d_v] -> [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = self.W_o(output)
        output = self.dropout(output)
        
        # 残差连接和层归一化
        # output = self.layer_norm(residual + output)
        
        return output, attention