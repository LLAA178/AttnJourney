import time
import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################################
# Scaled Dot-Product Attention
#########################################################
class ScaledDotProductAttention(nn.Module):
    """
    实现论文《Attention is All You Need》中提出的
    Scaled Dot-Product Attention。
    用于在多头注意力机制中计算注意力分数。
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量的维度（或隐藏层维度）。
            dropout: 在注意力权重计算后的dropout概率。
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        前向传播:
            query, key, value: 形状一般为 (batch_size, num_heads, seq_len, d_k)
            mask: 可选的注意力掩码，形状一般为
                  (batch_size, 1, seq_len, seq_len) 或 (batch_size, num_heads, seq_len, seq_len)
        
        返回:
            output: 加权后的上下文向量 (batch_size, num_heads, seq_len, d_k)
            attention_weights: 注意力分布 (batch_size, num_heads, seq_len, seq_len)
        """
        # 1) 点积计算
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # 2) 如果提供了mask，则对mask为0的地方进行填充负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 3) 通过softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4) 进行dropout
        attention_weights = self.dropout(attention_weights)
        
        # 5) 与value相乘得到输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


#########################################################
# 1. 标准Transformer多头注意力 (Encoder/Decoder通用)
#########################################################
class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制。可应用在Transformer的Encoder或Decoder中。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量的维度（或隐藏层维度）。
            num_heads: 多头数。
            dropout: dropout概率。
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除。"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 查询、键、值的线性映射
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.output_linear = nn.Linear(d_model, d_model)
        
        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor = None):
        """
        前向传播:
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # 1) 映射到多头空间
        query = self.query_linear(query)  # (batch_size, seq_len, d_model)
        key = self.key_linear(key)        # (batch_size, seq_len, d_model)
        value = self.value_linear(value)  # (batch_size, seq_len, d_model)
        
        # 2) 拆分为多头: (batch_size, num_heads, seq_len, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3) 计算注意力
        output, attention_weights = self.attention(query, key, value, mask)
        
        # 4) 拼接多头输出 (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5) 最后的线性映射
        output = self.output_linear(output)
        
        return output, attention_weights


#########################################################
# 2. GPT风格因果自注意力 (Decoder-only, Causal Mask)
#########################################################
class CausalSelfAttention(nn.Module):
    """
    GPT-style Causal Self-Attention。
    只允许关注当前和之前的token（通过因果掩码），
    实现自回归生成。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量的维度（或隐藏层维度）。
            num_heads: 多头数。
            dropout: dropout概率。
        """
        super(CausalSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除。"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 查询、键、值的线性映射
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        """
        前向传播:
            x: (batch_size, seq_len, d_model)，
               GPT通常只需要一个输入张量作为Self-Attention的Q、K、V。
        """
        bsz, seq_len, _ = x.size()
        
        # 1) 做线性变换并切分成多头
        q = self.query_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2) 计算 QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 3) 生成因果掩码（下三角为1, 上三角为0）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 4) 通过softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5) 得到上下文向量
        out = torch.matmul(attn_weights, v)
        
        # 6) 拼接多头
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_linear(out)
        
        return out, attn_weights


#########################################################
# 3. BERT自注意力 (Encoder-only, Bidirectional)
#########################################################
class BertSelfAttention(nn.Module):
    """
    BERT风格的自注意力机制，属于Encoder-only结构。
    不考虑因果关系，可双向关注上下文。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量的维度（或隐藏层维度）。
            num_heads: 多头数。
            dropout: dropout概率。
        """
        super(BertSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除。"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        前向传播:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, num_heads, seq_len, seq_len) 或能广播到这个形状的张量。
        """
        batch_size, seq_len, _ = hidden_states.size()

        # 1) 映射到Q, K, V
        query = self.query_linear(hidden_states)
        key = self.key_linear(hidden_states)
        value = self.value_linear(hidden_states)

        # 2) 分成多头
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3) 计算注意力分数
        output, attn_weights = self.attn(query, key, value, attention_mask)

        # 4) 拼接并还原形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 5) 输出线性层
        output = self.out_linear(output)
        output = self.dropout(output)

        return output, attn_weights


#########################################################
# 测试与对比
#########################################################
if __name__ == "__main__":
    # 准备测试数据
    batch_size = 2
    seq_len = 5
    d_model = 8
    num_heads = 2
    
    # (1) 测试标准Transformer多头注意力
    print("=== 测试1: MultiHeadAttention ===")
    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    start_time = time.time()
    output_mha, attn_mha = mha(query, key, value, mask=None)
    end_time = time.time()
    
    print(f"Output shape (MHA): {output_mha.shape}")
    print(f"Attention shape (MHA): {attn_mha.shape}")
    print(f"Output mean (MHA): {output_mha.mean().item():.4f}, std: {output_mha.std().item():.4f}")
    print(f"耗时 (MHA): {(end_time - start_time) * 1000:.2f} ms\n")
    
    # (2) 测试GPT风格因果自注意力
    print("=== 测试2: CausalSelfAttention (GPT) ===")
    x = torch.rand(batch_size, seq_len, d_model)
    gpt_attn = CausalSelfAttention(d_model, num_heads)
    
    start_time = time.time()
    output_gpt, attn_gpt = gpt_attn(x)
    end_time = time.time()
    
    print(f"Output shape (GPT): {output_gpt.shape}")
    print(f"Attention shape (GPT): {attn_gpt.shape}")
    print(f"Output mean (GPT): {output_gpt.mean().item():.4f}, std: {output_gpt.std().item():.4f}")
    print(f"耗时 (GPT): {(end_time - start_time) * 1000:.2f} ms\n")
    
    # (3) 测试BERT风格自注意力
    print("=== 测试3: BertSelfAttention ===")
    hidden_states = torch.rand(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, num_heads, seq_len, seq_len)  # 常见的BERT padding mask
    bert_attn = BertSelfAttention(d_model, num_heads)
    
    start_time = time.time()
    output_bert, attn_bert = bert_attn(hidden_states, attention_mask)
    end_time = time.time()
    
    print(f"Output shape (BERT): {output_bert.shape}")
    print(f"Attention shape (BERT): {attn_bert.shape}")
    print(f"Output mean (BERT): {output_bert.mean().item():.4f}, std: {output_bert.std().item():.4f}")
    print(f"耗时 (BERT): {(end_time - start_time) * 1000:.2f} ms\n")
