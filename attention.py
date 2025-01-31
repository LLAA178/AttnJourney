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
        # 1) 点积计算 + 缩放
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # 2) 如果提供了mask，则对mask为0的地方进行填充负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 3) 通过softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4) Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 5) 与value相乘得到最终输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


#########################################################
# 1. 标准Transformer多头注意力 (Encoder/Decoder通用)
#########################################################
class MultiHeadAttention(nn.Module):
    """
    标准多头注意力机制，可应用于Transformer的Encoder或Decoder。
    不包含因果掩码。
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
        
        # Q, K, V 的线性映射
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
            mask: 可选的注意力掩码 (batch_size, 1, seq_len, seq_len) 或其他可广播形状
        """
        batch_size = query.size(0)
        
        # 1) 映射到多头空间
        query = self.query_linear(query)  # (bsz, seq_len, d_model)
        key = self.key_linear(key)        # (bsz, seq_len, d_model)
        value = self.value_linear(value)  # (bsz, seq_len, d_model)
        
        # 2) 拆分为多头: (bsz, num_heads, seq_len, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3) 注意力计算
        output, attention_weights = self.attention(query, key, value, mask)
        
        # 4) 拼回原形状 (bsz, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5) 输出线性映射
        output = self.output_linear(output)
        
        return output, attention_weights


#########################################################
# 2. GPT风格因果自注意力 (Decoder-only, Causal Mask)
#########################################################
class CausalSelfAttention(nn.Module):
    """
    GPT-style Causal Self-Attention。
    只允许关注当前和之前的token，实现自回归生成。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        参数:
            d_model: 词向量维度。
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
        x: (batch_size, seq_len, d_model)，
           GPT场景下，自注意力Q=K=V=x。
        """
        bsz, seq_len, _ = x.size()
        
        # 1) 线性变换并切分多头
        q = self.query_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value_linear(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2) 计算 QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # 3) 因果掩码: 只允许关注到当前及之前位置
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 4) softmax 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5) 加权求和
        out = torch.matmul(attn_weights, v)
        
        # 6) 拼接并输出
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_linear(out)
        
        return out, attn_weights


#########################################################
# 3. DeepSeek-V3 Multi-Head Latent Attention (MLA)
#########################################################
def rope(x: torch.Tensor, n_heads: int, dim_head: int):
    """
    占位示例：对输入向量做RoPE（Rotary Positional Embedding）变换。
    在实际工程中，可直接调用已有RoPE函数。

    x: (bsz, seq_len, n_heads, dim_head)
    """
    # 此处仅返回原值，省略实际RoPE运算。
    return x

class MLA(nn.Module):
    """
    DeepSeek-V3中的 Multi-Head Latent Attention (MLA) 简化实现。
    关键点：对 Key/Value 进行低秩压缩，减少KV缓存占用。
    """
    def __init__(
        self,
        d_model: int,      # 原始embedding/hiddensize (如 7168)
        n_heads: int,      # 注意力头数
        d_c_kv: int,       # 压缩后KV总维度 (如 512)
        d_c_q: int,        # 压缩后Q总维度 (如 512)
        d_r_h: int,        # 每头RoPE维度 (如 64)
        d_c_h: int,        # 每头的“压缩维度” (如 64)
        dropout: float = 0.1
    ):
        """
        参数示例:
          d_model=7168, n_heads=128, d_c_kv=512, d_c_q=512,
          d_r_h=64, d_c_h=64, ...
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_c_kv = d_c_kv
        self.d_c_q = d_c_q
        self.d_r_h = d_r_h
        self.d_c_h = d_c_h
        
        # 降维投影
        self.W_DKV = nn.Linear(d_model, d_c_kv, bias=False)  # -> c^KV
        self.W_DQ  = nn.Linear(d_model, d_c_q,  bias=False)  # -> c^Q
        
        # 升维到多头
        self.W_UK = nn.Linear(d_c_kv, n_heads * d_c_h, bias=False)  # -> k^C
        self.W_UV = nn.Linear(d_c_kv, n_heads * d_c_h, bias=False)  # -> v^C
        self.W_UQ = nn.Linear(d_c_q,  n_heads * d_c_h, bias=False)  # -> q^C
        
        # 产生可加RoPE的向量
        self.W_KR = nn.Linear(d_model,  n_heads * d_r_h, bias=False)  # -> k^R
        self.W_QR = nn.Linear(d_c_q,    n_heads * d_r_h, bias=False)  # -> q^R
        
        # 输出投影
        self.W_O = nn.Linear(n_heads * d_c_h, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor):
        """
        h: (batch_size, seq_len, d_model)
        返回:
          out: (batch_size, seq_len, d_model)
        """
        bsz, seq_len, _ = h.size()

        # 1) 压缩到 c^KV, c^Q
        c_kv = self.W_DKV(h)  # (bsz, seq_len, d_c_kv)
        c_q  = self.W_DQ(h)   # (bsz, seq_len, d_c_q)
        
        # 2) 升维 -> k^C, v^C
        k_c = self.W_UK(c_kv) # (bsz, seq_len, n_heads*d_c_h)
        v_c = self.W_UV(c_kv) # (bsz, seq_len, n_heads*d_c_h)
        
        # 3) k^R (含 RoPE)
        k_r = self.W_KR(h).view(bsz, seq_len, self.n_heads, self.d_r_h)
        k_r = rope(k_r, self.n_heads, self.d_r_h)
        
        # 4) q^C, q^R (含 RoPE)
        q_c = self.W_UQ(c_q).view(bsz, seq_len, self.n_heads, self.d_c_h)
        q_r = self.W_QR(c_q).view(bsz, seq_len, self.n_heads, self.d_r_h)
        q_r = rope(q_r, self.n_heads, self.d_r_h)
        
        # 5) 合并 k, q:  [k^C, k^R], [q^C, q^R]
        k_c = k_c.view(bsz, seq_len, self.n_heads, self.d_c_h)
        v_c = v_c.view(bsz, seq_len, self.n_heads, self.d_c_h)
        k = torch.cat([k_c, k_r], dim=-1)  # (bsz, seq_len, n_heads, d_c_h + d_r_h)
        q = torch.cat([q_c, q_r], dim=-1)  # (bsz, seq_len, n_heads, d_c_h + d_r_h)
        
        # 6) 计算注意力: 先转为 (bsz, n_heads, seq_len, dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v_c = v_c.permute(0, 2, 1, 3)
        
        # dot-product => (bsz, n_heads, seq_len, seq_len)
        dim_total = (self.d_c_h + self.d_r_h)  # k,q拼接后维度
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim_total ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 只对 v^C 做加权
        out_heads = torch.matmul(attn, v_c)  # (bsz, n_heads, seq_len, d_c_h)
        
        # 7) 多头拼接 + 输出
        out_heads = out_heads.permute(0, 2, 1, 3).contiguous()
        out_heads = out_heads.view(bsz, seq_len, -1)  # (bsz, seq_len, n_heads*d_c_h)
        
        out = self.W_O(out_heads)  # (bsz, seq_len, d_model)
        return out


#########################################################
# 测试与对比
#########################################################
if __name__ == "__main__":
    # 测试时的超参（可根据需要调大看区别）
    batch_size = 32
    seq_len = 128
    d_model = 256
    num_heads = 8
    
    # 1) 标准多头注意力 (MHA)
    print("=== 测试1: MultiHeadAttention ===")
    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    
    start_time = time.time()
    output_mha, attn_mha = mha(query, key, value, mask=None)
    mha_time = (time.time() - start_time) * 1000
    print(f"Output shape (MHA): {output_mha.shape}")
    print(f"Attention shape (MHA): {attn_mha.shape}")
    print(f"Output mean (MHA): {output_mha.mean():.4f}, std: {output_mha.std():.4f}")
    print(f"耗时 (MHA): {mha_time:.2f} ms\n")
    
    # 2) GPT风格因果自注意力
    print("=== 测试2: CausalSelfAttention (GPT) ===")
    x = torch.rand(batch_size, seq_len, d_model)
    gpt_attn = CausalSelfAttention(d_model, num_heads)
    
    start_time = time.time()
    output_gpt, attn_gpt = gpt_attn(x)
    gpt_time = (time.time() - start_time) * 1000
    print(f"Output shape (GPT): {output_gpt.shape}")
    print(f"Attention shape (GPT): {attn_gpt.shape}")
    print(f"Output mean (GPT): {output_gpt.mean():.4f}, std: {output_gpt.std():.4f}")
    print(f"耗时 (GPT): {gpt_time:.2f} ms\n")
    
    # 3) DeepSeek-V3风格 MLA
    print("=== 测试3: MLA (DeepSeek-V3) ===")
    # 低秩压缩维度：通常 d_c_kv、d_c_q << d_model
    d_c_kv = 64  # 压缩后KV总维度
    d_c_q  = 64  # 压缩后Q总维度
    d_r_h  = 16  # 每头RoPE维度
    d_c_h  = 16  # 压缩后每头维度
    
    mla = MLA(d_model, num_heads, d_c_kv, d_c_q, d_r_h, d_c_h, dropout=0.0)
    x_mla = torch.rand(batch_size, seq_len, d_model)

    start_time = time.time()
    output_mla = mla(x_mla)
    mla_time = (time.time() - start_time) * 1000

    print(f"Output shape (MLA): {output_mla.shape}")
    print(f"Output mean (MLA): {output_mla.mean():.4f}, std: {output_mla.std():.4f}")
    print(f"耗时 (MLA): {mla_time:.2f} ms\n")

    print("=== 结果对比 ===")
    print(f"MHA耗时:  {mha_time:.2f} ms")
    print(f"GPT耗时:  {gpt_time:.2f} ms")
    print(f"MLA耗时:  {mla_time:.2f} ms")
