# AttnJourney

本项目旨在介绍并探讨 **Transformer** 及其衍生模型在大模型领域的发展历程。本节主要关注 **Attention** 机制，结合相关论文与常见问题进行解析。

## 1. 视频链接
[点击这里观看教程](https://www.bilibili.com/video/BV1njFDeNEzW)

## 2. Attention Is All You Need

- **文献链接**: [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/pdf/1706.03762)  
- **背景**: 该论文由 Vaswani 等人在 2017 年提出，为自然语言处理和序列建模带来了革新，其核心在于 **Self-Attention** 及 **Multi-Head Attention** 技术。

---

## 3. Scaled Dot-Product Attention

**核心公式**:  
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### 3.1. 为什么要除以 $\sqrt{d_k}$ ？

在计算注意力分数 ($\text{scores} = QK^\top$) 时，如果向量维度 $d_k$（通常等于 $d_\text{model} / \text{num-heads}$ ）较大，那么`scores`的数值会变大，导致 **Softmax** 的输入出现极端值，从而引发训练不稳定（梯度爆炸或消失）等问题。

为此，论文提出**缩放（Scaling）** 方案 ——  除以 $\sqrt{d_k}$ ：
- 保持注意力分数的数值分布相对稳定，利于训练收敛。

> **总结**：除以 $\sqrt{d_k}$ 可以让数值更平滑，避免Softmax的输出分布过度集中，减轻梯度问题。

---

### 3.2. 为什么mask要填充负无穷？

在注意力机制中，我们通常需要对某些位置进行“屏蔽”（Mask），包括：
- **Padding Mask** : 对无效位置（如填充符）进行屏蔽；
- **Causal Mask** : 在自回归模型中屏蔽未来时刻的token（只允许看到过去）。

做法是：

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

之后进行Softmax时，这些被置为`-∞`的位置，其注意力权重就会趋近于0，从而实现屏蔽的效果。

> **总结**：用`-∞`来使被遮挡位置的Softmax输出为0，彻底杜绝模型从无效区域获取信息。

---

### 3.3. Dropout的作用是什么？

```python
attention_weights = self.dropout(attention_weights)
```
- Dropout会随机把一部分注意力权重置为0，从而**抑制过拟合**，提升模型的泛化能力。
- 在注意力机制中，这被视作在训练时增加随机性，增强鲁棒性。

> **总结**：**Dropout**在这里是对注意力权重做随机丢弃，减少对特定连接的过度依赖。

---

### 3.4. 为什么是 $QK^\top$ 之后做Softmax，而不是 $QKV$ 一起？

1. **Q, K, V 各司其职**  
   - $Q$ (Query): “查询”向量  
   - $K$ (Key): “键”向量  
   - $V$ (Value): “值”向量

2. **注意力三步走**  
   1. 计算 $\text{scores} = QK^\top$ ，得到各元素之间的相似度；
   2. 通过 $\text{Softmax}$ 将相似度变为概率分布；
   3. 用得到的概率分布加权 $V$ ，得到最终上下文向量。

若将 $Q, K, V$ 直接合并后再Softmax，会混淆**相关性计算**($QK^\top$)与**信息提取**（加权 $V$ ）这两步。

> **总结**：先对 $QK^\top$ 做Softmax获得注意力分布，再基于此分布加权 $V$ ，这是**Attention**机制的核心流程。

---

## 4. Multi-Head Attention

**多头注意力（MHA）** 可以理解为并行地做多份“Scaled Dot-Product Attention”，然后将各头的结果拼接起来并投影到输出空间。其优点在于：
1. **并行关注不同位置**: 每个注意力头可捕捉不同的特征或上下文依赖；
2. **表达能力更强**: 增加了模型对序列多方面信息的捕获能力。

计算流程：
1. 对输入$X$线性映射成 $Q, K, V$ ；
2. 进行多头拆分（`view`, `transpose`），得到形如 $(batch\_size, num\_heads, seq\_len, d_k)$ 的张量；
3. 计算 $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$ ；
4. 结果拼接，过线性层得到最终输出。

---

## 5. GPT自注意力

### 5.1. GPT Attention的最早论文

- GPT-1 (2018): *Improving Language Understanding by Generative Pre-Training*  
  [OpenAI Blog](https://openai.com/blog/language-unsupervised) | [Paper (PDF)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

此后有GPT-2 (2019), GPT-3 (2020), GPT-4 (2023)等持续演进，都使用了**Decoder-only自注意力**和**因果掩码**来实现从左到右生成。

---

### 5.2. GPT注意力与标准多头注意力的主要区别

**GPT使用因果掩码**（Causal Mask）在计算Softmax前对未来位置进行屏蔽，保证只能看“过去”token。  
- 标准多头注意力（如Transformer Encoder）则不需要这种因果限制；
- 若是Transformer Decoder，也要显式传递一个look-ahead mask。

> **总结**：GPT与标准MHA最大区别在于**自动加“因果掩码”**。

---

### 5.3. 为什么GPT只有一个输入 $(x)$ 而标准MHA需要 $Q, K, V$ 三个参数？

1. 在**自注意力**场景下（Self-Attention）， $Q=K=V$ 都来自同一批张量，如GPT。  
2. 在**Encoder-Decoder Attention**场景下， $Q$ 通常来自Decoder隐状态， $K, V$ 来自Encoder隐状态，所以要分别传入。

> **总结**：GPT自注意力中 $Q=K=V$ 都是同一个输入；在通用的多头注意力中， $Q、K、V$ 被分开是为了更灵活的使用场景。

---

## 6. DeepSeek-V3 MLA

**MLA**（Multi-Head Latent Attention）在Key/Value上进行**低秩压缩**，可显著减小推理时的KV缓存开销，同时在实践中也会带来一定的加速。具体来说：

1. K, V不再是 $(batch\_size, seq\_len, d\_model)$，而是被映射到更小的维度 $(d_c)$ ；
2. 再将 $(d_c)$ 投影到多头所需的 $(d_c^h + d_r^h)$ 维度，用以参与注意力运算；
3. 只在注意力核心流程上保留较小的矩阵维度，从而降低计算量与内存。

### 6.1. MLA为什么也会加速？

1. **降维后运算量更小**  
   Key/Value的维度变小，Attention过程中的矩阵乘法规模也随之减小。
2. **KV缓存量更小**  
   对大模型推理尤其重要，可减少显存或网络带宽瓶颈。

### 6.2. GPT自注意力为何比标准MHA快？

- 在本地测试中，GPT自注意力封装更直接，函数调用路径更短或可能存在其他优化，从而运行更快。  
- **Mask**并不会真正“少算”矩阵乘法（一般PyTorch实现中只是在乘法后填充负无穷），更多是实现细节或随机波动造成时差。

### 6.3. 为什么只对 $Q$ 和 $K$ 做RoPE而不对 $V$ ？

- RoPE的核心思想是让 $QK^\top$ 中出现可体现位置关系的旋转算子， $Q, K$ 各自携带位置信息后，即可让注意力分布反映相对位置。  
- **V**不需要做旋转；若也做，则还得考虑“逆向”操作，复杂且无明显收益。

### 6.4. `permute`的作用

- PyTorch中`permute`用于**调整张量的维度顺序**，与单纯的2D“转置”不同，可在更高维张量中任意交换维度。  
- 在注意力中，常见用法是将 $(batch\_size, seq\_len, n\_heads, d\_k)$ 变为 $(batch\_size, n\_heads, seq\_len, d\_k)$ ，方便后续进行矩阵乘法或计算注意力分布。

---

## 7. 总结

1. **Scaled Dot-Product Attention**是Transformer的基础，除以 $\sqrt{d_k}$ 来稳定数值；
   Mask填充`-∞`可屏蔽无效位置，配合Softmax使注意力权重归零；
3. **Dropout**对注意力权重做随机丢弃，防止过拟合；
4. **Q, K, V**各自扮演“查询、键、值”角色，**先**计算 $QK^\top$ 相关性，再做Softmax加权 $V$ ；
5. **GPT自注意力**同样遵循多头注意力原理，但加了因果掩码来保证只能看过去；
6. **MLA**通过Key/Value低秩压缩，减小推理缓存与计算量，适合大模型优化；
7. **RoPE**只加在 $Q$ 和 $K$ 上，以在相关性分数中注入位置信息；
8. `permute`用于多维度交换，为注意力计算做形状适配。

以上即为对**Attention**相关问题与三种实现方式的简要梳理，以及相应原理的解释。希望能帮助你更好地理解**Transformer**体系及其演化思路。
