# The Evolution of Large Models

## Attention is All You Need

文献链接：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

## Scaled Dot-Product Attention

### 1. 为什么要除以 \(\sqrt{d_k}\)？

在计算注意力分数（\( \text{scores} = QK^T \)）时，若向量维度 \( d_k \) （通常等于 \( d_{\text{model}} / \text{num\_heads} \)）比较大，则内积数值会变得很大，导致 **Softmax** 的输入分布出现“极端值”，从而使得梯度变得不稳定或过小。  

论文 *Attention is All You Need*（Vaswani et al. 2017）中提出，为了缓解这个问题，需要在点积结果上做一次**缩放（Scaling）**，即**除以 \(\sqrt{d_k}\)**。当 \( d_k = d_{\text{model}} \) 时，也可以直接写成 \(\sqrt{d_{\text{model}}}\)。这样可以让注意力分数更稳定，有助于训练收敛。

> **总结**：除以 \(\sqrt{d_k}\) 是为了让数值不会过大或过小，使得 Softmax 更稳定，**减轻梯度消失或爆炸**的风险。

---

### 2. 为什么 mask 要填充负无穷？

在注意力机制中，我们会对需要**屏蔽**（mask）的部分（例如 Padding 位置、未来时刻）进行处理。做法是：  

```python
scores = scores.masked_fill(mask == 0, float('-inf'))
```

这一步将 mask 为 0 的位置的 \(\text{scores}\) 设置成 \(-\infty\)。  

随后通过
```python
attention_weights = F.softmax(scores, dim=-1)
```
Softmax 时，对于数值为 \(-\infty\) 的位置，计算得到的概率几乎是 0。这样就可以**彻底抑制**模型从这些屏蔽位置获取信息。

> **总结**：用 \(-\infty\) 让 Softmax 输出 0，**实现对某些时刻或 token 的注意力屏蔽**。

---

### 3. Dropout 的作用是什么？

```python
attention_weights = self.dropout(attention_weights)
```

是对注意力权重做一次 Dropout 操作。  
- Dropout 会在训练过程中以一定概率（如 0.1）将一些元素（权重）置为 0，从而**随机地减少网络中某些连接的影响**。  
- 这样做可以有效**缓解过拟合**、增强模型的鲁棒性。

> **总结**：Dropout 通过**随机丢弃一部分连接**来预防过拟合，提升模型泛化能力。

---

### 4. 为什么是 \(QK^T\) 之后做 Softmax，而不是 \(QKV\) 一起？

**Q, K, V 在注意力机制中的作用：**

- \(Q\)（Query）：表示“查询”向量，用于**提问**“我这个位置要从哪些地方（其他位置或词）获取信息？”  
- \(K\)（Key）：表示“键”向量，用于**回答**“你能匹配（或相关）哪些查询？”  
- \(V\)（Value）：表示“值”向量，用于**真正提供信息**给查询者。  

#### 注意力是怎么计算的？

1. 首先，用 \(QK^T\) 计算**相似度**或“匹配度”：  
   \[
     \text{scores} = QK^T
   \]
   通过对 \(\text{scores}\) 做 Softmax，可以得到“我（Query）应该关注 Key 对应的 Value 的程度”。  
2. 接着，将这一概率权重分配给 \(V\)：  
   \[
     \text{output} = \text{Softmax}(QK^T) \times V
   \]
   这一步才能拿到**加权后的上下文向量**，即**注意力输出**。

如果把 \(Q, K, V\) 一起做 Softmax，那就失去了区分**相关性计算**(\(QK^T\)) 和**信息融合**（\(\times V\)）的过程。而且在大多数数学实现里，你**只有在得到相关性分数**之后，才知道如何给 Value 加权。

> **总结**：  
> - **\(QK^T\) 计算相关性**（注意力得分）；  
> - **Softmax** 让这些得分转化成概率分布；  
> - **用该概率分布加权 \(V\)**，完成信息聚合。  
> 这是注意力机制核心的三步，**不能**把 Q, K, V 直接合并后再 Softmax。

---

## Multi-Head Attention

在 Transformer 中，多头注意力（Multi-Head Attention, MHA）通过**多个独立的注意力头**来处理输入信息，每个头使用不同的 `query`, `key`, `value` 线性投影，从而提升模型的表达能力。

MHA 计算流程如下：
1. 对输入 `X` 进行 `W_q`, `W_k`, `W_v` 线性变换，得到 `Q, K, V`；
2. 对 `Q, K, V` 进行 `view` 和 `transpose` 操作，确保多头并行计算；
3. 计算 `QK^T / \sqrt{d_k}`，然后进行 Masking 和 Softmax 归一化；
4. 计算 `Attention(Q, K, V) = \text{Softmax}(QK^T / \sqrt{d_k}) \times V`；
5. 将所有注意力头的输出拼接后，经过 `W_o` 投影得到最终输出。

通过 MHA，我们能够让 Transformer **关注不同位置的不同信息**，提升特征捕捉能力。

---

