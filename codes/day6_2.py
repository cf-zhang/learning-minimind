import torch

torch.set_printoptions(precision=4, sci_mode=False) 
# 4 个 token，每个 token 当前有一个 3 维输入向量 h
h = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 2.0, 1.0],
])

# 三套不同的 Linear 权重：3 维输入 -> 2 维 Q/K/V
Wq = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])

Wk = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
])

Wv = torch.tensor([
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

# 1. 同一组输入向量分别经过三套 Linear
Q = h @ Wq.T
K = h @ Wk.T
V = h @ Wv.T

# 2. Q 与 K 点积，并除以 sqrt(head_dim)
head_dim = Q.shape[-1]
scores = (Q @ K.T) / (head_dim ** 0.5)

# 3. 遮住右上方的未来位置
future_mask = torch.triu(
    torch.ones(4, 4, dtype=torch.bool),
    diagonal=1
)

masked_scores = scores.masked_fill(
    future_mask,
    float("-inf")
)

# 4. 每一行分别做 Softmax
weights = torch.softmax(masked_scores, dim=-1)


# 5. 每个读取者按自己的比例合并所有 V
result = weights @ V

print("Q shape：", Q.shape)
print("K shape：", K.shape)
print("V shape：", V.shape)
print("scores shape：", scores.shape)
print("result shape：", result.shape)

print("\nQ：")
print(Q)

print("\nK：")
print(K)

print("\nV：")
print(V)

print("\nCausal Mask（True 表示未来位置）：")
print(future_mask)

print("\nMask 后的分数：")
print(masked_scores)

print("\n读取比例：")
print(weights)

print("\n每一行比例之和：")
print(weights.sum(dim=-1))

print("\n每个 token 汇总读到的信息：")
print(result)








