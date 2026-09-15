import torch

Q = torch.tensor([
    [
        # Q Head 0：两个 token 各自的 Q
        [
            [0.0, 1.0],  # Q0_吃了
            [1.0, 2.0],  # Q0_苹果
        ],

        # Q Head 1：两个 token 各自的 Q
        [
            [1.0, 0.0],  # Q1_吃了
            [3.0, 1.0],  # Q1_苹果
        ],
    ]
])

K = torch.tensor([
    [
        # 共享的 K Head 0：两个 token 各自的 K
        [
            [2.0, 0.0],  # K0_吃了
            [0.0, 2.0],  # K0_苹果
        ]
    ]
])

V = torch.tensor([
    [
        # 共享的 V Head 0：两个 token 各自的 V
        [
            [10.0, 0.0],  # V0_吃了
            [0.0, 10.0],  # V0_苹果
        ]
    ]
])

print("Q shape：", Q.shape)
print("K shape：", K.shape)
print("V shape：", V.shape)

print("\n苹果的两个 Q：")
print("Q0_苹果：", Q[0, 0, 1])
print("Q1_苹果：", Q[0, 1, 1])

print("\n共享 K Head 中的两个 K：")
print("K0_吃了：", K[0, 0, 0])
print("K0_苹果：", K[0, 0, 1])




repeat_count = 2

K_repeated = K.repeat_interleave(
    repeat_count,
    dim=1
)

V_repeated = V.repeat_interleave(
    repeat_count,
    dim=1
)

print("重复前 K shape：", K.shape)
print("重复后 K shape：", K_repeated.shape)

print("重复前 V shape：", V.shape)
print("重复后 V shape：", V_repeated.shape)

print("\n重复后的 K Head 0：")
print(K_repeated[0, 0])

print("\n重复后的 K Head 1：")
print(K_repeated[0, 1])

print(
    "\n两个 K Head 的内容是否相同：",
    torch.equal(
        K_repeated[:, 0],
        K_repeated[:, 1]
    )
)

print(
    "两个 V Head 的内容是否相同：",
    torch.equal(
        V_repeated[:, 0],
        V_repeated[:, 1]
    )
)







head_dim = 2

# 1. 两个 Q Head 分别与对应位置的共享 K 计算
scores = (
    Q @ K_repeated.transpose(-2, -1)
) / (head_dim ** 0.5)

# 2. 两个 token 的 Causal Mask
causal_mask = torch.triu(
    torch.ones(2, 2, dtype=torch.bool),
    diagonal=1
)

masked_scores = scores.masked_fill(
    causal_mask,
    float("-inf")
)

# 3. 每个 Q Head、每个读取者分别做 Softmax
weights = torch.softmax(
    masked_scores,
    dim=-1
)

# 4. 两个 Q Head 分别读取重复后的共享 V
head_results = weights @ V_repeated

print("scores shape：", scores.shape)
print("weights shape：", weights.shape)
print("head_results shape：", head_results.shape)

print("\n两个 Q Head 的匹配分数：")
print(scores)

print("\n两个 Q Head 的读取比例：")
print(weights)

print("\n每行比例之和：")
print(weights.sum(dim=-1))

print("\n两个 Q Head 的读取结果：")
print(head_results)

print("\n苹果在 Q Head 0 中的分数：")
print(scores[0, 0, 1])

print("苹果在 Q Head 1 中的分数：")
print(scores[0, 1, 1])

print("\n苹果在 Q Head 0 中的读取结果：")
print(head_results[0, 0, 1])

print("苹果在 Q Head 1 中的读取结果：")
print(head_results[0, 1, 1])
















