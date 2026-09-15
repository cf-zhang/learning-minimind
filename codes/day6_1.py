import torch
q = torch.tensor([1.0, 2.0])
keys = torch.tensor([
    [2.0, 3.0],  # “吃了”的 K
    [4.0, 1.0],  # “红”的 K
])
values = torch.tensor([
    [10.0, 0.0],  # “吃了”的 V
    [0.0, 10.0],  # “红”的 V
])

scores = q @ keys.T

weights = torch.softmax(scores, dim=0)
result = weights @ values
print("匹配分数：", scores)
print("读取比例：", weights)
print("比例总和：", weights.sum())
print("读取结果：", result)

small_scores = torch.tensor([2.0, 1.0])
large_scores = torch.tensor([20.0, 10.0])

small_weights = torch.softmax(small_scores, dim=0)
large_weights = torch.softmax(large_scores, dim=0)

print("较小分数的 Softmax：", small_weights)
print("较大分数的 Softmax：", large_weights)


for dimension in [2, 8, 32, 128]:
    q = torch.ones(dimension)
    k = torch.ones(dimension)
    raw_score = q @ k
    scaled_score = raw_score / (dimension ** 0.5)

    print(
        "维度 =", dimension,
        "原始点积 =", raw_score.item(),
        "除以 sqrt(d) 后 =", scaled_score.item()
    )

import torch

torch.manual_seed(0)

sample_count = 10000

for dimension in [2, 8, 32, 128]:
    q = torch.randn(sample_count, dimension)
    k = torch.randn(sample_count, dimension)
    raw_scores = (q * k).sum(dim=1)
    scaled_scores = raw_scores / (dimension ** 0.5)

    print(
        "维度 =", dimension,
        "原始分数的标准差 =", raw_scores.std().item(),
        "缩放后分数的标准差 =", scaled_scores.std().item()
    )


import torch

scores = torch.tensor([2.0, 1.0, 3.0, 9.0])

weights_without_mask = torch.softmax(scores, dim=0)

masked_scores = scores.clone()
masked_scores[3] = float("-inf")

weights_with_mask = torch.softmax(masked_scores, dim=0)

print("原始分数：", scores)
print("不遮罩的读取比例：", weights_without_mask)
print("遮罩后的分数：", masked_scores)
print("遮罩后的读取比例：", weights_with_mask)
print("遮罩后比例总和：", weights_with_mask.sum())



