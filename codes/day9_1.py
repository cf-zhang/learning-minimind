import torch

torch.set_printoptions(precision=4, sci_mode=False)

q = torch.tensor([1.0, 0.0])
K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

V_original = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
])

# Q/K 完全不变，所以分数和读取比例固定
scores = q @ K.T
weights = torch.softmax(scores, dim=-1)

# 只把第二个 V 旋转 90°：
# [x, y] -> [-y, x]
V_rotated = V_original.clone()
x, y = V_rotated[1]
V_rotated[1] = torch.tensor([-y, x])

result_original = weights @ V_original
result_rotated = weights @ V_rotated

print("分数：", scores)
print("读取比例：", weights)
print("原始 V：", V_original)
print("只旋转第二个 V 后：", V_rotated)
print("原始读取结果：", result_original)
print("旋转 V 后的读取结果：", result_rotated)
print("分数是否相同：", torch.allclose(scores, scores))
print("比例是否相同：", torch.allclose(weights, weights))
print("最终结果是否相同：", torch.allclose(
    result_original,
    result_rotated
))

