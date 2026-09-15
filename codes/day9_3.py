import math
import torch

torch.set_printoptions(precision=4, sci_mode=False)

def rotate_2d(vector, position, step):
    angle = position * step
    cos_value = torch.cos(torch.tensor(angle))
    sin_value = torch.sin(torch.tensor(angle))

    x, y = vector

    return torch.stack([
        x * cos_value - y * sin_value,
        x * sin_value + y * cos_value,
    ])

raw_q = torch.tensor([1.0, 0.0])
raw_k = torch.tensor([1.0, 1.0])

# 每前进一个位置旋转 90°，也就是 π/2 弧度
step = math.pi / 2

# 情况 A：Q 在位置 1，K 在位置 0
q_position_1 = rotate_2d(raw_q, position=1, step=step)
k_position_0 = rotate_2d(raw_k, position=0, step=step)
score_a = q_position_1 @ k_position_0

# 情况 B：Q 在位置 2，K 在位置 1
q_position_2 = rotate_2d(raw_q, position=2, step=step)
k_position_1 = rotate_2d(raw_k, position=1, step=step)
score_b = q_position_2 @ k_position_1

print("原始 Q：", raw_q)
print("原始 K：", raw_k)

print("情况 A 的旋转后 Q：", q_position_1)
print("情况 A 的旋转后 K：", k_position_0)
print("情况 A 的点积：", score_a)

print("情况 B 的旋转后 Q：", q_position_2)
print("情况 B 的旋转后 K：", k_position_1)
print("情况 B 的点积：", score_b)

print("两个点积是否相同：", torch.allclose(
    score_a,
    score_b,
    atol=1e-6,
))

print("Q 旋转前后长度是否相同：", torch.allclose(
    torch.norm(raw_q),
    torch.norm(q_position_2),
    atol=1e-6,
))





