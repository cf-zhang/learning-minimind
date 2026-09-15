import torch

angles = torch.tensor([[0.1, 0.2]])

cos_half = torch.cos(angles)
sin_half = torch.sin(angles)

cos_full = torch.cat([cos_half, cos_half], dim=-1)
sin_full = torch.cat([sin_half, sin_half], dim=-1)

print("角度：", angles)
print("cos 半表：", cos_half)
print("sin 半表：", sin_half)
print("cos 完整表：", cos_full)
print("sin 完整表：", sin_full)

print("cos 半表 shape：", cos_half.shape)
print("cos 完整表 shape：", cos_full.shape)

import torch

def rotate_half(x):
    return torch.cat(
        (-x[..., x.shape[-1] // 2:],
          x[..., :x.shape[-1] // 2]),
        dim=-1
    )
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

print("原向量：", x)
print("rotate_half 后：", rotate_half(x))




import math
import torch

def rotate_half(x):
    return torch.cat(
        (-x[..., x.shape[-1] // 2:],
          x[..., :x.shape[-1] // 2]),
        dim=-1
    )

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

angles = torch.tensor([[0.0, math.pi / 2]])

cos_half = torch.cos(angles)
sin_half = torch.sin(angles)

cos_full = torch.cat([cos_half, cos_half], dim=-1)
sin_full = torch.cat([sin_half, sin_half], dim=-1)

rotated = (
    x * cos_full
    + rotate_half(x) * sin_full
)

print("cos 完整表：", cos_full)
print("sin 完整表：", sin_full)
print("x * cos_full: ", x * cos_full)
# print("x * cos_full: ", x * cos_full)
print("rotate_half(x)：", rotate_half(x))
print("完整旋转结果：", rotated)























