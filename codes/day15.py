import torch
from torch.nn.utils import clip_grad_norm_

p1 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
p2 = torch.nn.Parameter(torch.tensor([0.0, 12.0]))

p1.grad = torch.tensor([3.0, 4.0])
p2.grad = torch.tensor([0.0, 12.0])

before = torch.sqrt(
    (p1.grad ** 2).sum() + (p2.grad ** 2).sum()
)

returned_norm = clip_grad_norm_(
    [p1, p2],
    max_norm=5.0
)

after = torch.sqrt(
    (p1.grad ** 2).sum() + (p2.grad ** 2).sum()
)

print("裁剪前总范数：", before)
print("函数返回范数：", returned_norm)
print("裁剪后 p1.grad：", p1.grad)
print("裁剪后 p2.grad：", p2.grad)
print("裁剪后总范数：", after)
