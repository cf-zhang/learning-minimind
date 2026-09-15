import torch

dim = 4
rope_base = 100.0
group_starts = torch.arange(0, dim, 2).float()
print("二维组的起始维度：", group_starts)
print("除以 dim 后：", group_starts / dim)
freqs = 1.0 / (
    rope_base ** (group_starts / dim)
)

print("频率：", freqs)


