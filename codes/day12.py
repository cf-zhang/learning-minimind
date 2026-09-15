import torch
import torch.nn.functional as F

x = torch.tensor([[-2., -1., 0., 1., 2.]])
y = F.silu(x)

print("x：", x)
print("SiLU(x)：", y)
print("x shape：", x.shape)
print("y shape：", y.shape)
