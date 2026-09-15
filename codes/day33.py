import torch
from torch import nn

model = nn.Linear(3, 2)
bias_before = model.bias.detach().clone()

checkpoint = {"weight": torch.full((2, 3), 7.0)}
result = model.load_state_dict(checkpoint, strict=False)

print("加载结果：", result)
print("weight：", model.weight)
print("bias 是否保持原值：", torch.equal(model.bias, bias_before))
