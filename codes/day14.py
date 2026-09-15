import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.0]])
target = torch.tensor([0])

loss = F.cross_entropy(logits, target)
print(loss)

target = torch.tensor([2])
loss = F.cross_entropy(logits, target)
print(loss)

logits = torch.tensor([
    [2.0, 1.0, 0.0],
    [0.0, 1.0, 2.0],
])
targets = torch.tensor([0, -100])

loss = F.cross_entropy(
    logits,
    targets,
    ignore_index=-100
)
print(loss)

single_loss = F.cross_entropy(
    logits[:1],
    targets[:1]
)
print(single_loss)




import torch
import torch.nn.functional as F

logits = torch.tensor([
    [2.0, 1.0, 0.0],
    [0.0, 1.0, 2.0],
])

targets = torch.tensor([0, -100])

loss_with_ignore = F.cross_entropy(
    logits,
    targets,
    ignore_index=-100
)

loss_first_only = F.cross_entropy(
    logits[:1],
    torch.tensor([0])
)

loss_both = F.cross_entropy(
    logits,
    torch.tensor([0, 1])
)

print("忽略第二行：", loss_with_ignore)
print("只算第一行：", loss_first_only)
print("两行都算：", loss_both)


