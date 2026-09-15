import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=-1)
# 输出: [0.659, 0.242, 0.099]

log_probs = F.log_softmax(logits, dim=-1)
print(probs)
print(log_probs)
exp = torch.exp(log_probs)
print(exp)
