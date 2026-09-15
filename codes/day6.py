import torch

scores = torch.tensor([1.0, 3.0, 1.0, 2.0])

# 第一步：计算 e^分数
positive_scores = torch.exp(scores)

# 第二步：分别除以总和
manual_weights = positive_scores / positive_scores.sum()

# PyTorch 直接提供的 Softmax
torch_weights = torch.softmax(scores, dim=0)
print("匹配分数：", scores)
print("e^分数：", positive_scores)
print("e^分数的总和：", positive_scores.sum())
print("手工 Softmax：", manual_weights)
print("PyTorch Softmax：", torch_weights)
print("读取比例总和：", torch_weights.sum())
print("两种算法是否一致：", torch.allclose(manual_weights, torch_weights))




