import torch
from torch import nn
class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)
    def forward(self, x):
        y = self.layer(x)
        return y
      
model = SmallModel()
print(model)
          
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

x = torch.tensor([
    [1.0, 2.0, 3.0]
])

print("输入：", x)
print("输入 shape：", x.shape)

y = model(x)

print("输出：", y)
print("输出 shape：", y.shape)

weight = model.layer.weight
bias = model.layer.bias
print("weight：")
print(weight)

print("bias：")
print(bias)
with torch.no_grad():
    first_output = (
        x[0, 0] * weight[0, 0]
        + x[0, 1] * weight[0, 1]
        + x[0, 2] * weight[0, 2]
        + bias[0]
    )

    second_output = (
        x[0, 0] * weight[1, 0]
        + x[0, 1] * weight[1, 1]
        + x[0, 2] * weight[1, 2]
        + bias[1]
    )

print(first_output)
print(second_output)

manual_y = torch.stack([first_output, second_output]).unsqueeze(0)

print("手工计算：", manual_y)
print("模型计算：", y)
print("是否一致：", torch.allclose(manual_y, y))

with torch.no_grad():
    matrix_y = x @ weight.T + bias
print("矩阵公式：", matrix_y)
print("是否一致：", torch.allclose(matrix_y, y))




token_vectors = torch.tensor([
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
])
print("输入：")
print(token_vectors)
print("输入 shape：", token_vectors.shape)

with torch.no_grad():
    token_outputs = model(token_vectors)
print("输出：")
print(token_outputs)
print("输出 shape：", token_outputs.shape)


with torch.no_grad():
    first_token_output = model(token_vectors[:, 0, :])
    second_token_output = model(token_vectors[:, 1, :])

print("一起计算的第一个 token：", token_outputs[:, 0, :])
print("单独计算的第一个 token：", first_token_output)
print("一起计算的第二个 token：", token_outputs[:, 1, :])
print("单独计算的第二个 token：", second_token_output)

print(torch.allclose(
    token_outputs[:, 0, :],
    first_token_output
))

print(torch.allclose(
    token_outputs[:, 1, :],
    second_token_output
))


a = torch.tensor([1.0, 0.0])
b = torch.tensor([1.0, 0.0])
c = torch.tensor([0.0, 1.0])
d = torch.tensor([-1.0, 0.0])

print("a dot b =", torch.dot(a, b))
print("a dot c =", torch.dot(a, c))
print("a dot d =", torch.dot(a, d))


query = torch.tensor([1.0, 0.0])

eat_key = torch.tensor([0.9, 0.1])
publish_key = torch.tensor([0.1, 0.9])

eat_score = torch.dot(query, eat_key)
publish_score = torch.dot(query, publish_key)

print("与吃了的分数：", eat_score)
print("与发布的分数：", publish_score)

keys = torch.stack([
    eat_key,
    publish_key
])

print("keys：")
print(keys)
print("keys shape：", keys.shape)


scores = query @ keys.T

print("一次计算的分数：", scores)
print("scores shape：", scores.shape)

























