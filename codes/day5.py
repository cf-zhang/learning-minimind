import torch

x = torch.tensor(2.0)
target = torch.tensor(10.0)
weight = torch.tensor(
    3.0,
    requires_grad=True
)
optimizer = torch.optim.SGD(
    [weight],
    lr=0.1
)

for step in range(5):
    prediction = x * weight
    loss = (prediction - target) ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        "step =", step,
        "weight =", weight.item(),
        "prediction =", prediction.item(),
        "loss =", loss.item()
    )











