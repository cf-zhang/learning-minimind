import torch

x = torch.tensor(2.0)
target = torch.tensor(10.0)
weight = torch.tensor(
    3.0,
    requires_grad=True
)
prediction = x * weight
error = prediction - target
loss = error ** 2

print("weight：", weight)
print("prediction：", prediction)
print("error：", error)
print("loss：", loss)

print("weight.grad_fn：", weight.grad_fn)
print("prediction.grad_fn：", prediction.grad_fn)
print("error.grad_fn：", error.grad_fn)
print("loss.grad_fn：", loss.grad_fn)

print("backward 前 weight：", weight)
print("backward 前 weight.grad：", weight.grad)

loss.backward()

print("backward 后 weight：", weight)
print("backward 后 weight.grad：", weight.grad)

learning_rate = 0.1
with torch.no_grad():
      weight -= learning_rate * weight.grad
print("更新后的 weight：", weight)
print("原来的 weight.grad：", weight.grad)

new_prediction = x * weight
new_error = new_prediction - target
new_loss = new_error ** 2

print("新的 prediction：", new_prediction)
print("新的 error：", new_error)
print("新的 loss：", new_loss)

print("第二次 backward 前：", weight.grad)

new_loss.backward()

print("第二次 backward 后：", weight.grad)

weight.grad = None

print("清零后的 weight.grad：", weight.grad)
print("weight 本身：", weight)


prediction_after_clear = x * weight
loss_after_clear = (prediction_after_clear - target) ** 2

loss_after_clear.backward()

print("重新 backward 后的 weight.grad：", weight.grad)



















