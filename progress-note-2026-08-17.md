# 阶段整理：Backward 阶段整理与续学入口

- 日期：2026-08-17
- 学习阶段：神经网络训练的最小前置知识
- 今日主题：整理 Day 003-004 学习记录，固定恢复点
- 当前状态：文档整理完成，不计入新增知识进度

> 归档说明：本文件是跨日恢复快照，不是独立课程，也不改变 Day 编号。完整阅读顺序和状态规则见 [学习档案总索引](README.md)。恢复快照保留的价值是：即使学习中断，也能准确知道哪些步骤由学习者亲自完成、下一次只需从哪里继续。

> 时间边界：下文“尚未完成”描述的是 2026-08-17 整理当时的状态。这些项目后来已经在 [Day 005](day-005-2026-08-20.md) 完成；保留原快照是为了呈现真实跨日恢复过程。

## 一、今天做了什么

今天没有开展新的模型知识或代码实验。按照学习者要求，对此前 Backward 学习过程进行整理，避免跨日后丢失上下文，也不把未实践内容标记为完成。

详细实验数据仍保存在：

```text
learning-journal/day-003-2026-08-13.md
learning-journal/day-004-2026-08-14.md
```

## 二、当前已经掌握的学习链

### 1. 从预测到 loss

最小模型：

```text
x = 2
target = 10
weight = 3
prediction = x * weight = 6
error = prediction - target = -4
loss = error^2 = 16
```

已经理解不能直接最小化带符号 error：负误差可能通过越来越负获得更小数值，多个样本的正负 error 也可能相互抵消。平方 loss 消除正负并更强地惩罚大误差。

### 2. 从 loss 到梯度

在 `weight=3` 处：

```text
d(loss)/d(error) = 2 * error = -8
d(error)/d(weight) = x = 2
d(loss)/d(weight) = -8 * 2 = -16
```

已经理解梯度是当前位置的局部坡度：符号给出 loss 增长方向，绝对值反映局部变化速度。真实训练不知道最低点位置，而是利用梯度选择下降方向。

### 3. grad_fn 与计算关系

学习者实际观察：

```text
weight.grad_fn = None
prediction.grad_fn = MulBackward0
error.grad_fn = SubBackward0
loss.grad_fn = PowBackward0
```

对应：

```text
weight -> 乘法 -> prediction -> 减法 -> error -> 平方 -> loss
```

`grad_fn` 表示中间结果由什么可追踪运算产生，以及反向经过该步骤时使用什么规则；它不是梯度数值，也不表示 backward 已经执行。

### 4. backward 与参数更新

学习者实际验证：

```text
backward 前：weight=3，weight.grad=None
loss.backward()
backward 后：weight=3，weight.grad=-16
```

结论：

```text
backward 计算并存储梯度
backward 不修改参数
```

手工更新：

```text
weight = weight - 0.1 * weight.grad
       = 3 - 0.1 * (-16)
       = 4.6
```

更新后：

```text
prediction=9.2
error=-0.8
loss=0.64
```

loss 从 16 降到 0.64，验证更新方向正确。

### 5. torch.no_grad()

已经理解参数更新是训练控制动作，不属于模型 forward。`torch.no_grad()` 在更新时临时停止建立计算图，但：

```text
不会永久关闭 requires_grad
不会自动清空 weight.grad
不会阻止下一轮 forward 产生新的 grad_fn
```

## 三、尚未完成的内容

### Backward 主题

```text
[待实践] 梯度默认累加
[待理解] 为什么 PyTorch 选择累加
[待实践] 梯度清零
[待学习] optimizer.zero_grad()
[待学习] optimizer.step()
```

### 模型主题

```text
[待学习] Attention 是什么
[待学习] Attention Head 是什么
[待学习] MiniMindBlock
[待学习] RMSNorm 与残差连接
[后续] Q/K/V、Softmax、GQA、RoPE、KV Cache
```

## 四、两个原始遗留问题的状态

问题一：

> `Backward` 表示 PyTorch 记录反向关系并计算 weight、bias 如何调整，这句话怎么理解？

当前状态：主体已经通过一个参数的模型实践完成；还差梯度累加、清零和 optimizer，完成后正式收口。

问题二：

> Attention Head 是什么？

当前状态：尚未正式开始。后续必须从“普通 Linear 不让 token 交换信息”的已知结论出发，先解释 Attention 的任务，再解释 Head，不能直接抛出 Q/K/V。

## 五、下一次唯一入口

下一次先完成梯度累加实验，不同时进入 Attention。

重新创建状态：

```python
import torch

x = torch.tensor(2.0)
target = torch.tensor(10.0)
weight = torch.tensor(3.0, requires_grad=True)

prediction = x * weight
loss = (prediction - target) ** 2
loss.backward()

with torch.no_grad():
    weight -= 0.1 * weight.grad

new_prediction = x * weight
new_loss = (new_prediction - target) ** 2

print(weight)
print(weight.grad)
print(new_loss)
```

预期恢复：

```text
weight=4.6
weight.grad=-16
new_loss=0.64
```

然后执行：

```python
new_loss.backward()
print(weight.grad)
```

预期观察旧梯度 `-16` 与新梯度约 `-3.2` 累加为约 `-19.2`。得到学习者实际输出后，再解释累加原因和梯度清零。

## 六、记录原则

- 今天只做文档整理，不记作新知识掌握。
- Day 004 未完成的项目保持未完成状态。
- 后续仍采用一次一个小关卡的交互方式。
- 不因跨日而跳过实际实践或重复宣布完成。
