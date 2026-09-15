import torch

q_projected = torch.arange(24).reshape(1, 3, 8)

print("拆分前 Q：")
print(q_projected)
print("拆分前 shape：", q_projected.shape)

q_split = q_projected.reshape(1, 3, 2, 4)

print("直接整理后 shape：", q_split.shape)

q_heads = q_split.transpose(1, 2)

print("交换 sequence 和 head 后 shape：", q_heads.shape)
print("整理后的 Q：")
print(q_heads)


head_0_q = q_heads[:, 0, :, :]
head_1_q = q_heads[:, 1, :, :]

print("Head 0 的 Q：")
print(head_0_q)
print("Head 0 shape：", head_0_q.shape)

print("Head 1 的 Q：")
print(head_1_q)
print("Head 1 shape：", head_1_q.shape)

print("Head 0 和 Head 1 是否共享数值：",
      torch.equal(head_0_q, head_1_q))

