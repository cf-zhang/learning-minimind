import torch
from torch import nn

torch.manual_seed(7)

batch = 1
sequence = 3
hidden_size = 8
num_heads = 2
head_dim = 4

# 模拟第一层收到的 Embedding 输出
h = torch.arange(
    batch * sequence * hidden_size,
    dtype=torch.float32
).reshape(batch, sequence, hidden_size)


# 三套不同的 Linear
q_proj = nn.Linear(8, 8, bias=False)
k_proj = nn.Linear(8, 8, bias=False)
v_proj = nn.Linear(8, 8, bias=False)

# 先让完整 h 分别经过三套投影
q_projected = q_proj(h)
k_projected = k_proj(h)
v_projected = v_proj(h)

# 再分别拆成两个 Head
q_heads = q_projected.reshape(
    batch, sequence, num_heads, head_dim
).transpose(1, 2)

k_heads = k_projected.reshape(
    batch, sequence, num_heads, head_dim
).transpose(1, 2)

v_heads = v_projected.reshape(
    batch, sequence, num_heads, head_dim
).transpose(1, 2)


print("h shape：", h.shape)
print("q_projected shape：", q_projected.shape)
print("k_projected shape：", k_projected.shape)
print("v_projected shape：", v_projected.shape)

print("q_heads shape：", q_heads.shape)
print("k_heads shape：", k_heads.shape)
print("v_heads shape：", v_heads.shape)

print("\n第一个 token 的原始 h：")
print(h[0, 0])

print("\n第一个 token 投影后的完整 Q：")
print(q_projected[0, 0])

print("\n第一个 token 拆成两个 Q Head：")
print("Head 0：", q_heads[0, 0, 0])
print("Head 1：", q_heads[0, 1, 0])


# 把 Head 重新拼回去，验证拆分没有修改数值
q_restored = q_heads.transpose(1, 2).reshape(
    batch, sequence, hidden_size
)

print(
    "\n投影前 h 和投影后 Q 是否相同：",
    torch.allclose(h, q_projected)
)

print(
    "拆 Head 后再拼回，是否等于 q_projected：",
    torch.allclose(q_restored, q_projected)
)

print(
    "Q 和 K 是否相同：",
    torch.allclose(q_projected, k_projected)
)




# 取出 Head 0 和 Head 1 的 Q、K
head_0_q = q_heads[:, 0, :, :]
head_0_k = k_heads[:, 0, :, :]

head_1_q = q_heads[:, 1, :, :]
head_1_k = k_heads[:, 1, :, :]

print("Head 0 的 Q shape：", head_0_q.shape)
print("Head 0 的 K shape：", head_0_k.shape)
print("Head 1 的 Q shape：", head_1_q.shape)
print("Head 1 的 K shape：", head_1_k.shape)

# 每个 Head 在自己的 3 个 token 内计算分数
head_0_scores = head_0_q @ head_0_k.transpose(-2, -1)
head_1_scores = head_1_q @ head_1_k.transpose(-2, -1)

print("Head 0 的分数 shape：", head_0_scores.shape)
print("Head 1 的分数 shape：", head_1_scores.shape)

print("Head 0 的分数：")
print(head_0_scores)

print("Head 1 的分数：")
print(head_1_scores)


head_dim = head_0_q.shape[-1]

# 3 个 token，只允许看自己和过去
causal_mask = torch.triu(
    torch.ones(3, 3, dtype=torch.bool),
    diagonal=1
)

# Head 0
head_0_scaled = head_0_scores / (head_dim ** 0.5)
head_0_masked = head_0_scaled.masked_fill(
    causal_mask,
    float("-inf")
)
head_0_weights = torch.softmax(
    head_0_masked,
    dim=-1
)

# Head 1
head_1_scaled = head_1_scores / (head_dim ** 0.5)
head_1_masked = head_1_scaled.masked_fill(
    causal_mask,
    float("-inf")
)
head_1_weights = torch.softmax(
    head_1_masked,
    dim=-1
)

print("Causal Mask：")
print(causal_mask)

print("\nHead 0 的读取比例：")
print(head_0_weights)
print("\nHead 1 的读取比例：")
print(head_1_weights)

print("\nHead 0 每行比例之和：")
print(head_0_weights.sum(dim=-1))
print("\nHead 1 每行比例之和：")
print(head_1_weights.sum(dim=-1))

head_0_v = v_heads[:, 0, :, :]
head_1_v = v_heads[:, 1, :, :]

head_0_result = head_0_weights @ head_0_v
head_1_result = head_1_weights @ head_1_v

print("Head 0 的 V shape：", head_0_v.shape)
print("Head 1 的 V shape：", head_1_v.shape)

print("Head 0 的结果 shape：", head_0_result.shape)
print("Head 1 的结果 shape：", head_1_result.shape)

print("\nHead 0 的读取结果：")
print(head_0_result)

print("\nHead 1 的读取结果：")
print(head_1_result)


all_head_results = torch.stack(
    [head_0_result, head_1_result],
    dim=1
)

print("两个 Head 放在一起后的 shape：",
      all_head_results.shape)
combined = all_head_results.transpose(1, 2).reshape(
    batch,
    sequence,
    hidden_size
)

print("按 token 拼接后的 shape：", combined.shape)

print("\n按 token 拼接后的结果：")
print(combined)

o_proj = nn.Linear(
    hidden_size,
    hidden_size,
    bias=False
)

attention_output = o_proj(combined)

print("combined shape：", combined.shape)
print("attention_output shape：",
      attention_output.shape)

print("\n第一个 token 拼接后的结果：")
print(combined[0, 0])

print("\n第一个 token 经过 o_proj 后：")
print(attention_output[0, 0])

print(
    "\n经过 o_proj 前后数值是否相同：",
    torch.allclose(
        combined,
        attention_output
    )
)






