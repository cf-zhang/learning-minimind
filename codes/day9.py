import torch


torch.set_printoptions(precision=4, sci_mode=False)

# 当前读取者“苹果”的 Q
q_apple = torch.tensor([1.0, 0.0])

# 甲、乙各自的 K
k_jia = torch.tensor([1.0, 0.0])
k_yi = torch.tensor([0.0, 1.0])

# 甲、乙各自携带的 V
v_jia = torch.tensor([10.0, 0.0])
v_yi = torch.tensor([0.0, 20.0])

# 顺序一：甲 乙 苹果
K_jia_yi = torch.stack([k_jia, k_yi])
V_jia_yi = torch.stack([v_jia, v_yi])


scores_jia_yi = (q_apple @ K_jia_yi.T) / (2 ** 0.5)
weights_jia_yi = torch.softmax(scores_jia_yi, dim=-1)
result_jia_yi = weights_jia_yi @ V_jia_yi

# 顺序二：乙 甲 苹果
# K 和 V 要跟随所属 token 一起交换
K_yi_jia = torch.stack([k_yi, k_jia])
V_yi_jia = torch.stack([v_yi, v_jia])

scores_yi_jia = (q_apple @ K_yi_jia.T) / (2 ** 0.5)
weights_yi_jia = torch.softmax(scores_yi_jia, dim=-1)
result_yi_jia = weights_yi_jia @ V_yi_jia

print("甲乙顺序的分数：", scores_jia_yi)
print("甲乙顺序的比例：", weights_jia_yi)
print("甲乙顺序的结果：", result_jia_yi)

print("乙甲顺序的分数：", scores_yi_jia)
print("乙甲顺序的比例：", weights_yi_jia)
print("乙甲顺序的结果：", result_yi_jia)

print("交换顺序后结果是否相同：", torch.allclose(
    result_jia_yi,
    result_yi_jia
))



