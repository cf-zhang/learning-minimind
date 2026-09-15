# 阶段一小结：从零基础到读通 MiniMind 原生推理链

- 覆盖范围：Day 001 至 Day 012
- 阶段主题：大模型基础、Transformer 核心结构与 MiniMind 原生推理源码
- 阶段状态：已完成
- 主仓库：minimind
- 学习方式：仓库驱动、问题触发、交互验收
- 下一阶段：闭合并验证完整推理数据流，然后进入训练链路

## 一、这一阶段解决了什么问题

学习开始时，学习者具备编程基础，但没有系统接触数学、神经网络和
大模型。阶段一没有先学习一整套抽象数学课程，而是从 MiniMind 的真实
运行和源码出发，逐步回答下面这条主线中的问题：

    一段用户文字
    -> 为什么模型不能直接处理文字
    -> 文字怎样变成 token ID
    -> token ID 怎样变成向量
    -> 向量怎样经过神经网络
    -> token 怎样有选择地读取上下文
    -> 模型怎样理解 token 的位置
    -> 多层 Transformer 怎样加工表示
    -> hidden state 怎样变成词表候选分数
    -> 怎样从候选分数中选择下一个 token
    -> 为什么生成答案需要不断循环
    -> KV Cache 怎样避免重复计算
    -> token ID 怎样重新变成回答文字

到 Day 012 结束时，已经能够沿真实调用顺序解释 MiniMind 原生模型从
eval_llm.py 启动，到生成并返回 token 序列的主干代码。

这个阶段的成果不是“记住所有术语”，而是建立了一条可以回到源码定位
每个术语的因果链。

## 二、阶段学习地图

| 学习日 | 核心问题 | 阶段成果 |
| --- | --- | --- |
| Day 001 | 文字怎样进入模型 | 跑通模型，理解 Tokenizer、token ID、Embedding 和上下文化 |
| Day 002 | 神经网络怎样加工向量 | 理解 nn.Module、Linear、矩阵乘法、点积和多 token shape |
| Day 003 | 模型怎样知道自己错了 | 理解 prediction、target、error、loss 和梯度来源 |
| Day 004 | backward() 到底做什么 | 观察计算图、grad_fn、梯度和手工参数更新 |
| Day 005 | 参数怎样连续学习 | 理解梯度累加、清零、SGD 和最小训练循环 |
| Day 006 | token 怎样读取上下文 | 理解单 Head Attention、Q/K/V、Softmax、缩放和 Causal Mask |
| Day 007 | 为什么需要多个 Head | 理解 Q/K/V 投影、拆 Head、独立计算、拼接和 o_proj |
| Day 008 | 为什么 Q Head 比 K/V Head 多 | 理解 GQA、repeat_kv、参数量与 KV Cache 收益 |
| Day 009 | 相同内容怎样区分位置 | 理解 RoPE、二维旋转、频率表和相对位置 |
| Day 010 | 程序从哪里开始执行 | 沿 eval_llm.py 读完入口、配置、模板、编码和外层返回 |
| Day 011 | generate() 怎样组织生成 | 区分 generate() 与 forward()，定位生成准备和原生入口 |
| Day 012 | 原生 MiniMind 内部怎样完成推理 | 读通模型构造、Block、Attention、FeedForward、LM Head、采样和 Cache |

每天的完整互动、用户实践输出、错误纠正和恢复点仍保存在各 Day 文档；
本文件只负责把它们组织成一个连续章节。

## 三、文本进入模型：Tokenizer、token ID 与 Embedding

### 3.1 为什么需要 Tokenizer

模型内部只能进行数值计算，不能直接接收字符串。因此输入首先经过：

    字符串
    -> Tokenizer
    -> 一串整数 token ID

实际观察过：

    我正在学习MiniMind大模型！

它被编码为 11 个 token ID。Tokenizer 的切分单位不必等于汉字或单词，
例如“正在”“学习”“模型”可以各自成为一个 token，英文片段也可能按
子词切分。

token ID 只是词表中的整数下标，本身不表示语义大小：

    token ID 463 不是比 token ID 80 更有意义
    它们只是两行不同 Embedding 权重的位置

### 3.2 Embedding 是一次查表

真实模型的 Embedding 权重：

    embedding.weight.shape = [6400, 768]

含义：

    6400：词表里有 6400 个 token
    768：每个 token 初始表示为 768 个数

输入 11 个 token：

    input_ids.shape = [1, 11]

查表后：

    vectors.shape = [1, 11, 768]

实验验证：

    vectors[0, 0] == embedding.weight[463]

因此 Embedding 首先可以理解为：用 token ID 找到对应的 768 维可训练
向量。

### 3.3 相同 token 的初始向量相同，但后续表示不同

输入中两个字面相同的 M 拥有相同 token ID，因此刚查 Embedding 时两者
向量相同。经过第一层 Transformer 后，两个 M 的 hidden state 不再相同。

原因不是 Embedding 为同一个 token 保存了多份数据，而是 Transformer
根据它们所在位置和前文，为它们加入了不同上下文信息。

这是从“静态 token 向量”到“上下文化 token 表示”的第一次观察。

## 四、神经网络计算基础：Linear、矩阵与 shape

### 4.1 Linear 做了什么

最小实验：

    Linear(in_features=3, out_features=2, bias=True)

输入 x 为 [1,2,3]，权重 shape 为 [2,3]，偏置 shape 为 [2]。计算：

    output = x @ weight.T + bias

每个输出值都是输入的 3 个数与一行权重做点积，再加一个偏置：

    [1, 3] -> Linear(3, 2) -> [1, 2]

### 4.2 多 token 不会被 Linear 自动混在一起

输入：

    [batch, token, feature] = [1, 2, 3]

经过 Linear(3,2)：

    [1, 2, 3] -> [1, 2, 2]

Linear 独立处理每个 token 的最后一维。把两个 token 一起送入 Linear，
与分别送入再拼起来，结果一致。

由此建立通用规则：

    nn.Linear 只改变最后一维
    前面的 batch、sequence、head 等维度被保留

### 4.3 同一个 @，不同的数据含义

Linear 的手工公式与 Attention 的 Q/K 点积都使用 @，因为它们本质上都
使用矩阵乘法：

    Linear：token 特征 @ 可训练权重
    Attention：Query @ Key 转置

看源码时，必须同时关注张量代表什么和 shape 能否相乘。

## 五、模型怎样学习：loss、梯度与 SGD

### 5.1 最小预测问题

实验使用：

    prediction = weight * 2
    target = 10
    error = prediction - target
    loss = error ** 2

当 weight=3 时：

    prediction=6
    error=-4
    loss=16

loss 把“预测错了多少”变成一个需要尽量减小的标量。

### 5.2 梯度是当前位置的变化方向和速度

通过链式关系：

    loss 对 error 的变化
    × error 对 prediction 的变化
    × prediction 对 weight 的变化

得到：

    weight.grad = -16

负号不是说把 weight 改成负数，而是说明在当前位置增大 weight 会让
loss 下降。梯度下降：

    weight = weight - learning_rate * gradient

当 gradient 为负时，减去负数会使 weight 增大。

### 5.3 backward() 计算梯度，不直接改参数

执行 loss.backward() 后：

    weight 仍然是 3
    weight.grad 变成 -16

参数更新需要单独执行，通常由 optimizer 完成。手工更新后：

    weight：3 -> 4.6
    loss：16 -> 0.64

说明梯度给出的方向确实降低了 loss。

### 5.4 梯度默认累加

连续两次 backward()，如果没有清零：

    第一次后：-16
    第二次后：-19.2

正确训练循环需要：

    optimizer.zero_grad()
    -> forward
    -> loss
    -> loss.backward()
    -> optimizer.step()

五轮最小 SGD 实验中，weight 逐步接近 5，loss 逐步接近 0。阶段一已经
建立训练的最小闭环，但尚未进入 MiniMind 的真实训练脚本。

## 六、Attention：token 怎样选择性读取上下文

### 6.1 Attention 的基本问题

对当前 token 而言，并不是前文所有 token 都同样重要。Attention 解决：

    当前 token 应该从哪些位置读取信息
    每个位置读取多少
    读取的具体内容是什么

一条最小读取链：

    Query 与每个 Key 点积
    -> 匹配分数
    -> Softmax
    -> 读取比例
    -> 按比例加权汇总 Value
    -> 当前 token 的读取结果

### 6.2 Q、K、V 不是 Embedding 的三块数据

同一个 token 的 hidden state 分别经过三套 Linear：

    Q = q_proj(hidden_state)
    K = k_proj(hidden_state)
    V = v_proj(hidden_state)

可以先用三个角色理解：

    Q：当前 token 想找什么
    K：每个 token 能用什么特征被匹配
    V：匹配后真正被读取的内容

Q/K 决定读取比例，V 提供读取内容。

### 6.3 Softmax 只负责把分数变成读取比例

例如匹配分数：

    [1, 3, 1, 2]

Softmax：

    [0.0826, 0.6103, 0.0826, 0.2245]

比例总和为 1。Softmax 不产生 V，也不负责计算匹配分数，它只把已有分数
转换为可用于加权读取的比例。

### 6.4 为什么除以 sqrt(head_dim)

Q/K 维度越大，随机点积的波动通常越大，容易让 Softmax 过度尖锐。
实验观察：

    原始点积标准差随维度增长
    除以 sqrt(d) 后，标准差稳定在约 1

因此缩放分数：

    scores = Q @ K.T / sqrt(head_dim)

不是为了改变 shape，而是为了稳定数值尺度。

### 6.5 两种 Mask

Causal Mask：

    屏蔽未来 token
    保证第 i 个 token 只能读取自己和更早位置

attention_mask：

    屏蔽 padding 等无效 Key
    有效位置 1 -> 加 0
    无效位置 0 -> 加 -1e9

它们解决不同问题，不能混为一个 mask。

## 七、Multi-Head Attention 与 GQA

### 7.1 为什么拆多个 Head

MiniMind 的 hidden size 为 768，Q Head 数为 8：

    head_dim = 768 / 8 = 96

Q 投影后的 shape 变化：

    [batch, sequence, 768]
    -> [batch, sequence, 8, 96]
    -> [batch, 8, sequence, 96]

每个 Head 使用自己的 96 维子空间计算 Attention，可以学习不同的匹配和
读取方式。

Head 计算完后：

    [batch, 8, sequence, 96]
    -> transpose
    -> [batch, sequence, 8, 96]
    -> reshape
    -> [batch, sequence, 768]
    -> o_proj
    -> [batch, sequence, 768]

拼接只是放在一起，o_proj 才负责重新混合各 Head 的结果。

### 7.2 GQA 为什么只有 4 个 K/V Head

真实配置：

    Q Head：8
    K Head：4
    V Head：4

每两个 Q Head 共享一组 K/V。计算 Attention 前：

    K/V [batch, sequence, 4, 96]
    -> repeat_kv
    -> [batch, sequence, 8, 96]

repeat_kv 只是为了与 8 个 Q Head 对齐；被复制的两份 K/V 内容相同。

当前投影参数量：

    q_proj：768 × 768 = 589,824
    k_proj：384 × 768 = 294,912
    v_proj：384 × 768 = 294,912
    o_proj：768 × 768 = 589,824

相对于 MHA，GQA 减少 K/V 投影参数和 KV Cache 大小，代价是 K/V 可学习
的多样性减少。

## 八、RoPE：怎样把位置加入 Q/K

### 8.1 为什么只靠内容不够

如果没有位置机制，同时交换 K 和 V，Attention 加权和可能保持不变。
模型需要区分 token 在哪里、两个 token 相隔多远，以及顺序是否改变。

### 8.2 RoPE 的核心做法

RoPE 把 Q/K 的特征两两分组，每组看成二维坐标，并根据 token 位置旋转：

    head_dim = 96
    -> 48 组二维坐标

不同二维组使用不同旋转频率。位置越靠后，累计旋转角度越大；不同频率
共同工作，使长范围内的位置更容易区分。

RoPE 只旋转 Q/K，不旋转 V：

    位置影响 Q/K 点积
    -> 影响读取比例

    V 保持内容本身
    -> 不直接改写被读取信息

### 8.3 真实源码中的工具表

precompute_freqs_cis() 预先生成：

    freqs_cos：[max_position_embeddings, head_dim]
    freqs_sin：[max_position_embeddings, head_dim]

默认 shape：

    [32768, 96]

rotate_half() 完成二维旋转公式中的另一半，例如：

    [1, 2, 3, 4] -> [-3, -4, 1, 2]

forward 时根据当前绝对位置切出所需 cos/sin，再由
apply_rotary_pos_emb() 作用到 Q/K。

## 九、MiniMind 的真实模型结构

当前原生配置：

    vocab_size = 6400
    hidden_size = 768
    num_hidden_layers = 8
    num_attention_heads = 8
    num_key_value_heads = 4
    head_dim = 96
    intermediate_size = 2432
    dropout = 0.0
    max_position_embeddings = 32768
    use_moe = False
    tie_word_embeddings = True

构造链：

    MiniMindForCausalLM
    -> MiniMindModel
       -> Embedding(6400, 768)
       -> 8 个 MiniMindBlock
       -> 最终 RMSNorm
       -> RoPE cos/sin buffer
    -> lm_head(768, 6400)

Embedding 与 lm_head 共享同一个权重 Parameter：

    Embedding：token ID -> 768 维
    lm_head：768 维 -> 6400 个候选分数

共享减少一份参数，并让输入表示和输出词表空间建立直接联系。

### 9.1 一个 MiniMindBlock

Block 的主干：

    x
    -> RMSNorm
    -> Attention
    -> + x                         第一条残差
    -> RMSNorm
    -> FeedForward
    -> + Attention 残差后的结果    第二条残差

普通 FeedForward：

    x -> gate_proj -> SiLU
    x -> up_proj
    两条 2432 维结果逐元素相乘
    -> down_proj
    -> 768 维

它独立处理每个 token 的最后一维，不负责 token 之间的信息交流；
token 之间的读取由 Attention 完成。

## 十、从命令到回答：原生推理完整链

真实运行命令：

    python eval_llm.py \
      --load_from model \
      --device cpu \
      --max_new_tokens 64 \
      --temperature 0.1 \
      --top_p 0.9 \
      --show_speed 1

本地原生权重：

    minimind/out/full_sft_768.pth

验证结果：

    参数键数量：91
    参数 dtype：float16
    strict=True：全部匹配
    参数量：63,912,192
    CPU 推理速度：约 62 至 64 tokens/s

### 10.1 两条模型加载路径

原生路径：

    --load_from model
    -> model/tokenizer.*
    -> MiniMindConfig
    -> MiniMindForCausalLM
    -> out/full_sft_768.pth
    -> MiniMind 自己的 generate()

导出路径：

    --load_from ./minimind-3
    -> config.json
    -> Qwen3ForCausalLM
    -> model.safetensors
    -> Transformers GenerationMixin.generate()

两条路径之后都使用 chat template、tokenizer 和 decode，但
model.generate() 的内部实现不同。

### 10.2 输入准备

结构化消息：

    {"role": "user", "content": "你是谁"}

经过 chat template 变成模型协议字符串，再由 tokenizer 变成：

    input_ids：[1, 21]
    attention_mask：[1, 21]

attention_mask 全为 1，表示 21 个位置全部有效。

### 10.3 第一次 forward

    input_ids                 [1, 21]
    Embedding                 [1, 21, 768]
    Q                         [1, 21, 8, 96]
    K/V                       [1, 21, 4, 96]
    GQA 对齐后 Q/K/V           [1,8,21,96]
    Attention 输出            [1, 21, 768]
    8 层后 hidden_states       [1, 21, 768]
    lm_head logits            [1, 21, 6400]

每层同时返回自己的 K/V Cache：

    K：[1, 21, 4, 96]
    V：[1, 21, 4, 96]

8 层各有一对缓存，不能混合。

### 10.4 从 logits 选出一个 token

generate 只取最后位置：

    outputs.logits[:, -1, :] -> [1, 6400]

这 6400 个值是候选 token 的原始分数。依次经过：

    除以 temperature
    -> repetition penalty（当前为 1，实际跳过）
    -> Top-K（原生默认 50）
    -> Top-P（当前 0.9）
    -> Softmax
    -> multinomial 采样
    -> next_token [1, 1]

Top-K 固定限制候选数量；Top-P 根据累计概率动态限制候选集合。它们只做
筛选，真正选择 token 发生在采样或 argmax。

### 10.5 KV Cache 怎样增长

第一次生成一个 next_token 后：

    input_ids：[1,21] -> [1,22]
    Cache：仍为长度 21

原因：新 token 只是先被追加到完整序列，还没有经过模型。

下一轮：

    past_len = 21
    input_ids[:, 21:] -> [1,1]

只把最新 token 送入模型。Attention 将它的新 K/V 与历史缓存拼接：

    历史 K/V：[1,21,4,96]
    新 K/V：  [1, 1,4,96]
    新缓存：  [1,22,4,96]

因此完整 input_ids 保存最终序列，KV Cache 保存历史 Attention 所需的
K/V，二者协作避免反复计算旧 token。

### 10.6 停止和返回

每轮检查：

    新 token 是否为 eos_token_id
    所有序列是否都 finished
    是否达到 max_new_tokens

默认 generate() 返回：

    prompt token + 新生成 token

eval_llm.py 再按原 prompt 长度切片，只解码模型新生成的部分。

## 十一、贯穿整个阶段的 shape 规则

1. nn.Linear(in,out) 只把最后一维从 in 改成 out。
2. view、reshape、transpose 只重新组织已有数值，不训练新参数。
3. 矩阵乘法看最后两个维度，前面的 batch/head 维按批量规则保留。
4. Attention 分数最后一维是 key_len，Softmax 也沿 key_len 执行。
5. 残差相加要求两侧 shape 完全一致。
6. RMSNorm 不改变 shape，只处理最后一维。
7. Cache 的 sequence 维是拼接历史 token 的维度。
8. lm_head 不改变 batch 和 token 数量，只把 768 改成 6400。
9. 采样把 [batch,6400] 变成 [batch,1] 的 token ID。
10. generated_ids 是完整序列，decode 前的切片才只保留新回答。

## 十二、学习过程中的重要纠偏

阶段一不仅记录知识，也形成了稳定的学习方法。

### 12.1 不让学习者盲猜

问题必须建立在已经解释过的概念上。遇到新术语时，先说明变量前提、
作用和最小例子，再让学习者回答。

### 12.2 不一次输出整章

固定节奏：

    讲一个小概念
    -> 用户回答或实践
    -> 解释结果
    -> 验收
    -> 再进入下一步

### 12.3 必须按真实调用关系读源码

源码阅读顺序：

    当前函数
    -> 遇到被调用函数
    -> 进入该函数并读完相关路径
    -> 返回调用者下一行

不能在 MiniMindForCausalLM 尚未读完时跳到后面的 Model，也不能在
self.self_attn() 尚未展开时直接跳过 Attention。

### 12.4 假设例子与真实运行必须分开

真实命令的 batch 是 1；[3,21] 只是在解释 num_return_sequences=3 时的
假设 shape。任何 shape 都要标明是当前实际值，还是教学示例。

### 12.5 Cache 长度要看计算时间点

新 token 追加到 input_ids 后，Cache 不会立刻增长；必须等下一轮 forward
真正计算它之后，Cache 才增长。这是静态 shape 推理中必须保留的时间关系。

## 十三、当前掌握边界

已经达到当前深度的内容：

    - 能运行 MiniMind 原生模型并观察输出
    - 能解释 token、token ID、Embedding 和上下文化
    - 能手工计算 Linear、点积和简单矩阵乘法
    - 能解释 loss、梯度、backward、清零和 SGD
    - 能手算简化 Attention 和 Softmax
    - 能解释 Q/K/V、Causal Mask、Multi-Head、GQA 和 RoPE
    - 能跟踪 MiniMind Block 内的主要 shape
    - 能按调用栈读通原生 forward 和 generate
    - 能解释 logits、温度、Top-K、Top-P、采样、EOS 和 KV Cache
    - 能区分原生 .pth 与 Transformers 导出路径

尚未完成的能力：

    - 尚未在真实运行中系统打印并核对全部推理 shape
    - 尚未读通 MiniMind 的数据集和真实训练脚本
    - 尚未学习交叉熵在语言模型中的 label shift 和 mask 实现
    - 尚未读优化器、学习率调度、混合精度、梯度累积和 checkpoint 保存
    - 尚未学习 Pretrain、SFT、LoRA、DPO、PPO、GRPO 的完整差异
    - 尚未独立重写一个可训练、可生成的 MiniMind Lite
    - 尚未进入 MiniMind-V 与 MiniMind-O

因此当前水平可以描述为：

> 已经从“会运行模型”进入“能沿源码解释小型 Decoder-only Transformer
> 的推理主干”；下一阶段需要把这种静态理解转成真实追踪能力，并补齐
> 训练数据与训练循环。

## 十四、下一阶段入口

阶段二首先闭合推理侧最后一段：

    完整 generated_ids
    -> 只切出新 token
    -> decode
    -> 保存 assistant 对话历史
    -> 统计 tokens/s

然后用真实 shape trace 验证阶段一的静态推导。推理闭环验收后，进入：

    trainer/train_full_sft.py

训练阶段将沿真实调用顺序回答：

    训练数据长什么样
    -> Dataset 怎样产生 input_ids 和 labels
    -> 模型怎样计算每个位置的 logits
    -> 为什么 logits 和 labels 要错开一位
    -> Cross Entropy 怎样得到 loss
    -> backward 怎样把梯度传给 6391 万个参数
    -> optimizer 怎样更新参数
    -> checkpoint 怎样保存和恢复

阶段一到此结束。后续 Day 文档继续保留真实互动过程，本阶段小结作为
复习入口和新学习者的连续章节。
