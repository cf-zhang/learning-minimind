# MiniMind 仓库驱动学习档案

这不是知识点摘要目录，而是一套持续生长的交互式入门课程。课程从真实仓库和真实运行结果出发，按学习日保存完整过程，让没有数学和模型基础、但具备编程基础的学习者也能按照相同步骤复现。

长期路线见：[当前目录三仓库学习路线](roadmap-three-repos.md)。范围严格限定为当前目录直属的 `minimind`、`minimind-v`、`minimind-o`，不扩展到目录外的其他仓库。

## 一、阅读顺序

| 顺序 | 文档 | 状态 | 主题 |
| --- | --- | --- | --- |
| 0 | [学习启动记录](kickoff-2026-08-11.md) | 已归档，不计 Day | 背景、目标、三个仓库的关系、长期路线 |
| 1 | [Day 001](day-001.md) | 已完成 | 运行模型、Tokenizer、token、Embedding、上下文化现象 |
| 2 | [Day 002](day-002.md) | 已完成 | 最小 PyTorch 模型、Linear、矩阵乘法与点积 |
| 3 | [Day 003](day-003.md) | 已完成 | prediction、target、loss 与梯度来源 |
| 4 | [Day 004](day-004.md) | 已完成 | `grad_fn`、`backward()` 与手工参数更新 |
| - | [阶段整理](progress-note-2026-08-17.md) | 已归档，不计 Day | Backward 跨日恢复点，不包含新学习 |
| 5 | [Day 005](day-005.md) | 已完成 | 梯度累加、清零、SGD 与最小训练循环 |
| 6 | [Day 006](day-006.md) | 已完成 | Attention、Q/K/V、缩放点积、Causal Mask 与单 Head |
| 7 | [Day 007](day-007.md) | 已完成 | 多 Head 的维度拆分、独立计算、拼接与 `o_proj` |
| 8 | [Day 008](day-008.md) | 已完成 | GQA 与共享 K/V Head |
| 9 | [Day 009](day-009.md) | 已完成 | RoPE 与 token 位置信息 |
| 10 | [Day 010](day-010.md) | 已完成 | 从 `eval_llm.py` 按执行顺序阅读源码 |
| 11 | [Day 011](day-011.md) | 已完成 | `generate()` 外层准备与 MiniMind 原生入口定位 |
| 12 | [Day 012](day-012.md) | 已完成 | 从运行命令重新建立 MiniMind 原生调用链 |
| - | [阶段一小结](stage-01-days-001-012.md) | 已完成，不计 Day | 从文本、训练基础、Attention 到 MiniMind 原生推理链 |
| 13 | [Day 013](day-013.md) | 已完成 | 闭合推理外层并验证完整数据流 |
| 14 | [Day 014](day-014.md) | 已完成 | 从 Full SFT 入口进入训练链路，止于训练循环开始前 |
| 15 | [Day 015](day-015.md) | 已完成 | Full SFT 训练循环、最小真实训练、权重保存与重新推理 |
| 16 | [Day 016](day-016.md) | 已完成 | 进入 Pretrain 训练主线，比较 Pretrain 与 Full SFT |
| 17 | [Day 017](day-017.md) | 已完成 | 从 `train_tokenizer.py` 开始阅读 Tokenizer 训练 |
| 18 | [Day 018](day-018.md) | 已完成 | MoE 单层路由、专家合并与 `aux_loss` |
| 19 | [Day 019](day-019.md) | 已完成 | MoE 总参数量、激活参数量与计算成本 |
| 20 | [Day 020](day-020.md) | 已完成 | LoRA 注入、权重管理与训练更新 |
| 21 | [Day 021](day-021.md) | 已完成 | LoRA 权重加载、合并与推理验收 |
| 22 | [Day 022](day-022.md) | 已完成 | 蒸馏中的 teacher、student 与损失目标 |
| 23 | [Day 023](day-023.md) | 已完成 | 蒸馏真实 forward、反向传播与参数更新 |
| 24 | [Day 024](day-024.md) | 已完成 | DPO 的 chosen/rejected 数据与偏好目标 |
| 25 | [Day 025](day-025.md) | 已完成 | GRPO rollout、奖励与组内相对优势 |
| 26 | [Day 026](day-026.md) | 暂停 | PPO 的直观目标、critic、return、GAE 与 actor 更新 |

配套实验：

- [Day 006 Attention Notebook](notebooks/day-006-attention.ipynb)：Softmax、Q/K/V、缩放点积、Causal Mask 和完整简化单 Head Attention。
- [Day 007 Multi-Head Attention Notebook](notebooks/day-007-multi-head-attention.ipynb)：Q/K/V 投影、拆 Head、并行 Attention、按 token 拼接与 `o_proj`。
- [Day 008 GQA Notebook](notebooks/day-008-gqa.ipynb)：共享 K/V、`repeat_kv`、完整简化 GQA、参数收益与真实 MiniMind-3 权重。
- [Day 010 `eval_llm.py` 调用链 Notebook](notebooks/day-010-eval-llm-call-chain.ipynb)：入口、参数、配置、chat template、tokenizer、生成控制参数、结果切片、对话历史与速度统计。
- [Day 011 `generate()` 调用链 Notebook](notebooks/day-011-generate-call-chain.ipynb)：真实 Qwen3/Transformers 路径、生成配置、采样策略、输入准备和 MiniMind 原生精读入口。
- [Day 012 MiniMind 原生推理调用链 Notebook](notebooks/day-012-native-inference-call-chain.ipynb)：Attention 输出合并、Block/Model/LM Head 返回值、采样、KV Cache 时间线、完整序列与 decode 切片。
- [Day 013 推理外层数据流 Notebook](notebooks/day-013-eval-output-flow.ipynb)：完整序列切片、流式显示与回答字符串、跨轮历史、生成 token 数与近似吞吐量。
- [Day 014 Full SFT 入口 Notebook](notebooks/day-014-sft-entry.ipynb)：训练参数、SFT 数据整理、labels、预训练权重加载、GradScaler 与 AdamW 初始化。
- [Day 015 Full SFT 训练循环 Notebook](notebooks/day-015-sft-training-loop.ipynb)：epoch、DataLoader、next-token loss、梯度更新、checkpoint、4 条数据真实训练和训练后推理诊断。
- [Day 016 Pretrain 数据 Notebook](notebooks/day-016-pretrain.ipynb)：`text` 样本、BOS/EOS/PAD、预训练 labels、next-token 对齐与 `-100` 忽略。
- [Day 017 Tokenizer 训练 Notebook](notebooks/day-017-tokenizer-training.ipynb)：现有 tokenizer 的词表、特殊 token、encode/decode 可逆性和粗略压缩率。
- [Day 018 MoE 分支 Notebook](notebooks/day-018-moe-branch.ipynb)：普通 FFN 回顾、router、softmax、top-k、专家分发、`index_add_`、`aux_loss` 和多层汇总。
- [Day 019 MoE 参数量 Notebook](notebooks/day-019-moe-parameters.ipynb)：普通 FFN 与 MoE 参数量、完整模型对照、激活参数量、expert-token routes 与近似计算成本。
- [Day 020 LoRA 注入与训练 Notebook](notebooks/day-020-lora-injection-training.ipynb)：LoRA A/B、目标层注入、适配器保存/加载/合并、冻结参数和梯度累积。
- [Day 022 蒸馏基础 Notebook](notebooks/day-022-distillation-basics.ipynb)：teacher/student、logits 对齐、temperature、KL、alpha 组合和有效位置 mask。
- [Day 024 DPO 偏好 Notebook](notebooks/day-024-dpo-preference.ipynb)：chosen/rejected 数据、回答 loss mask、log probability、reference 锚点和 DPO loss。
- [Day 025 GRPO Notebook](notebooks/day-025-grpo-overview.ipynb)：rollout、policy/reference/reward 角色、组内 reward、advantage、ratio 和 KL 主线。

Day 编号按实际发生的学习日递增，不按自然日期递增。只整理文档、安装环境或由助手代跑程序，不单独算作一个 Day。

## 二、每份 Day 文档保存什么

每个学习日保留两层内容。

第一层是“逐轮互动档案”，按真实发生顺序记录：

```text
为什么进入这个问题
  -> 助手当时讲了什么
  -> 学习者提出了什么疑问或纠偏
  -> 学习者亲自执行了什么
  -> 得到了什么实际输出
  -> 如何解释输出
  -> 学习者是否确认理解
  -> 为什么进入下一步或暂停
```

第二层是“可复现教程”，把真实过程整理为其他新人可以独立执行的材料：

```text
前置知识 -> 概念讲解 -> 命令或代码 -> 预期现象
         -> 实际样例输出 -> 结果解释 -> 检查问题 -> 知识欠账
```

互动档案不是逐字聊天转录。无法确认原句时不会伪造引号，而会明确写成“过程整理”。学习者真实提交过的命令、代码、输出、问题和回答会尽量原样保留。

## 三、状态标记

后续文档统一使用以下状态，避免把“助手做过”误写成“学习者学会了”。

| 标记 | 含义 |
| --- | --- |
| `[环境]` | 为学习准备好了环境、模型或文件，可能由助手完成 |
| `[讲解]` | 概念已经说明，但学习者还没有实践或确认 |
| `[实践]` | 学习者已经亲自运行、计算或观察 |
| `[验收]` | 学习者已经回答、复述或明确确认，当前深度算掌握 |
| `[纠偏]` | 教学顺序或表达方式出现问题，学习者指出后重新调整 |
| `[欠账]` | 已经遇到但依赖前置知识，明确留到后续学习 |

一个知识点只有在必要讲解、实践和确认完成后，才写成“已掌握”。只有 `[讲解]` 的内容不得在总结中冒充完成项。

## 四、交互式学习规则

每次只推进一个足够小的概念：

```text
助手讲清一个概念
  -> 停下
  -> 学习者理解或提问
  -> 助手给一个小实验
  -> 学习者亲自执行并提交结果
  -> 双方解释实际结果
  -> 验收后才进入下一步
```

具体约束：

1. 不让学习者猜尚未讲过的术语。
2. 不一次引入多个互相依赖的专业概念。
3. 助手代跑成功只证明环境可用，不证明学习者掌握。
4. 学习者说“没有疑问”只确认刚刚那一个小步骤，不自动验收后续内容。
5. 计划中的内容如果当天没有实际学习，必须保留为未完成。
6. 新术语必须从已经验证过的事实引出。
7. 每次学习结束时记录准确恢复点，下次从该点继续。

## 五、当前主线

```text
文本
  -> Tokenizer 与 token ID                  Day 001
  -> Embedding 初始向量                     Day 001
  -> Linear 如何加工每个 token              Day 002
  -> loss、梯度与训练闭环                    Day 003-005
  -> Attention、Q/K/V 与单 Head 实现          Day 006
  -> 多 Head 的拆分、计算与组合               Day 007
  -> GQA：Q Head 与 K/V Head 数量不同         Day 008
  -> RoPE：为 Q/K 加入位置信息                Day 009
  -> 从 eval_llm.py 按执行顺序阅读源码         Day 010（已完成）
  -> generate() 外层准备与原生入口定位        Day 011（已完成）
  -> 从运行命令重新建立原生 MiniMind 调用链    Day 012（已完成）
  -> 阶段一小结：基础知识与原生推理主线      已完成
  -> Day 013：外层 decode 收尾与完整数据流    已完成
  -> Day 014：`trainer/train_full_sft.py` 入口  已完成（止于第 158 行训练循环）
  -> Day 015：Full SFT 真实训练循环与最小实训      已完成
  -> Day 016：进入 `train_pretrain.py`，比较 Pretrain 与 Full SFT    已完成
  -> Day 017：从 `train_tokenizer.py` 开始阅读 Tokenizer 训练    已完成
  -> Day 018：MoE 单层路由、专家合并与 aux_loss    已完成
  -> Day 019：MoE 总参数量、激活参数量与计算成本    已完成
  -> Day 020：LoRA 注入、权重管理与训练更新    已完成
  -> Day 021：LoRA 权重加载、合并与推理验收    已完成
  -> Day 022：蒸馏中的 teacher、student 与损失目标    已完成
  -> Day 023：蒸馏真实 forward、反向传播与参数更新    已完成
  -> Day 024：DPO 的 chosen/rejected 数据与偏好目标    已完成
  -> Day 025：GRPO rollout、奖励与组内相对优势    已完成
  -> Day 026：PPO 的直观目标、critic、return、GAE 与 actor 更新    暂停
```

截至 Day 012，已经沿原生推理链学习了 RoPE 在 Q/K 上的应用、KV Cache
的逐 token 增长、RMSNorm、两条残差连接、普通 FeedForward 和完整
`MiniMindBlock` 主干。Day 013-015 又完成了推理外层、Full SFT 数据与
训练循环，并用 4 条数据真实验证了参数更新、保存和重新加载。Pretrain、LoRA、DPO
等后续训练阶段已经完成；当前尚未完成的是 GRPO/CISPO/PPO、Agent RL 以及视觉和
Omni 阶段。准确的早期掌握边界见
[阶段一小结](stage-01-days-001-012.md)，Full SFT 的最新边界见
[Day 015](day-015.md)。

## 六、练习代码说明

`codes/` 保存学习过程中使用过的脚本。早期脚本文件名是在 Day 编号规则完全固定前创建的，因此存在历史错位：

| 文件 | 实际对应内容 |
| --- | --- |
| `codes/day3.py` | Day 002 的 `SmallModel`、Linear、多 token 与点积实验 |
| `codes/day4.py` | Day 004 的计算图、backward、手工更新，以及 Day 005 开头的累加/清零实验 |
| `codes/day5.py` | Day 005 的五轮 SGD 训练 |
| `codes/day6.py` | 历史空文件；Day 006 实践已整理到配套 Notebook |

从 Day 006 起，适合交互实验的内容优先进入 `notebooks/`。Markdown 保留完整教学互动和掌握状态，Notebook 保留可运行代码与输出；`.py` 只在需要独立脚本时创建。

归档保留这些历史文件名，避免把后来整理出的内容误装成当时就存在。新人复现时以各 Day 文档中的完整代码块和步骤为准。

## 七、真实性边界

本档案会补充新人理解所需的解释，但遵守三条边界：

1. 不伪造学习者没有执行过的输出。
2. 不把后来补写的教学说明说成当时的逐字对话。
3. 不把尚未回答、尚未实践或尚未验收的内容勾选为完成。

因此，已完成的 Day 可以同时作为复习教程和过程证据；进行中的 Day 则会明确停在最后一个真实确认点。

当前恢复点：PPO 暂停，下一阶段建议从 `minimind/trainer/train_agent.py` 的
`AgentRLDataset` 和工具调用数据流开始；PPO 的 critic/value、return、TD error、
GAE 留作后续可选专题。
