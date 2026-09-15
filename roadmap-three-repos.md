# 当前目录三仓库学习路线

- 范围：`/home/zcf/githubs/minimind`
- 仓库：`minimind`、`minimind-v`、`minimind-o`
- 目标：沿真实源码和真实运行结果，逐步理解、运行、修改这三套模型代码。
- 学习方式：每天一个 Day 文档；每次只推进一个小代码块；学习者回答或实践后才进入下一步。

## 一、三个仓库的递进关系

```text
minimind
纯文本语言模型：理解模型结构、训练和推理
    |
    | 增加图像输入
    v
minimind-v
视觉语言模型：理解图像如何变成语言模型可以读取的向量
    |
    | 再增加语音输入和语音输出
    v
minimind-o
Omni 模型：理解文本、图像、语音共同进入模型，以及语音生成
```

本路线只使用当前目录中的这三个仓库，不向父目录或其他同级仓库扩展。

## 二、第一阶段：完成 `minimind`

已完成 Day 001-015：

```text
Tokenizer、token、Embedding、Linear
loss、梯度、backward、SGD、AdamW
Attention、Multi-Head、GQA、RoPE、KV Cache
eval_llm.py 与 MiniMind 原生推理链
Full SFT 入口、Dataset、labels、训练循环和 checkpoint
GPU、单卡 forward/backward、双卡 NCCL/DDP 验证
```

### 1. Pretrain：约 2-3 个学习日

入口：

```text
minimind/trainer/train_pretrain.py
minimind/dataset/lm_dataset.py::PretrainDataset
```

重点：

```text
PretrainDataset 与 SFTDataset 的输入格式
普通文本如何变成 input_ids
BOS、EOS、padding 的作用
预训练 labels 为什么覆盖文本内容
预训练为什么学习续写
SFT 为什么学习按角色和指令回答
```

用同一个 prompt 对比：

```text
pretrain_768.pth
full_sft_768.pth 或训练完成的 full_sft_gpu_768.pth
```

### 2. Tokenizer 训练：约 1-2 个学习日

入口：

```text
minimind/trainer/train_tokenizer.py
minimind/model/tokenizer.json
minimind/model/tokenizer_config.json
```

重点：词表 6400 的产生、训练语料、特殊 token、chat template，以及
Tokenizer 训练产物如何被模型和数据集加载。

### 3. MoE 分支：约 3-4 个学习日

入口：

```text
minimind/model/model_minimind.py
```

重点：专家、router、top-k、专家输出合并、aux loss、总参数量与激活参数量。

### 4. LoRA：约 3-4 个学习日

入口：

```text
minimind/model/model_lora.py
minimind/trainer/train_lora.py
```

重点：冻结原模型、低秩矩阵、可训练参数量、LoRA 权重保存/加载/合并，
并用少量数据验证原模型与 LoRA 参数的差异。

### 5. 蒸馏：约 2-3 个学习日

入口：

```text
minimind/trainer/train_distillation.py
```

重点：teacher、student、soft target、temperature、蒸馏 loss 与交叉熵组合。

### 6. DPO：约 3-5 个学习日

入口：

```text
minimind/dataset/lm_dataset.py::DPODataset
minimind/trainer/train_dpo.py
```

重点：prompt、chosen、rejected、log probability、reference model 和偏好目标。

### 7. GRPO/CISPO/PPO：约 7-10 个学习日

入口：

```text
minimind/trainer/train_grpo.py
minimind/trainer/train_ppo.py
minimind/trainer/rollout_engine.py
```

顺序：rollout -> reward -> 多个回答 -> advantage -> reference/KL -> ratio/clip
-> policy loss；之后再补 PPO 的 Actor、Critic、value 和 return。

### 8. Tool Use 与 Agent RL：约 4-6 个学习日

入口：

```text
minimind/trainer/train_agent.py
minimind/dataset/lm_dataset.py::AgentRLDataset
minimind/scripts/web_demo.py
minimind/scripts/serve_openai_api.py
```

重点：工具定义进入模板、tool_call 解析、工具执行、结果回填、多轮 rollout
以及工具正确性和最终回答奖励。

### 9. 转换、服务、评估：约 2-3 个学习日

入口：

```text
minimind/scripts/convert_model.py
minimind/scripts/serve_openai_api.py
minimind/scripts/web_demo.py
minimind/eval_llm.py
```

目标：完成原生/Transformers 权重转换、OpenAI 兼容接口、流式输出、工具调用
解析和固定评测。

`minimind` 剩余预计约 25-35 个实际学习日。

## 三、第二阶段：学习 `minimind-v`

不重复已经掌握的语言模型主干，只学习视觉新增部分，约 10-15 个学习日。

### 1. 运行和调用链

```text
minimind-v/eval_vlm.py
-> 图片预处理
-> vision encoder
-> image feature
-> vision projector
-> image token 注入 MiniMind
-> 文本回答
```

### 2. 模型与数据

```text
minimind-v/model/model_vlm.py
minimind-v/dataset/lm_dataset.py::VLMDataset
```

重点：`VLMConfig`、SigLIP2、`MMVisionProjector`、`image_token_len`、
`image_bytes`、Parquet、`pixel_values`、多模态 labels 和 batch 对齐。

### 3. 视觉训练

```text
minimind-v/trainer/train_pretrain_vlm.py
minimind-v/trainer/train_sft_vlm.py
```

重点比较 `freeze_llm=0/1/2`，并用仓库内测试图片和极小实验验证视觉输入。

## 四、第三阶段：学习 `minimind-o`

这是三套仓库中最复杂的一套，预计约 15-22 个学习日。

### 1. Omni 总体结构

```text
minimind-o/eval_omni.py
minimind-o/model/model_omni.py
```

先区分 Thinker 和 Talker：

```text
Thinker：理解文本、图像、语音，产生文本隐藏状态/文本 logits
Talker：根据 Thinker 信息和音频 code，生成音频 code
```

### 2. 音频、视觉和 Talker

重点：`SenseVoiceAudioProcessor`、音频 encoder、`MMAudioProjector`、
`MMVisionProjector`、`TalkerEmbedding`、`TalkerHead`、8 层 audio codebook、
不同的文字词表和音频词表，以及 9 路输入。

### 3. Omni 数据和训练

```text
minimind-o/dataset/omni_dataset.py
minimind-o/trainer/train_sft_omni.py
minimind-o/trainer/train.sh
```

重点：文字 labels、8 路 audio labels、audio/image 输入、speaker embedding、
scheduled sampling、`mode=all/audio_proj/vision_proj`、backbone 冻结策略和
多路 loss。

### 4. 实时能力

最后学习流式语音生成、VAD、打断、近双工交互、voice cloning 和 Web UI。

## 五、最终跨仓库验收

完成后能够独立解释：

```text
文本如何变成 token 并训练/生成
图像如何变成视觉位置并进入语言模型
语音如何变成音频特征并进入 Thinker
Talker 如何把隐藏状态变成多层音频 code
三个仓库哪些代码复用，哪些代码新增
```

三个仓库剩余预计约 50-70 个实际学习日。这个数字不是硬性课表；只有真正
学习和验收才计 Day，训练等待、安装环境和单纯整理文档不单独计 Day。

## 六、与具身智能的连接

这三套仓库能够提供语言模型、视觉语言、语音交互、工具调用、任务规划和
Agent 的基础，但不等于完整机器人系统。完成它们后仍需在机器人方向补充：

```text
ROS2、计算机视觉、机器人学、运动学/动力学
导航定位、机械臂控制、强化学习控制、仿真和安全约束
```

三仓库学习完成后，适合把语言模型作为机器人高层任务理解和规划模块，
再通过工具接口连接视觉、导航、抓取和控制系统。

## 七、固定学习节奏

```text
指定真实源码恢复点
-> 讲一个小代码块和新术语
-> 学习者回答或实践
-> 对照真实输出纠偏
-> 验收
-> 记录 Markdown 和 Notebook
```

当前恢复点：第三阶段 `minimind-o` 进行中。Day 032 已完成整体地图与 eval_omni.py
mode=0 逐行推理链（Thinker出文本 + Talker出audio code 流式并行，Mimi解码成24kHz语音；
输入侧三模态占位符挖坑复用 minimind-v）。下一小步进入 `minimind-o/model/model_omni.py`：
OmniConfig/MiniMindOmni 结构、audio/vision projector 填坑、Bridge 中间层取表征、
Talker+MTP 一次预测8层Mimi code、generate 内部 Thinker–Talker 协作。PPO 的
critic/value、return、TD error、GAE 留作后续可选专题。
