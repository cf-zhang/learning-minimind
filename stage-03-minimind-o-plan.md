# 第三阶段学习计划：minimind-o

- 学习阶段：第三阶段 Omni 模型
- 主仓库：minimind-o
- 计划状态：未掌握，重新启动
- 预计投入：18-22 个实际学习日
- 学习方式：仓库驱动、一次一个概念、真实代码、最小实验、交互验收

## 一、计划说明

之前已经接触过 minimind-o 的整体结构和推理入口，但接触过不等于掌握。
因此本阶段从零开始重新学习。已有的 Day 记录保留为历史接触记录，不把其中
的讲解或预览自动计入本阶段掌握成果；后续必须重新完成讲解、实践和验收。

本阶段不追求一次读完所有源码，而是沿着一条可以反复验证的因果链推进：

    完整推理地图
        -> Thinker 的多模态输入
        -> Bridge 连接 Thinker 与 Talker
        -> Talker 预测 8 层 audio code
        -> OmniDataset 构造 9 路输入和监督
        -> text loss + 8 路 audio loss
        -> 流式解码、音色和实时交互

每次学习遵循以下循环：

    一个小概念
      -> 解释它解决什么问题
      -> 定位真实代码
      -> 运行或手工构造最小实验
      -> 解释实际输出
      -> 学习者复述或回答一个验收问题
      -> 确认后再进入下一个概念

安装环境、下载权重、等待训练和整理文档不单独计学习日。

## 二、阶段目标

完成后能够独立解释并用真实张量形状验证下面这条链路：

    文本 -> text token -> Thinker -> text logits
    图像 -> SigLIP2 -> VisionProjector -> Thinker
    语音 -> fbank -> SenseVoice -> AudioProjector -> Thinker
    Thinker 中间层 -> Bridge -> Talker
    历史 audio code + Bridge -> Talker -> 8 路 audio logits
    8 路 audio code -> Mimi -> 24kHz 波形

最终还要能够：

- 读懂 eval_omni.py 的文本、音频、图像和混合输入模式。
- 解释 MiniMindOmni.forward() 的 Thinker/Talker 两路计算。
- 解释 stream_generate() 中 MTP 的时间错位和流式输出。
- 从一条 Parquet 样本还原 9 路输入、文本 labels 和 8 路 audio labels。
- 跑通一次最小训练 step，并验证 loss、梯度、参数更新和 checkpoint 加载。
- 解释音色克隆、VAD、barge-in 和近双工交互的工程链路。
- 对代码做一个小改动，给出修改前后的可重复对照实验。

## 三、环境准备（不计学习日）

真正运行完整模型前，需要准备以下资源：

    minimind-o/model/SenseVoiceSmall/
    minimind-o/model/siglip2-base-p32-256-ve/
    minimind-o/model/mimi/
    minimind-o/model/campplus/
    minimind-o/model/vad/silero_vad.onnx
    minimind-o/out/llm_768.pth
    minimind-o/dataset/sft_t2a_mini.parquet
    minimind-o/dataset/sft_a2a_mini.parquet
    minimind-o/dataset/sft_i2t_mini.parquet

资源准备完成后，先单独验证 CUDA，再确认以下资源可以独立加载：

    Tokenizer
    MiniMind 语言权重
    SenseVoice / SigLIP2 编码器
    Mimi 编解码器

如果资源尚未下载，仍然可以先学习配置、数据布局、伪输入和小张量实验，
不因为等待下载而中断概念主线。

## 四、学习日安排

### 单元一：完整推理地图（第 1-2 天）

#### 第 1 天：MiniMind-O 新增了什么

代码入口：

    minimind-o/README.md
    minimind-o/eval_omni.py::init_model
    minimind-o/eval_omni.py::eval_sample

只建立四个组件的基本分工：

    SenseVoice：语音输入编码器
    Thinker：理解文本、图像和语音，并生成文本相关结果
    Talker：根据 Thinker 的语义条件生成 audio code
    Mimi：在音频波形和离散 audio code 之间转换

实践：

- 画出一次“文本 -> 文本+语音”的调用链。
- 找到 init_model、eval_sample、model.generate 和 mimi.decode。
- 条件允许时运行一条最短文本推理。

验收：

- 能说明四个组件分别接收什么、输出什么。
- 能说明这不是简单的 ASR -> LLM -> TTS 串联。

#### 第 2 天：六种推理模式和输入边界

代码入口：

    minimind-o/eval_omni.py::main

认识以下模式：

    mode=0  文本 -> 文本+语音
    mode=1  多轮文本 -> 文本+语音
    mode=2  语音 -> 文本+语音
    mode=3  指定或克隆音色 -> 文本+语音
    mode=4  图像 -> 文本+语音
    mode=5  文本+语音+图像 -> 文本+语音

实践：对每种模式追踪传给 eval_sample 的参数，整理成输入参数表。

验收：能够判断什么时候需要 audio_inputs、pixel_values、ref_codes 和
spk_emb，并区分“问题的语音内容”和“回答的音色条件”。

### 单元二：Thinker 的多模态输入（第 3-6 天）

#### 第 3 天：OmniConfig 和模块树

代码入口：

    minimind-o/model/model_omni.py::OmniConfig
    minimind-o/model/model_omni.py::MiniMindOmni.__init__

重点配置：

    text vocab / audio vocab
    audio_pad_token / audio_stop_token / audio_spk_token
    audio_ids / image_ids
    image_token_len
    bridge_layer
    talker_hidden_size

实践：打印关键配置、模块树和参数量，确认 thinker 是语言主干的别名。

验收：能够区分文本词表、音频词表、音频特殊 code 和图像/音频占位 token。

#### 第 4 天：语音如何变成 Thinker 特征

代码入口：

    SenseVoiceAudioProcessor
    MiniMindOmni.load_sensevoice
    OmniDataset.process_audio
    MiniMindOmni.encode_audio_inputs

数据流：

    声音波形
    -> 16kHz 重采样
    -> fbank（约 [时间, 560]）
    -> SenseVoice encoder
    -> 连续音频特征（约 512 维）
    -> AudioProjector
    -> MiniMind hidden_size 维

实践：对一段音频打印采样率、波形长度、fbank 形状、valid_len 和投影后
特征形状，并确认 SenseVoice 参数被冻结。

验收：能区分波形、fbank、SenseVoice 连续特征和 Projector 输出；能解释
audio_lens 为什么不能被 padding 后的统一长度替代。

#### 第 5 天：音频特征如何填入文本序列

代码入口：

    MiniMindOmni.inject_audio_features
    MiniMindOmni.forward 中的 audio 分支

核心机制：

    prompt 放置 N 个 audio_pad 占位符
    -> tokenizer 得到 N 个占位位置
    -> 先得到普通 token embedding
    -> 用 N 个连续音频特征覆盖这些位置
    -> 序列长度保持不变

实践：用人工小张量模拟占位和覆盖，验证非占位位置不变。

验收：能够解释“覆盖 embedding”而不是“把音频特征追加到序列末尾”，并说明
占位符数量和有效音频特征长度的关系。

#### 第 6 天：图像注入和 Bridge

代码入口：

    get_image_embeddings
    encode_image_inputs
    count_vision_proj
    forward 中的 bridge_states

先复习 minimind-v 的路径：

    图像 -> SigLIP2 -> VisionProjector -> 覆盖 image token

再学习：

    Thinker embedding
    -> Thinker 前若干层
    -> 保存 bridge_states
    -> Thinker 继续生成 text logits
    -> bridge_states 交给 Talker

实践：用 hook 打印 bridge_states 和最终 h_thinker 的形状与数值差异，
并观察改变 bridge_layer 后取出的层发生什么变化。

验收：能解释 Bridge 是信息分叉点，且 Talker 接收的是 hidden state，不是文本
字符串，也不是最终采样出的文本 token。

### 单元三：Talker 和 8 层 audio code（第 7-11 天）

#### 第 7 天：Mimi 和 8 层 codebook

数据流：

    波形 -> Mimi encode -> [batch, 8, frames] audio codes
    audio codes -> Mimi decode -> 波形

实践：对短音频执行 encode/decode，打印波形长度、帧数、8 层形状和采样率，
并听原始与重建音频。

验收：能区分“一帧中的 8 个 codebook code”和“连续的 8 个时间帧”，并说明
Talker 为什么预测 code 而不是直接预测 24kHz 波形采样点。

#### 第 8 天：TalkerEmbedding

代码入口：

    minimind-o/model/model_omni.py::TalkerEmbedding

输入和输出：

    audio_ids: [batch, 8, sequence]
    talker_emb: [batch, sequence, talker_hidden_size]

每个 codebook 使用共享 base embedding 和自己的 adapter，8 路结果再组合。

实践：用小词表和小隐藏维度手工核对某个位置的 8 路 embedding 组合。

验收：能解释为什么需要“共享 base + 每层 adapter”，以及为什么 8 路 code
最后要压成一条 Talker hidden 序列。

#### 第 9 天：Bridge 和历史 audio code 的融合

代码核心：

    hidden_states =
        embed_proj(bridge_states) * text_scale
        + codec_proj(talker_emb) * audio_scale

实践：打印两条支路的形状，临时令 text_scale=0 或 audio_scale=0，观察
Talker 输入变化。

验收：能说明 Talker 同时依赖 Thinker 语义和历史 audio code，且这里是逐位置
相加而不是序列拼接。

#### 第 10 天：TalkerHead 和 8 路 logits

代码入口：

    TalkerHead
    TalkerModule
    MiniMindOmni.forward

输出：

    text_logits:  [B, T, text_vocab_size]
    audio_logits: 8 个张量，每个为 [B, T, audio_vocab_size]

实践：运行一次小前向，打印文本 logits 和全部 8 路 audio logits 的形状，
查看 base head 与 adapter 的参数量。

验收：能区分 TalkerEmbedding 和 TalkerHead，并解释为什么需要 8 份 audio
logits。

#### 第 11 天：MTP 的错位生成时间线

代码入口：

    MiniMindOmni.stream_generate

先用数字表格模拟，不直接依赖完整模型：

    生成步 0：产生文本 token
    生成步 1：audio codebook 0 开始产生 code
    生成步 2：codebook 0、1 产生 code
    ...
    生成步 8：8 个 codebook 均参与生成

重点理解：

    frame = [audio_codes[i][step - 7 + i] for i in range(8)]

实践：手工写出前 10-12 步的 8 路缓冲区，找出第一帧完整 audio frame 的时刻。

验收：能够画出 MTP 的 diagonal delay，并解释 8 路并行为什么仍有层间错位。

### 单元四：Omni 数据集和 9 路训练输入（第 12-15 天）

#### 第 12 天：三类 Parquet 数据

代码入口：

    minimind-o/dataset/omni_dataset.py::OmniDataset

重点字段：

    conversations
    question_audios
    answer_audios
    image_bytes
    ref_audios
    spk_emb

实践：各取一条真实样本，只打印字段名、类型和长度，画出字段流向。

验收：能根据字段判断 T2A、A2A 和 I2T，并区分问题音频、回答音频和参考音频。

#### 第 13 天：prompt 和 text_labels

代码入口：

    create_chat_prompt
    generate_text_labels

实践：解码一条 input_ids，标记 text_labels != -100 的位置，验证只训练
最后一个 assistant 回答。

验收：能逐位置说明 input_ids 和 text_labels 的关系，并解释为什么旧对话
可以保留在输入中却不参与当前文本 loss。

#### 第 14 天：8 路 audio_labels 和 9 路输入

核心形状：

    X_audio      [8, T-1]
    X_text       [1, T-1]
    input_ids    [9, T-1]
    text_labels  [T-1]
    audio_labels [8, T-1]

重点：前 8 路是 codebook 历史输入，第 9 路是文本；输入和 labels 都进行
next-token 左右错位；每一路 audio target 还具有自己的 MTP 偏移。

实践：用短 audio code 序列打印矩阵前若干列，解释 audio_pad、audio_stop
和有效 target，并与第 11 天的生成时间线对齐。

验收：能从 audio_labels 反推出 diagonal delay，并说明为什么输入去掉最后一
列、labels 去掉第一列。

#### 第 15 天：speaker、reference audio 和 scheduled sampling

代码入口：

    audio_spk_token
    spk_emb
    ref_codes
    apply_scheduled_sampling

实践：对比无音色条件、只有 spk_emb、同时有 spk_emb + ref_codes 的布局；
固定随机种子观察 scheduled sampling 替换的位置，并验证 image token 连续区间
受到保护。

验收：能区分说话人身份向量和参考音频 code，并解释 scheduled sampling 要解决
训练时输入正确历史、推理时输入可能出错历史之间的差异。

### 单元五：训练链路（第 16-18 天）

#### 第 16 天：一个完整训练 step

代码入口：

    minimind-o/trainer/train_sft_omni.py::omni_collate_fn
    minimind-o/trainer/train_sft_omni.py::train_epoch

训练主线：

    Dataset
    -> collate
    -> 9 路 input_ids
    -> MiniMindOmni.forward
    -> text_logits + 8 路 audio_logits
    -> text_loss + audio_loss + aux_loss
    -> backward
    -> optimizer.step

实践：跑一个 batch，分别打印 text loss、audio loss、梯度和参数变化，验证
一次 checkpoint 保存与加载。

验收：能解释 -100 mask、audio stop token 的加权，以及多路 loss 如何形成一个
最终标量。

#### 第 17 天：训练模式和冻结策略

对照两组参数：

    mode=audio_proj / vision_proj / all
    freeze_backbone=all / last1 / none

实践：对每种组合打印可训练参数名和参数量，实际检查哪些参数产生梯度。

推荐理解的训练顺序：

    T2A 全量训练
    -> A2A 只训练 AudioProjector
    -> A2A 全量联合训练
    -> I2T 只训练 VisionProjector
    -> I2T 联合训练
    -> A2A/I2T 小学习率巩固

验收：给出任意训练命令，能够判断哪些模块更新，并说明每阶段权重为何从前
一阶段继续加载。

#### 第 18 天：mini 数据完整训练

实践顺序：

1. 用极少样本进行一次可控训练。
2. 保存独立实验权重，不覆盖正式权重。
3. 比较训练前后的固定文本、audio code 和解码音频。
4. 再决定是否运行仓库给出的 mini 完整训练管线。

验收标准不是“命令没有报错”，而是同时观察 loss、梯度、参数变化、checkpoint
加载和推理结果，形成完整闭环。

### 单元六：语音产品能力（第 19-21 天）

#### 第 19 天：流式 audio code 到 PCM

代码入口：

    minimind-o/model/model_omni.py::MiniMindOmni.stream_generate
    minimind-o/webui/web_demo.py::stream_pcm
    minimind-o/webui/web_demo.py::pcm_bytes

理解两层流式：

    模型流式生成 audio frame
    -> Mimi 分块 decode
    -> PCM 字节
    -> 浏览器边接收边播放

实践：记录首个文本 token、首个完整 audio frame 和首段可播放 PCM 的时间。

#### 第 20 天：音色克隆

代码入口：

    voice_args
    build_clone_voice
    CAMPPlus speaker embedding
    Mimi reference codes

实践：对比默认音色、内置音色和参考音频克隆音色；分别移除 ref_codes 和
spk_emb，观察两种条件的作用。

验收：能够说明音色控制是推理时的条件输入，不是每次都重新训练模型。

#### 第 21 天：VAD、实时打断和近双工

代码入口：

    minimind-o/model/model_omni.py::SileroVAD
    minimind-o/model/model_omni.py::RealtimeSession
    minimind-o/webui/web_demo.py::realtime

状态链路：

    listening
    -> speaking
    -> speech_end
    -> generating
    -> interrupt

实践：先用离线音频块测试状态机，再验证一次正常 WebSocket 对话和一次中途打断。

验收：能区分模型推理、VAD 检测和 WebSocket 调度，并说明“近双工”不等于真正
同时持续听和说。

## 五、最终验收日（第 22 天，可按需要拆分）

不增加新概念，只做闭环验收：

- 从 eval_omni.py 追到模型输出和 Mimi 解码。
- 从一条 Parquet 样本还原 9 路输入和对应 labels。
- 解释 forward() 中 Thinker、Bridge、Talker 的数据流。
- 解释 stream_generate() 中文本 token、8 路 code 和完整 frame 的时间关系。
- 运行一次最小训练并证明参数更新。
- 完成一个小改动并给出修改前后对照，例如调整 bridge_layer、stop loss 权重、
  scheduled sampling 或流式解码块大小。

只有完成解释、实践、结果分析和复述，才把第三阶段标记为完成。

## 六、当前恢复点

当前刚开始第一个概念，已经完成一次初步复述，但还需要纠正两个术语：

    SenseVoice 不是直接把声音变成离散 audio code。
    它输出连续的音频特征，之后经 AudioProjector 进入 Thinker。

    Talker 不是把文本 token 解码成声音。
    它根据 Thinker 的 bridge hidden state 和历史 audio code，预测新的 audio code。

当前准确链路是：

    声音波形
    -> SenseVoice：波形 -> 连续音频特征
    -> AudioProjector：音频特征 -> MiniMind hidden state
    -> Thinker：理解并生成文本相关结果
    -> Bridge hidden state + 历史 audio code
    -> Talker：预测 8 层 audio code
    -> Mimi：audio code -> 声音波形

下一次从这个问题继续，不进入下一个概念，直到能够区分：

    SenseVoice 的连续特征
    Mimi 的离散 audio code
