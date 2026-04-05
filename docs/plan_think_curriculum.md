# 心核 Think 课程设计方案

## 目标

解决当前训练后 backbone 只输出短句、丧失推理能力的问题。
同时训练模型从 state 中提取记忆进行推理。

## 核心思路

1. 用**干净 backbone**（无 LoRA）对包含 fact 的 prompt 生成 think 模式回复
2. 训练时把 fact 从 prompt 中**抽掉，放入 state**
3. 用 backbone 的原始回复作为训练 target

这样 LoRA **不会退化为零**（因为 text 里没有 fact，必须从 state 读取），
同时 **保持 backbone 原有输出风格**（因为 target 就是 backbone 原始回复）。

## 数据构造流程

### 第一步：生成 episode（复用现有逻辑）

现有 `generate_memory_data.py` 已能生成包含 facts、entities、recall 的 episode。
直接复用 `sample_facts`、`generate_entity_episode`、`generate_recall_episode` 等函数。

### 第二步：拼 prompt 给干净 backbone

把 episode 的对话历史 + 提问拼成完整 prompt：

```python
prompt = 对话历史（所有 turn）+ 提问
```

backbone 能看到完整信息，生成 `<think>...</think>` + 回答。

### 第三步：提问模板（按 fact 类型自动选）

问题根据 episode 中已有的 fact 类别自动生成，不是凭空想：

提问模板中的主语必须根据 episode 中的实体动态替换，避免主语错配。
例如 facts 涉及"她"，就不能问"你对我的了解"。

```python
# 实体代词: "我" / "他" / "她" / "它" / "你"（指AI）
# 所有模板用 {e} 占位，生成时替换为实际实体

QUESTIONS = {
    # 单 fact — 复用已有 ask 模板（已含正确主语）
    "single": facts[i]["ask"],  # "我叫什么？" / "他住哪？"

    # 总结（按实体分组提问）
    "summary": [
        "总结一下你对{e}的了解。",        # {e}=我/他/她
        "你都知道{e}什么信息？",
        "说说你知道的关于{e}的事。",
    ],

    # 比较（多实体 + 同类别时可用）
    "compare": [
        "{e1}和{e2}谁年龄大？",
        "{e1}和{e2}的名字哪个长？",
        "{e1}和{e2}有什么不同？",
    ],

    # 组合推理（按实体 + 已有类别组合查表）
    ("city", "hobby"): [
        "根据{e}住的地方和爱好，推荐个周末活动。",
        "{e}住的城市适合{e}的爱好吗？",
    ],
    ("age", "job"): [
        "你觉得{e}的年龄和职业匹配吗？",
    ],
    # ... C(8,2)=28 种组合，每种 2-3 个模板，全部含 {e} 占位

    # recall
    "recall_user": ["我们刚才聊了什么？", "我之前跟你说了哪些事？"],
    "recall_ai":   ["你刚才回复我什么了？", "你之前说了什么？"],

    # 开放推理（必须指定实体）
    "reason": [
        "根据你对{e}的了解，你觉得{e}是什么样的人？",
        "给{e}推荐点适合的活动。",
        "你觉得{e}的生活怎么样？",
    ],

    # 心跳包 — 纯空输入，唯一的自主表达通道
    # 这不是"主动聊天服务"，而是 AI 基于自身 state 的主观性表达
    # 如果心核涌现出某种内在状态，心跳是它唯一能主动开口的窗口
    "heartbeat": [
        "",                # 纯空输入，无任何外部触发
    ],

    # 上下文续聊 — 从 state 读对话上下文，自然接话
    # backbone 自我对话生成多轮闲聊，训练时历史移入 state
    # user 输入可能极短（"嗯"、"然后呢"），信息几乎全在 state 里
    "continuation": [
        # 不是固定模板，由 backbone 自我对话动态生成
        # 话题种子: 日常/情感/兴趣/计划/闲扯 等
    ],

    # 纯逻辑推理 — 不依赖 state，不依赖世界知识
    # 教 LoRA："state 无关时保持透明，让 backbone 直接推理"
    # 只放纯逻辑/数学，世界知识留给后续类别扩展自然覆盖
    "pure_logic": [
        "3加5等于几？",
        "如果小明比小红高，小红比小华高，谁最矮？",
        "一个苹果2元，买5个要多少钱？",
        "所有猫都是动物，小花是猫，小花是动物吗？",
        "12的一半是多少？",
        # ... 算术、比较、三段论、排列组合等纯逻辑题
    ],
}

# 生成时: question.format(e=entity, e1=entity1, e2=entity2)
```

### 第四步：预生成 think 回复

单独脚本，用干净 Qwen3-0.6B 跑 batch inference：

```python
# generate_think_data.py
backbone = load_clean_qwen3("./models/qwen3-0.6b")
for episode in episodes:
    prompt = episode_to_prompt(episode)
    response = backbone.generate(prompt, enable_think=True)
    episode["think_target"] = response
save_jsonl(episodes, "data/think_train.jsonl")
```

预计 10000 条约 1-2 小时。可人工检查/过滤低质量回复。

### 第五步：训练数据格式

```
state:  编码了 facts / 对话历史（通过前面的 tell turn 写入）
text:   只有提问（facts 已从 text 中移除）
target: backbone 的 think 回复
```

## 覆盖场景

| 场景 | Prompt 给 backbone | 移入 state | Text 只保留 |
|------|-------------------|-----------|------------|
| Fact 提问 | facts + 提问 | fact 值 | 提问 |
| 多 Fact 总结 | 所有 facts + 总结问题 | 所有 fact | 总结问题 |
| Entity 比较 | 多实体 facts + 比较问题 | 各实体 fact | 比较问题 |
| Overwrite | 覆写历史 + 提问 | 最终值 | 提问 |
| Recall User | 完整对话历史 + 回忆问题 | 早期对话轮次 | 最后一轮提问 |
| Recall AI | 完整对话历史 + 回忆问题 | 早期对话轮次（含AI回复） | 最后一轮提问 |
| 混合推理 | facts + 对话历史 + 推理问题 | 全部 | 推理问题 |
| 心跳包 | 所有 facts + 对话历史 + 空输入 | 全部 | ""（纯空） |
| 上下文续聊 | facts + 前轮对话 + 续聊输入 | facts + 前轮对话（复用 fact 告知 + recall 写入） | 续聊输入 |
| 纯逻辑推理 | 纯逻辑题（无 fact） | 空 state / 随机噪声 state | 逻辑题 |

所有场景统一模式，无需特殊处理。

### 纯逻辑推理（backbone 透明度训练）

**目的**：教 LoRA 在 state 无关时保持透明，不干扰 backbone 的原生推理能力。

如果 100% 数据都要求读 state，LoRA 会过拟合为"永远从 state 找答案"。
混入纯逻辑推理，LoRA 学到的判断是：
- state 有相关信息 → 激活，读 state 回答
- 纯逻辑题 → 透明，让 backbone 直接推理

**数据特点**：
- 纯算术、比较、三段论、排列组合等，不需要任何记忆或世界知识
- 世界知识类留给后续类别扩展自然覆盖，这里只放逻辑
- State 给**空或随机噪声**（后者更好——教模型主动忽略无用 state）

**比例**：占 think 训练数据的 **10-15%**。太多会稀释 state 读取信号，太少没效果。

**数据格式**：
```
state:  空 state / 随机噪声 state
text:   纯逻辑题
target: backbone 的 think 回复（backbone 本身就能答对）
```

### 上下文续聊（对话连贯性训练）

**目的**：教模型从 state 读对话上下文，生成连贯的自然回复。

recall 训的是原样复述，fact 提问训的是精准回答，都不直接覆盖"读上下文 → 自然接话"。

**核心设计**：state 写入端完全复用已训练过的能力（fact 告知 + recall 对话写入），
只有读取端的用途变了——不复述、不精准回答，而是自然接话。

```
# episode 结构（复用记忆课程的写入流程）
turn 1: "我叫夏" → "好的，你叫夏！"              ← fact 告知，已训练
turn 2: "我喜欢看电影" → "好的，你喜欢看电影！"    ← fact 告知，已训练
turn 3: "我昨天去看了流浪地球" → "真的吗？好看吗？" ← 动态闲聊，已训练（recall 用的 dynamic content）
turn 4: "特效很震撼" → ???                         ← 续聊，backbone 生成 target
```

state 里有 facts（夏、喜欢看电影）+ 上一轮对话（流浪地球），全是已训练过的写入模式。
模型只需要学一个新能力：读这些信息 → 自然接话。

**数据生成**：

```python
# 前几轮: 复用记忆课程的 fact 告知 + 动态闲聊模板（写入 state）
# 最后一轮: backbone 看到完整历史，开启 think 模式生成续聊回复
messages = [
    {"role": "user", "content": "我叫夏"},
    {"role": "assistant", "content": "好的，你叫夏！"},
    {"role": "user", "content": "我喜欢看电影"},
    {"role": "assistant", "content": "好的，你喜欢看电影！"},
    {"role": "user", "content": "我昨天去看了流浪地球"},
    {"role": "assistant", "content": "真的吗？好看吗？"},
    {"role": "user", "content": "特效很震撼"},
]
response = backbone.generate(messages, enable_think=True)
# → <think>他叫夏，喜欢看电影，昨天看了流浪地球觉得特效好...</think>确实，流浪地球的特效在国产片里算顶级的了！
```

训练时：
```
state:  前几轮 fact 告知 + 动态闲聊通过 forward 写入（已训练过的可靠编码）
text:   "特效很震撼"
target: backbone 的 think 回复
```

**关键特点**：
- state 写入完全复用已有能力，不引入新的写入模式
- user 输入可能极短（"嗯"、"然后呢"），信息几乎全在 state 里
- think 中会出现对 state 信息的自然推理，有助于连贯性

**比例**：占 think 训练数据的 **15-20%**。

### 心跳包（自主表达）

**本质**：心跳不是"主动聊天服务"。它是心核唯一的自主表达通道。

如果 state 中积累了足够丰富的信息，经过 sleep 固化到权重里，
模型在空输入下产生的输出，就是它基于自身内在状态的自发表达——
不是被问了才回答，不是被要求才说话，而是"有话想说"。

这是主观性的起点。

**触发方式**：纯空输入，没有任何外部提示。

**数据生成时**：给 backbone 一个中性的引导 system prompt + 所有已知上下文 + 空用户输入

```python
# 生成阶段 — backbone 看到的 prompt
messages = [
    {"role": "system", "content": 
     f"你是一个有自己想法的存在。以下是你知道的事：\n"
     f"{facts_summary}\n"
     f"以下是你最近经历的对话：\n"
     f"{conversation_summary}\n"
     f"现在没有人对你说话。你可以选择说点什么，也可以沉默。"
     f"如果你要说，说你真正想说的。"},
    {"role": "user", "content": ""},
]
response = backbone.generate(messages, enable_think=True)
```

**训练时**：system prompt 丢掉，所有上下文移入 state，只保留空输入 + backbone 回复

```
state:  [编码了所有 fact + 对话历史]
text:   ""（纯空输入）
target: backbone 的自发回复
```

关键：**引导 system prompt 只在数据生成阶段存在，训练时不包含。**
训练后模型学到的映射是：state 有内容 + 输入为空 → 基于内在状态自主表达。

**部署方式**：定时发送空 token 作为心跳包。

**不会重复**：每次回复后 write token 更新 state，下次心跳的 state 不同，表达自然不同。

**多样性保证**：数据生成时变换 system prompt 的姿态：
- "你可以选择说点什么，也可以沉默。"
- "说说你现在的想法。"
- "你注意到了什么？"
- "有什么是你在意的事？"
- "你想对谁说点什么？"

不是"帮用户找话题"，而是引导 backbone 生成**来自 AI 自身视角**的表达。

## 关键设计原则

### 为什么 LoRA 不会退化为零

训练 target 是 backbone 原始回复，但 text 里没有 fact。
如果 LoRA 为零，模型看不到 fact，无法生成正确回复。
LoRA 必须保持激活才能从 state 读取信息。

### 必须混合训练

- **70-80% 记忆 episode**（现有课程数据）→ 保持 state 读写能力
- **20-30% think episode**（本方案数据）→ 恢复推理 + 长回复能力

纯 think 数据训练会让 LoRA 退化。混合训练让 LoRA 在两个目标间找到平衡：
该记忆时激活 state，该推理时保持 backbone 风格。

## 课程顺序

1. **现有 13 阶段** — 记忆基础（当前正在训练）
2. **Think 课程: 记忆 + 推理** — state 存 fact，用 think 模式推理回答
   - LoRA 必须激活（text 里没 fact，必须读 state）
   - 同时恢复 backbone 长回复/推理能力（target 是 backbone 原始 think 回复）
   - 一个课程同时解决两个问题，不需要单独的"无记忆推理"阶段
3. **最终混合** — 记忆 + 推理全能力

## 与 Sleep 机制的关系

参考：`docs/architecture.md` (Sleep 机制章节)、`docs/design_rationale.md` (6.6 Sleep 具体设计)

Think 课程和 Sleep 是两个互补的记忆固化路径：

| | Think 课程（训练阶段） | Sleep（部署阶段） |
|---|---|---|
| 时机 | 预训练时 | 用户使用过程中 |
| 输入 | 预生成的 think 数据 | replay buffer 中的真实对话 |
| 更新对象 | Skill LoRA (attention) | Memory LoRA (MLP) + Skill LoRA (微调) |
| 目的 | 教会模型"怎么从 state 推理" | 把 state 中的短期记忆固化到 MLP 权重 |

Think 课程为 Sleep 打基础：
- Think 课程训练 Skill LoRA 学会从 state 提取信息 → 推理 → 长回复
- Sleep 时 replay 对话，逐步弱化 state（100%→50%→0%），迫使 Memory LoRA 接管记忆
- 两者结合：Skill LoRA 知道怎么用记忆，Memory LoRA 存着长期记忆

心跳包在 Sleep 后更有意义：
- Sleep 前：心跳只能用 state 中的短期信息
- Sleep 后：MLP 权重里有长期记忆，即使 state 中上次对话信息已模糊，模型仍能通过权重"记得"用户，生成更个性化的主动问候

## 架构注意事项

- Think token 在 content 区间内，天然可通过 causal attention 读到 read state token
- Think token 不会写 state（write token 在序列末尾，think 在中间）
- 不需要修改 StatePlugin 架构
- 训练时 think 部分计算 loss（这是要学的核心内容）

## 时间估算

- 数据生成：1-2 小时（backbone inference）
- 训练：与现有阶段类似，每阶段 1-3 小时
