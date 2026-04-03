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

所有场景统一模式，无需特殊处理。

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

## 架构注意事项

- Think token 在 content 区间内，天然可通过 causal attention 读到 read state token
- Think token 不会写 state（write token 在序列末尾，think 在中间）
- 不需要修改 StatePlugin 架构
- 训练时 think 部分计算 loss（这是要学的核心内容）

## 时间估算

- 数据生成：1-2 小时（backbone inference）
- 训练：与现有阶段类似，每阶段 1-3 小时
