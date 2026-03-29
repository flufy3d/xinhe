# 训练数据规范

---

## 1. 文件格式

JSONL（每行一个 JSON 对象），UTF-8 编码，兼容 ShareGPT 格式。

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `conversations` | array | 是 | 对话轮次列表，user/assistant 交替 |
| `conversations[].role` | string | 是 | `"user"` 或 `"assistant"` |
| `conversations[].content` | string | 是 | 对话内容，纯文本 |

### 约束

- `conversations` 长度必须 >= 4（至少 2 轮 user+assistant）
- 轮次必须 user/assistant 严格交替，user 在前
- 总轮次数建议为偶数（完整的 user+assistant 对）
- content 不含特殊 token（`<|im_start|>` 等由 tokenizer 自动添加）

### 示例

```json
{
  "conversations": [
    {"role": "user", "content": "我叫张三，住在北京。"},
    {"role": "assistant", "content": "你好张三！北京是个好地方。"},
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "assistant", "content": "今天天气不错，适合出门。"},
    {"role": "user", "content": "你还记得我叫什么吗？"},
    {"role": "assistant", "content": "你叫张三。"}
  ]
}
```

---

## 2. 数据处理管线

```
JSONL 文件
  ↓ ConversationDataset
每 2 条 message (user+assistant) = 1 个 segment
  ↓ tokenize_turn()
apply_chat_template → token ids + label masking
  ↓
(input_ids, labels) tensor pair，padding 到 segment_length
  ↓ collate_episodes()
batch: [(ids_batch, labels_batch), ...] 每个 shape (B, T)
```

### tokenize 流程

1. 用 `tokenizer.apply_chat_template` 将 user+assistant 转为 token 序列
2. 两步定位 assistant 起始位置：
   - Step 1: tokenize `[user_msg]` + `add_generation_prompt=True` → `prefix_len`
   - Step 2: tokenize `[user_msg, assistant_msg]` → `full_ids`
3. 构建 labels:
   - `[0, prefix_len)`: `-100`（user + template token，不计算 loss）
   - `[prefix_len, end)`: 实际 token id（assistant 回答，计算 loss）
   - padding 位置: `-100`

### chat template

所有 backbone 统一使用 ChatML 格式：

```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
{content}<|im_end|>
```

- Qwen: tokenizer 自带 chat_template，直接使用
- MiniMind: 无 chat_template，由 `ensure_chat_template()` 自动设置 ChatML fallback

---

## 3. Episode 结构

一条 JSONL = 一个 episode = 一段连续多轮对话。

训练时 state 在 episode 内的 segment 间传递，episode 之间 state 重置。

| 概念 | 对应关系 |
|------|---------|
| 1 个 JSONL 行 | 1 个 episode |
| 1 轮 user+assistant | 1 个 segment |
| segment 数量上限 | `episode_length`（默认 16） |
| segment token 长度 | `segment_length`（默认 256） |

---

## 4. 合成记忆数据

### 4.1 生成脚本

```bash
# 默认生成
python scripts/generate_memory_data.py

# 自定义参数
python scripts/generate_memory_data.py \
  --num-train 5000 \
  --num-val 500 \
  --num-facts 2 \
  --min-distance 1 \
  --max-distance 6 \
  --max-turns 16 \
  --seed 42

# 预览（不写文件）
python scripts/generate_memory_data.py --preview 3
```

### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-train` | 2000 | 训练集 episode 数量 |
| `--num-val` | 200 | 验证集 episode 数量 |
| `--num-facts` | 1 | 每个 episode 注入的事实数量 |
| `--min-distance` | 1 | 告知→回忆的最小间隔轮数 |
| `--max-distance` | 4 | 告知→回忆的最大间隔轮数 |
| `--max-turns` | 16 | 每个 episode 最大轮数 |
| `--seed` | 42 | 随机种子（可复现） |
| `--out-dir` | `data` | 输出目录 |

### 4.3 Episode 结构模板

```
[告知事实 x num_facts]  →  [闲聊填充 x distance]  →  [回忆提问 x 1]  →  [闲聊补充至 max_turns]
```

示例（num_facts=1, distance=2）：

| Turn | 类型 | User | Assistant |
|------|------|------|-----------|
| 1 | 告知 | 我叫张三。 | 好的，张三，很高兴认识你！ |
| 2 | 填充 | 今天天气怎么样？ | 今天天气不错... |
| 3 | 填充 | 推荐一部电影吧。 | 推荐《星际穿越》... |
| 4 | **回忆** | **我叫什么名字？** | **你叫张三。** |
| 5-16 | 填充 | （闲聊） | （闲聊） |

### 4.4 记忆类别

| 类别 | 素材池大小 | 示例值 |
|------|-----------|--------|
| name | 16 | 小明、张三、梓涵 |
| city | 16 | 北京、成都、大连 |
| food | 16 | 火锅、螺蛳粉、红烧肉 |
| color | 8 | 红色、紫色、橙色 |
| hobby | 16 | 跑步、弹吉他、写代码 |
| pet | 8 名字 x 4 种类 | 猫叫旺财、狗叫豆豆 |
| job | 8 | 程序员、医生、律师 |

每个类别有 2-3 种陈述模板和 2-3 种提问模板，避免模型只学会固定句式。

---

## 5. 加入真实数据

真实对话数据（ShareGPT、firefly 等）只需满足上述 JSONL 格式，直接追加到 `train.jsonl`：

```bash
# 追加真实数据
cat real_conversations.jsonl >> data/train.jsonl
```

### 数据来源建议

| 数据集 | 语言 | 格式 | 用途 |
|--------|------|------|------|
| ShareGPT (HuggingFace) | 中英 | ShareGPT | 通用多轮对话 |
| firefly | 中文 | 需转换 | 中文指令跟随 |
| belle | 中文 | 需转换 | 中文对话 |

### 格式转换要点

如果外部数据不是 `{"conversations": [...]}` 格式，转换时确保：

1. role 只用 `"user"` 和 `"assistant"`（不用 `"human"` / `"gpt"` 等）
2. 去掉 system message（当前不支持 system role）
3. 确保 user/assistant 严格交替

---

## 6. 文件组织

```
data/
├── train.jsonl          # 训练集（合成 + 真实混合）
├── val.jsonl            # 验证集
└── README               # (可选) 数据版本说明
```

路径在 `configs/base.yaml` 中配置：

```yaml
data:
  train_path: "./data/train.jsonl"
  val_path: "./data/val.jsonl"
```

---

## 7. 各 Milestone 数据配置建议

| Milestone | num_facts | distance | 数据量 | 备注 |
|-----------|-----------|----------|--------|------|
| 3. 1轮记忆 | 1 | 1-4 | 2000 | 当前默认配置 |
| 4. 多轮记忆 | 1 | 1-12 | 5000 | 加大 distance 测试衰减曲线 |
| 5. 信息覆写 | 需新模板 | - | - | 需扩展生成脚本：同类别先说 A 再说 B |
| 6. Wipe 对比 | 同 3 | 同 3 | 同 3 | 数据不变，测试时 wipe state |
