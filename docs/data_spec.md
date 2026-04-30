# 训练数据规范

---

## 1. 文件格式

JSONL（每行一个 JSON 对象），UTF-8 编码，兼容 ShareGPT 格式。

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...", "train_loss": true, "value": "..."}, ...]}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `conversations` | array | 是 | 对话轮次列表，user/assistant 交替 |
| `role` | string | 是 | `"user"` 或 `"assistant"` |
| `content` | string | 是 | 对话内容 |
| `train_loss` | bool | 否 | assistant 轮是否计算 loss（默认 true） |
| `value` | str \| list[str] | 否 | 5× VALUE 权重 token 对应的字符串；list 用于多 fact 单句 |

### 约束

- 轮次 user/assistant 严格交替，user 在前
- `content` 不含特殊 token（`<|im_start|>` 等由 tokenizer 自动添加）
- `value` 字符串必须出现在 `content` 里才能被 tokenize 层定位到

### 示例

**单 value**：
```json
{
  "role": "assistant",
  "content": "你叫陈杰。",
  "train_loss": true,
  "value": "陈杰"
}
```

**多 value（多 fact 一句话）**：
```json
{
  "role": "assistant",
  "content": "好的陈杰，35 岁的北京人，都记下了。",
  "train_loss": true,
  "value": ["陈杰", "35", "北京"]
}
```

**拒答（无 value）**：
```json
{
  "role": "assistant",
  "content": "你还没告诉我你的名字呢。",
  "train_loss": true
}
```

---

## 2. 数据处理管线

```
JSONL 文件
  ↓ ConversationDataset
每 2 条 message (user+assistant) = 1 个 segment
  ↓ tokenize_turn()
apply_chat_template → token ids + label masking + weight 分配
  ↓
(input_ids, labels, weights) tensor 三元组，padding 到 segment_length
  ↓ collate_episodes()
batch: [(ids_batch, labels_batch, weights_batch), ...] 每个 shape (B, T)
```

### tokenize_turn 输出

| 位置 | labels | weights |
|---|---|---|
| user / chat template | `-100` | `0.0` |
| assistant 普通 token | 实际 token id | `1.0` |
| assistant value token（命中 `value` 字符串）| 实际 token id | `5.0` (VALUE_WEIGHT) |
| padding | `-100` | `0.0` |

多 value 支持：`value` 为 list 时，每个子串独立定位 offset 并打 5× 权重；空 list / None / str 都兼容。

### chat template

所有 backbone 统一使用 ChatML：

```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
{content}<|im_end|>
```

`ensure_chat_template()` 强制覆盖 tokenizer 的 chat_template，避免 Qwen3 等模型自带的 `<think>` 干扰。

---

## 3. Episode 结构

一条 JSONL = 一个 episode = 一段连续多轮对话。

训练时 state（Delta Rule W）在 episode 内 segment 间传递，episode 之间重置。

| 概念 | 对应 |
|------|------|
| 1 条 JSONL | 1 个 episode |
| 1 轮 user+assistant | 1 个 segment |
| segment 数量上限 | `episode_length`（默认 16） |
| segment token 长度 | `segment_length`（默认 256） |

---

## 4. Persona 数据（当前主数据）

**生成器**：`xinhe/data/generate_persona_data.py`，统一入口 `scripts/generate_data.py` 通过 `type: persona` 分发。

### 4.1 Persona 结构

每个 episode 采一个 Persona，包含 8 个槽（name / age / city / food / job / hobby / pet / number）的 ground truth，以及 `reveal_order`（随机 4-6 个槽的披露顺序）。未披露的槽是 refusal 候选源。

```python
@dataclass
class Persona:
    name: str; age: str; city: str; food: str
    job: str; hobby: str; pet: str; number: str
    reveal_order: list[str]    # 本 episode 计划披露的槽
    revealed: set[str]          # 对话过程中累积增长
    third_party: dict           # 25% 概率带第三方人物
```

### 4.2 10 种 turn kind（`DEFAULT_TURN_MIX`）

详见 [curriculum_learning.md](curriculum_learning.md) "Stage 1 的 turn kind 分布"。

### 4.3 DeepSeek 对话语料

`distract_chat` 和 `world_qa` 两种语料统一收纳到 `xinhe/data/dicts/files/*.jsonl`，
按 SHA1(user) % 10 切到 train/val/test，可直接 `bank.load_pairs(name, split)` 消费。

生成方式：
```bash
# 需要 DEEPSEEK_API_KEY
python scripts/build_dicts.py --expand --target-pairs 5000
```

输出：
- `xinhe/data/dicts/files/distract_chat.jsonl`：日常闲聊 turn（事件 E 消费，train_loss=lm_only）
- `xinhe/data/dicts/files/world_qa.jsonl`：事实 Q&A turn（Stage 1 1B 子流消费，train_loss=true，无 VALUE）

质量过滤在 PairBank 加载时做（user 字符串去重）；采样阶段做主题分散 + 长度约束。

### 4.4 结构化 retention pattern

`generate_persona_data.py` 里的 `generate_stress_retention_episode` 和 `generate_multi_slot_retention_episode` 产生高度结构化的 episode：

- `stress_retention`：reveal A → chat × 2-5 → recall A（单槽 retention）
- `multi_slot_retention`：reveal A → reveal B [→ reveal C] → chat × 2-5 → recall 全部

通过 config 里的 `stress_retention_ratio` / `multi_slot_retention_ratio` 控制比例。

### 4.5 合成池规模

8 个槽的 ground truth 素材池（见 `generate_memory_data.py`）：
- name: ~30k 种组合（姓 × 名字）
- age: 1-99
- city: 200+ 中国地级市
- food: 做法 × 食材 ≈ 320 种 + 经典菜 26 种
- job: 前缀 × 职业 ≈ 240 种 + 基础职业 32 种
- hobby: 修饰 × 活动 ≈ 240 种 + 基础爱好 30 种
- pet: 颜色 × 动物 ≈ 200 种 + 基础宠物 20 种
- number: 1-6 位随机数字（10^6 上限）

每 episode 采一次值，所以 LoRA 无法记忆具体答案，只能学"从 state 读"的通用技能。

---

## 5. Val 集（4 指标联合早停用）

`scripts/build_val_sets.py` 一次性生成 4 个 val jsonl：

| 文件 | 用途 | 默认数量 | 结构 |
|---|---|---|---|
| `data/val/val_value.jsonl` | VALUE/FRAME/TELL breakdown | 200 ep | 常规 persona episode |
| `data/val/val_worldqa.jsonl` | WorldQA 准确率 | 150 ep | 单轮 Q/A（从 world_qa cache 抽） |
| `data/val/val_refusal.jsonl` | Refusal 率 | 200 ep | 尾 turn 强制 refusal 的 episode |
| `data/val/val_compositional.jsonl` | 多 fact 全对率 | 100 ep | 尾 turn 强制 reveal_multi |

生成命令：
```bash
python scripts/build_val_sets.py --out-dir data/val
```

Val 集 seed 固定（12345），跨训练可比。

---

## 6. 合成记忆数据（Stage 0a fact bootstrap 用）

**生成器**：`xinhe/data/generate_memory_data.py`（纯 memory 数据生成，persona_unified 的 Stage 0a fact bootstrap 用于教 W_fact 的 Delta Rule 读写原语）。

### 6.1 生成脚本

```bash
# 默认参数
python -m xinhe.data.generate_memory_data --preview 3

# 完整 CLI 通过 scripts/generate_data.py
python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --stage 0a_fact_bootstrap
```

### 6.2 Episode 结构模板

Bootstrap stage 用最简单的 2-turn：
```
[告知 fact] → [recall fact]
```

生成器支持更复杂的 episode 类型（多 fact × 多 entity × 不同距离 × 覆写 × 对话回忆），持久化在代码中但当前 Stage 0a 只用最简形式（1 fact、2 turn、no filler）。

---

## 7. 加入真实数据

真实对话数据（ShareGPT 等）只需满足 JSONL + ChatML 格式。但当前训练流程是 persona 驱动，**真实数据不在 data pipeline 里**。

如要引入，建议：
1. 加一个新 turn kind（如 `real_dialogue`）到 `generate_persona_data.py`
2. 在 turn 状态机里 pop from 预处理的真实数据池
3. 在 `curriculum_persona.yaml` 加对应 ratio

不建议直接 append 到训练集 —— 会破坏 persona 驱动的 state 演化结构。

---

## 8. 文件组织

```
xinhe/data/dicts/files/           # 入 git：素材根（小，可复现根）
├── surnames.txt 等 12 个 .txt    # 字符串字典
├── distract_chat.jsonl            # 闲聊语料 (事件 E)
├── world_qa.jsonl                 # 世界知识 QA (Stage 1 1B)
└── version.json                   # 版本与切分清单

data/                              # 不入 git:训练数据(可重生成)
├── skeleton/
│   ├── train.jsonl              # 训练集(dict_split=train)
│   └── val.jsonl                # 验证集(dict_split=val,即 OOD entity)
├── dialog/
│   ├── train.jsonl
│   └── val.jsonl
└── mix/
    ├── train.jsonl
    └── val.jsonl
```

词典/语料统一在 `xinhe/data/dicts/files/`(入 git,由 `scripts/build_dicts.py` 维护),训练数据在 `data/`(不入 git,由 `scripts/generate_data.py` 按需生成)。

路径通过 `data:` 段配置（见 `configs/persona_unified_0.8b.yaml`）。
