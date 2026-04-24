# 课程学习：心核训练的核心策略

## 为什么需要课程学习

同时学"state 读写原语"和"真实对话中的使用"会陷入局部最优 —— plugin 还没学会 Delta Rule 读写时，LoRA 会学出"不用 state"的策略。从零端到端训练的早期（v1/v2 时代）失败多次都是这个原因。

**解法**：分阶段递增难度。每阶段只引入一个新维度，前一阶段已掌握能力保持稳定，LR 和目标函数只对新维度施加压力。

---

## 当前课程：persona_unified (v7 Hippocampus 3-stage)

`configs/curriculum_persona.yaml` 定义所有 stages，`configs/persona_unified_0.8b.yaml`（0.8b）和 `configs/persona_unified_4b.yaml`（4b）引用它 + 加 backbone-specific 的 batch/lr overrides。

```
Stage 0a: 0a_hippocampus_rw
  数据: type=memory, 1 fact（only name）, 2 轮 tell+recall, 无 filler, 无覆写
  训练: episode_length=2, freeze_lora=true, plugin_lr_mult=1.0
        max_steps=1500, early_stop_value=0.98（只看 VALUE）
  目标: Hippocampus 的 k_proj/v_proj/q_projs/o_projs/beta_proj 学会 Delta Rule 读写
  γ 预期: time_shift 自然不动（无 distractor 信号），γ 保持静态先验 σ(θ_h)

    ↓ Delta Rule 读写已通

Stage 0b: 0b_gamma_gating
  数据: type=persona, stress_retention(0.50) + multi_slot_retention(0.30) + meta/verbatim(0.20)
  训练: freeze_lora=true（保持）, plugin_lr_mult=1.0
        episode_length=8, max_steps=2500, early_stop_value=0.90
  目标: time_shift 学会"废话 token → 负偏移 → γ 变小 → 自动遗忘"
  核心监控: gamma_token_std 从 0 升到 > 0.05；distractor γ < fact γ

    ↓ γ 内容 gating 已学会

Stage 1: 1_persona_unified
  数据: type=persona, 10-16 轮 persona 驱动对话 + 结构化 retention patterns
  训练: freeze_lora=false, plugin_lr_mult=0.3
        max_steps=8000, 6 指标联合早停
  目标: LoRA + plugin 协同；retention/compositional/refusal 全面达标
```

### Stage 1 的 turn kind 分布（`DEFAULT_TURN_MIX`）

| Kind | 概率 | train_loss | value | 说明 |
|---|---|---|---|---|
| general_chat | 34% | false | — | DeepSeek teacher 采样的闲聊，不算 loss，防 LoRA 漂移 |
| world_qa | 10% | true | — | 世界常识 Q&A，均匀 1.0 权重 |
| reveal_single | 14% | true | str | "我叫 X" 类单槽披露 |
| reveal_multi | 8% | true | list | "我叫 X, Y 岁" 多 fact 一句话 |
| recall | 15% | true | str | 问已披露槽 |
| refusal | 10% | true | —（空 value）| 问未披露 → 拒答（不打 5× 避免死记） |
| overwrite | 4% | true | str | "不对我叫 Y" 纠正 |
| third_party | 3% | true | str | "我朋友叫..." 第三方 |
| compositional | 2% | true | list | 跨槽组合问答 |

### 结构化 retention pattern

- **stress_retention**：reveal A → chat × 2-5 → recall A。强化"告知后穿插 chat 再召回"。
- **multi_slot_retention**：reveal A → reveal B [→ reveal C] → chat × 2-5 → recall 全部。强化多槽同时 retention。
- **adversarial_temporal**：v6 时序碰撞对抗集（作为 stress 诊断保留）。

### 6 指标联合早停

`trainer.py` 每 eval_every 步计算：

| 指标 | val 集 | 计算方式 | 默认阈值 |
|---|---|---|---|
| VALUE | 自动生成的 val.jsonl | recall 轮 value token argmax 准确率 | 0.95 |
| WorldQA | data/val/val_worldqa.jsonl | 单轮 QA 整体 token 准确率 | 0.82 |
| Refusal | data/val/val_refusal.jsonl | 预测含拒答 regex 关键词 / 无 fabrication | 0.92 |
| Compositional | data/val/val_compositional.jsonl | 多 fact 单句全部 value 命中 | 0.92 |
| RapidOverwrite | data/val/val_rapid_overwrite.jsonl | 末轮应召回最新值 | 0.85 |
| Decay | data/val/val_decay.jsonl | 远端应拒答（FORGET 模板）| 0.65 |

**所有 active 指标全过才触发早停**。

---

## 失败诊断路径（v7 课程的核心价值）

v7 把两个新机制拆到 2 个 bootstrap 阶段独立验证，失败时能立即定位问题：

### Stage 0a 失败（VALUE < 90%）

- 症状：简单 2 轮 tell+recall 都做不到 98%
- 根因：Hippocampus 的 Delta Rule 基础读写没学起来
- 检查：
  - `beta_mean`（是否一直接近 0？→ beta_proj 没学到"要写"）
  - `read_scale`（是否爬升？→ 模型是否真的用 W）
  - k_proj / v_proj 的 grad norm（是否在更新？）

### Stage 0b 失败（retention VALUE < 70%）

- 症状：单槽 retention 通过 2-5 轮 distractor 后忘记
- 根因：γ 内容 gating 未学出，模型无法区分 fact 和 chatter
- 检查：
  - `gamma_token_std` 是否 ≈ 0（time_shift 没激活）
  - distractor token 上的 γ 是否 ≈ fact token 的 γ（没区分）
  - 若 gamma_token_std = 0 → LR / 数据分布问题
  - 若 gamma_token_min 从未 < 0.5 → 模型没学会"减寿"

### Stage 1 失败（联合指标不过）

- 症状：0a+0b 都过了但 Stage 1 某指标持续低
- 根因：LoRA 放开后把 plugin 学到的东西"忘"了
- 检查：
  - `read_scale` 是否一直爬升（塌陷说明 LoRA 抄了近路）
  - `gamma_prior_range`（所有 head 是否还保持分散）
  - `γ_token_std` 是否维持 > 0.05（内容 gating 仍激活）

---

## 配置组合

| 目标 | 配置文件 | 入口 |
|---|---|---|
| 0.8b 端到端 | `persona_unified_0.8b.yaml` | `python scripts/train.py --config configs/persona_unified_0.8b.yaml` |
| 4b 端到端 | `persona_unified_4b.yaml` | `python scripts/train.py --config configs/persona_unified_4b.yaml` |
| 只跑 0a | 同上 + `--from-stage 0a_hippocampus_rw` | 省时调试 Delta Rule 读写 |
| 从 0b 开始 | 同上 + `--from-stage 0b_gamma_gating` | 跳过已完成的 0a |
| 从 Stage 1 开始 | 同上 + `--from-stage 1_persona_unified` | 跳过已完成的 0a + 0b |

`train.py` 的 auto-skip 逻辑会自动跳过 `checkpoints/curriculum/{stage_name}.pt` 已存在的 stage。要重跑，删掉 ckpt 或用 `--from-stage`。

---

## 关键调参经验

### 阈值的严松决定训练质量上限

```
宽松阈值（70 / 85 / 85 / 85）: step 1000 早停，但 WorldQA 真实 71% → 答错 30%
严阈值（95 / 85 / 92 / 92）:  step 3250 早停，WorldQA 真实 85% → 答错 15%
```

严阈值让模型继续训出细节。

### VALUE 爬升不是线性的

观察 v5c 4b 从零训练的 VALUE 曲线：
```
step 1000: 68   (plugin 摸到 Delta Rule)
step 3000: 78   (稳定爬)
step 5000: 85   (plateau)
step 7000: 85   (还卡)
step 8000: 96   (突破 plugin 训练死点)
step 9000: 98   (收敛)
```

中段的 plateau 不是训够了，是 plugin 和 LoRA 在协调写策略。**别过早停**。

### plugin_lr_multiplier 的取值

- Stage 0a/0b bootstrap: 1.0（plugin 需要大 LR 快速学会 Delta Rule + γ gating）
- Stage 1 main: 0.3（plugin 已会读写 + gating，大 LR 会震荡；LoRA 从零学需要 1.0）
- 如果从 `--resume` 继续训: 0.1（plugin 保护，LoRA 继续微调）

---

## 下一步可能的迭代

如果发现新失败 pattern，按这个模板加：

1. 在 `xinhe/data/generate_persona_data.py` 加 `generate_xxx_episode` 函数
2. 在 `generate_persona_episode` 加一个 ratio 参数 + 分发
3. 在 `curriculum_persona.yaml` stage 1 的 data 段加 `xxx_ratio: 0.xx`
4. 重跑同一 config

示例已有：`stress_retention_ratio`、`multi_slot_retention_ratio`、`adversarial_temporal_ratio`。未来可能需要：
- `bilingual_ratio`（跨语言 retention：英文问中文 value）
- `multi_answer_ratio`（一次回复多槽值："告诉我名字 年龄 来自哪里"）
- `temporal_ratio`（时序 retention：昨天告知今天召回）

每次新 pattern 约 50 行代码 + 重训一次 stage 1。架构不动。

---

## Phase 2 预告：Neocortex (夜晚模式)

当前 Hippocampus 是**短期工作记忆**（对话级、episode 末 reset）。Phase 2 将实现 **Neocortex** —— 通过 Sleep 机制把 W 中高 γ 存活的记忆蒸馏到 MLP LoRA 权重：

- **白天**（Phase 1 当前）：Hippocampus + Attention LoRA 运转
- **夜晚**（Phase 2 未来）：冻结 Hippocampus，开 MLP LoRA（gate_proj/up_proj/down_proj），用 replay buffer 蒸馏
- **翌日重启**：W 清空，保留永久烙印的 MLP 权重

详见 `docs/roadmap.md` "中期：长期记忆固化"。
