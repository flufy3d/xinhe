# 课程学习：心核训练的核心策略

## 为什么需要课程学习

同时学"state 读写原语"和"真实对话中的使用"会陷入局部最优 —— plugin 还没学会 Delta Rule 读写时，LoRA 会学出"不用 state"的策略。从零端到端训练的早期（v1/v2 时代）失败多次都是这个原因。

**解法**：分阶段递增难度。每阶段只引入一个新维度，前一阶段已掌握能力保持稳定，LR 和目标函数只对新维度施加压力。

---

## 当前课程：persona_unified（2 stage 端到端）

`configs/curriculum_persona.yaml` 定义所有 stages，`configs/persona_unified.yaml`（0.8b）和 `configs/persona_unified_4b.yaml`（4b）引用它 + 加 backbone-specific 的 batch/lr overrides。

```
Stage 0: 0_bootstrap
  数据: type=memory, 1 fact（only name）, 2 轮 tell+recall, 无 filler, 无覆写
  训练: episode_length=2, tbptt=2, freeze_lora=true, plugin_lr_mult=1.0
        max_steps=1500, early_stop_value=0.98（只看 VALUE）
  目标: plugin 学会 Delta Rule 基本读写原语
  耗时: ~15 min on 4b A100, ~5 min on 0.8b

    ↓ plugin 已会读写，LoRA 未动

Stage 1: 1_persona_unified
  数据: type=persona, 12-20 轮 persona 驱动对话
         10 种 turn kind 混合（见下）
         + 10% stress_retention + 10% multi_slot_retention 结构化 episode
  训练: episode_length=16, tbptt=8, freeze_lora=false, plugin_lr_mult=0.3
        max_steps=5000, 4 指标联合早停
  目标: 真实分布 + retention + refusal + 多 fact
  耗时: 4b 从 scratch 约 8000-10000 步收敛；0.8b 从 bootstrap 继续约 3000-5000 步
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

### 结构化 retention pattern（覆盖 turn mix 里学不到的）

- **stress_retention**（10%）：reveal A → chat × 2-5 → recall A。强化"告知后穿插 chat 再召回"的 retention。
- **multi_slot_retention**（10%）：reveal A → reveal B [→ reveal C] → chat × 2-5 → recall A/B/C。强化多槽同时 retention（用户实测发现的失败 pattern）。

生成器是 permanent infrastructure —— 新失败 pattern 加一个 `generate_*_retention_episode` + 给个比例参数，不改训练流程。

### 4 指标联合早停

`trainer.py` 每 eval_every 步计算：

| 指标 | val 集 | 计算方式 | 默认阈值 |
|---|---|---|---|
| VALUE | 自动生成的 val.jsonl | recall 轮 value token argmax 准确率 | 0.98 |
| WorldQA | data/val/val_worldqa.jsonl | 单轮 QA 整体 token 准确率 | 0.85 |
| Refusal | data/val/val_refusal.jsonl | 预测含拒答 regex 关键词 / 无 fabrication | 0.95 |
| Compositional | data/val/val_compositional.jsonl | 多 fact 单句全部 value 命中 | 0.95 |

**4 个全过才触发早停**。单个过了其他没过 → 继续训。严阈值（严比松好）是 persona_unified 迭代中的关键发现 —— 70% 太松会留大量 fabrication 空间。

---

## 配置组合

| 目标 | 配置文件 | 入口 |
|---|---|---|
| 0.8b 端到端 | `persona_unified.yaml` | `python scripts/train.py --config configs/persona_unified.yaml` |
| 4b 端到端 | `persona_unified_4b.yaml` | `python scripts/train.py --config configs/persona_unified_4b.yaml` |
| 只跑 bootstrap | 同上 + `--from-stage 0_bootstrap` | 省时调试 plugin 读写 |
| 从 bootstrap 后接 | 同上 + `--from-stage 1_persona_unified` | 跳过已完成的 bootstrap |

`train.py` 的 auto-skip 逻辑会自动跳过 `checkpoints/curriculum/{stage_name}.pt` 已存在的 stage。要重跑，删掉 ckpt 或用 `--from-stage`。

---

## 关键调参经验

### 阈值的严松决定训练质量上限

```
宽松阈值（70 / 85 / 85 / 85）: step 1000 早停，但 WorldQA 真实 71% → 答错 30%
严阈值（98 / 85 / 95 / 95）: step 3250 早停，WorldQA 真实 85% → 答错 15%
```

严阈值让模型继续训出细节。宽松阈值是 v5c 单 stage（非 persona）时的失败配置。

### VALUE 爬升不是线性的

观察 4b 从零训练的 VALUE 曲线：
```
step 1000: 68   (plugin 摸到 Delta Rule)
step 3000: 78   (稳定爬)
step 5000: 85   (plateau)
step 7000: 85   (还卡)
step 8000: 96   (突破 plugin 训练死点)
step 9000: 98   (收敛)
```

中段的 plateau 不是训够了，是 plugin 和 LoRA 在协调写策略。**别过早停**。

### bootstrap 有没有必要

4b 单 stage 从零能训到 98% 但花了 10000 步；加 bootstrap 后估计 5000-6000 步达同样水平。0.8b 的老 curriculum 就是 13-stage bootstrap，只是更碎片化。2-stage 是平衡点。

### plugin_lr_multiplier 的取值

- Stage 0 bootstrap: 1.0（plugin 需要大 LR 快速学会 Delta Rule）
- Stage 1 main: 0.3（plugin 已会读写，大 LR 会震荡；LoRA 从零学需要 1.0）
- 如果从 `--resume` 继续训（persona 二次迭代）: 0.1（plugin 保护，LoRA 继续微调）

---

## Legacy：v1-v5c 的老课程（保留作参考）

以下文件仍在仓库中，**但不是当前训练路径**：

| 文件 | 用途 | 状态 |
|---|---|---|
| `curriculum.yaml` | 13-stage memory curriculum（v5c） | 保留，legacy |
| `curriculum_think.yaml` | Think 课程 | 保留，**失败** (TELL=66%，已被 persona_unified 的 turn_mix 吸收) |
| `curriculum_migrate.yaml` | 基座迁移 M0-M3 | 保留，暂未验证 v5c 迁移 |
| `curriculum_qwen3.5-{0.8b,4b,9b}.yaml` | 老入口 | 保留，和 persona_unified 并存 |
| `think_qwen3.5-*.yaml` | 思考入口 | 保留，不推荐 |
| `migrate_*.yaml` | 迁移入口 | 保留，不推荐 |

要跑老 curriculum 做对比，直接用原文件即可 —— 和 persona_unified 不互相干扰。

---

## 下一步可能的迭代

如果发现新失败 pattern，按这个模板加：

1. 在 `xinhe/data/generate_persona_data.py` 加 `generate_xxx_episode` 函数
2. 在 `generate_persona_episode` 加一个 ratio 参数 + 分发
3. 在 `curriculum_persona.yaml` stage 1 的 data 段加 `xxx_ratio: 0.xx`
4. 重跑同一 config

示例已有：`stress_retention_ratio`、`multi_slot_retention_ratio`。未来可能需要：
- `bilingual_ratio`（跨语言 retention：英文问中文 value）
- `multi_answer_ratio`（一次回复多槽值："告诉我名字 年龄 来自哪里"）
- `temporal_ratio`（时序 retention：昨天告知今天召回）

每次新 pattern 约 50 行代码 + 重训一次 stage 1。架构不动。
