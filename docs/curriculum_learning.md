# 课程学习:心核训练的核心策略

## 为什么需要课程学习

同时学"state 读写原语"和"真实对话中的使用"会陷入局部最优 —— Hippocampus 还没学会 Delta Rule 读写时,LoRA 会学出"不用 state"的策略。从零端到端训练的早期(v1/v2 时代)失败多次都是这个原因。

**解法**:分阶段递增难度。每阶段只引入一个新维度,前一阶段已掌握能力保持稳定,LR 和目标函数只对新维度施加压力。

---

## 当前课程:v8 三阶段

`configs/curriculum_v8.yaml` 定义所有 stages,`configs/persona_unified_v8_0.8b.yaml`(0.8b)和 `_4b.yaml`(4b)引用它 + 加 backbone-specific 的 batch overrides。

```
Stage 0: 0_atomic_skeletons
  数据: stage0 generator 按 11 个骨架(S1–S11)加权混合,distance bucket 控远近
        S1=1.0 简单写读, S5=1.0 stale-read, S7=1.0 多写+部分覆写+走神, S10=0.8 反向删除, ...
  训练: max_turns=12, max_steps=6000, freeze_lora=false, plugin_lr_mult=1.0
        多指标早停(stage0_val_overall ≥ 0.90, S1 ≥ 0.95, S5/S7/S10 ≥ 0.85, S11 ≥ 0.80)
  目标: Hippocampus 的 k_proj/v_proj/q_projs/o_projs/beta_proj 学会 Delta Rule 读写,
        LoRA 学会和 Hippocampus 协同(不抄写到 backbone 自己记)

    ↓ 骨架级读写已通

Stage 1: 1_5beat_natural
  数据: stage1 generator (DeepSeek 采样) 5-beat 自然长对话,10–14 turn,1A 子流
  训练: max_turns=16, max_steps=10000, plugin_lr_mult=0.3
        早停: stage1_val_overall ≥ 0.88, substream_1A ≥ 0.90, tier_hard ≥ 0.92
  目标: 跨 turn retention,自然语言中的 reveal/recall/refusal/overwrite 综合达标

    ↓ 自然分布已学会

Stage 2: 2_joint_consolidation
  数据: stage0+stage1 混采(15% : 85%) 联合巩固
  训练: max_turns=16, max_steps=4000, plugin_lr_mult=0.5
        早停: stage0_val_overall ≥ 0.85 AND stage1_val_overall ≥ 0.85
  目标: 防止 stage 1 训完后 stage 0 骨架能力退化,全面巩固
```

---

## Warmup baseline (debug 用)

`configs/persona_unified_v8_0.8b_warmup.yaml` + `curriculum_v8_warmup_only.yaml` 是快速 debug baseline:

```
Stage: 0_warmup_name
  数据: skeleton_weights={S_simple: 1.0} + force_relation=self_name
        2-turn 写读,姓+名合成 10000 条
  训练: max_turns=2, max_steps=1000, batch=32
  目标: 1000 步内 chat 5/5 (验证写读通路 + 训练后端正确性)
```

Warmup 通过 = 整条通路 OK。失败说明:LoRA 配置/写 kernel/数据格式/loss mask 之一坏了。

---

## Stage 0 早停的 11 个骨架(S1–S11)

stage0 generator 把单 episode 在 11 种"原子模式"中混采:

| 骨架 | 模式 | 早停权重 |
|---|---|---|
| S1  | 单 fact 写 → 单读 | 高(0.95) |
| S2  | 双写 + 双读 | 中 |
| S3  | 写 → 走神 → 读 | 中 |
| S4  | 多写 + 跳读 | 中 |
| S5  | 写 → 覆写 → stale-query(陈述/疑问两种句式)| 高(0.85) — 合并原 S5+S6 |
| S7  | 多写 + 部分覆写 + 走神 + 全读(实战最像)| 高(0.85) |
| S8  | 嵌套补充 | 中低 |
| S9  | 走神 dominated | 中 |
| S10 | 反向删除(写 → 删除 → 应拒答) | 高(0.85) |
| S11 | 多写部分删除(最难)| 高(0.80) |
| S_simple | warmup 专用 2-turn 写读(默认 weight=0,显式拉满才采) | — |

骨架定义在 `xinhe/data/skeletons/library.py`。早停按 per-skeleton VALUE 准确率(`stage0_val_S{i}`)分别设阈,任一未过都不切下一阶段。

---

## 失败诊断路径

### Stage 0 失败(stage0_val_overall < 0.90)

- 症状:多骨架混训卡在某 plateau
- 检查:
  - **某个骨架显著低**(比如 S10/S11 < 0.5):那个骨架的"删除/部分覆写"没学会 → 加 num_train / 提高那骨架的 weight
  - **read_scale 一直 < 0.05**:模型没真用 W → 检查 plugin_lr_mult / k_proj.weight.grad
  - **VALUE 高但 chat 失败**:训练分布与 chat 分布不匹配 → 跑 warmup baseline 对照

### Stage 1 失败(stage1_val_overall < 0.88)

- 症状:0 通过了但 1 卡 substream_1A 或 tier_hard
- 根因:LoRA 放开后把 stage 0 学到的 Delta Rule 用法"漂"掉了
- 检查:
  - 切回 stage 0 val 跑一次 evaluate,看 stage0 能力有没有退化
  - `read_scale` 是否塌陷(LoRA 抄了近路,模型在 backbone 里直接记)
  - 把 plugin_lr_mult 从 0.3 调到 0.5,给 plugin 更多保留信号

### Stage 2 失败

- 症状:joint val 某一边低
- 处理:调 sources ratio (默认 0.15 stage0 / 0.85 stage1),低的一边比例上调

---

## 关键调参经验

### 训练后端必须用 torch

`xinhe/model/hippocampus.py:write_from_content` 已经在 `model.train()` 时强制 backend="torch"。**不要绕过这条**。FLA Triton backward 在 bf16 上对 grad 累加误差 5–25%(read_scale 甚至 25%+),长序列下优化器收敛到读不出 W 的劣解。

推理时 `model.eval()` 自动用 auto → FLA(Linux+CUDA),forward 误差 < 0.5%,可以放心用以加速。

### plugin_lr_multiplier 的取值

| 阶段 | 取值 | 理由 |
|---|---|---|
| Stage 0 | 1.0 | plugin 需要大 LR 快速学会 Delta Rule 读写 |
| Stage 1 | 0.3 | plugin 已会读写,大 LR 会震荡;LoRA 从零学需要保 1.0 主 LR |
| Stage 2 | 0.5 | 联合巩固,plugin 需要轻微调整 |
| `--resume` 继续训 | 0.1 | plugin 保护,LoRA 继续微调 |

### VALUE 爬升不是线性的

观察曲线常见 plateau:
```
step 1000: 68   (plugin 摸到 Delta Rule)
step 3000: 78   (稳定爬)
step 5000: 85   (plateau)
step 7000: 85   (还卡)
step 8000: 96   (突破 plugin 训练死点)
```

中段 plateau 不是训够了,是 plugin 和 LoRA 在协调写策略。**别过早停**。

### 严阈值让模型继续训出细节

```
宽阈值(70/85/85): step 1000 早停, chat retention 71%
严阈值(95/85/92): step 3250 早停, chat retention 85%
```

严阈值额外几千步换来真实可用质量,值得。

---

## 入口命令

| 目标 | 命令 |
|---|---|
| 0.8b 端到端 | `uv run python scripts/train.py --config configs/persona_unified_v8_0.8b.yaml` |
| 4b 端到端 | `uv run python scripts/train.py --config configs/persona_unified_v8_4b.yaml` |
| Warmup baseline (debug) | `uv run python scripts/train.py --config configs/persona_unified_v8_0.8b_warmup.yaml` |
| 只跑 stage 0 | 同上 + `--from-stage 0_atomic_skeletons` |
| 从 stage 1 开始 | 同上 + `--from-stage 1_5beat_natural` |
| 从 stage 2 开始 | 同上 + `--from-stage 2_joint_consolidation` |

`train.py` 的 auto-skip 逻辑会自动跳过 `checkpoints/curriculum/{stage_name}.pt` 已存在的 stage。要重跑,删掉 ckpt 或用 `--from-stage`。

数据生成走 cache check:`data/v8/{stage0,stage1,stage2,warmup}/` 下 train.jsonl + val.jsonl 存在且条数够,就跳过生成。`--force` 强制重生成,或直接 rm。

---

## Phase 2 预告:Neocortex (夜晚模式)

当前 Hippocampus 是**短期工作记忆**(对话级,episode 末 reset)。Phase 2 将实现 **Neocortex** —— 通过 Sleep 机制把当日 W 中存活的记忆蒸馏到 per-layer 全秩并联 Memory MLP:

- **白天**(当前):Hippocampus + Attention LoRA 运转,Memory MLP 冻结观摩
- **夜晚**(未来):冻结 Hippocampus,Teacher 重建白天 W 轨迹,Student 强制 W 读为 0,靠 Memory MLP 自己复现完整通路 → KL + 隐藏态 MSE 蒸馏
- **翌日重启**:W 清空,保留永久烙印的 Memory MLP 权重

详见 `docs/心核  架构蓝图：大一统快慢交替记忆网络.md` 的"夜晚模式"章节。
