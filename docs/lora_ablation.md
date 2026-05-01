# LoRA 挂载位置消融:qkvo vs without_o vs full

`persona_unified_0.8b` (Qwen3.5-0.8B backbone, 6 标准 attention 层 + 18 线性 attention 层, Hippocampus 上挂)
三种 target_modules 挂载方案的对比。配置定义在 `configs/base.yaml`。

## 三种配置

| Run         | target_modules                                                 | 可训练参数            | 注意力侧               | 线性注意力侧                    |
| ----------- | -------------------------------------------------------------- | --------------------- | ---------------------- | ------------------------------- |
| `qkvo`      | q,k,v,o                                                        | 30,457,873  (3.9%)    | Q/K/V + **O**          | —                               |
| `without_o` | q,k,v + in_proj_a/b/z                                          | 31,646,737  (4.0%)    | Q/K/V                  | 写读三键 (a/b/z)                |
| `full`      | q,k,v,o + in_proj_a/b/z + out_proj                             | 32,826,385  (4.2%)    | Q/K/V + **O**          | 写读三键 + out_proj             |

注:`qkvo` 使用旧 curriculum (skeleton max=6000, dialog max=10000),`without_o`/`full` 使用现 curriculum (3500 / 4000)。同 step 直比有 LR 衰减偏置。

## Stage 0 — skeleton (单 fact 写读)

| step                | qkvo overall / VALUE | full overall / VALUE | 备注                |
| ------------------- | -------------------- | -------------------- | ------------------- |
| 1000                | 76.11 / 86.33        | 74.34 / 85.06        | qkvo 略好           |
| 2000                | 86.73 / 92.15        | 85.84 / 92.15        | 持平                |
| 3000                | **93.81 / 96.20**    | 85.84 / 92.66        | qkvo 显著领先 +8pp  |
| 3500 (full final)   | —                    | 85.84 / 93.16        | full 在低 lr 已停滞 |
| 6000 (qkvo final)   | 91.15 / 95.95        | —                    | 微回退,大体收敛    |

`without_o` skeleton 阶段从既有 ckpt 直接跳过,无原始数据。

**难子集 (S3 / S7 / S11)**:
- qkvo @3000: S7=85.71 / S11=83.33 / S3=86.67  (基本达标)
- full @3500: S7=71.43 / S11=83.33 / S3=73.33  (S7 / S3 仍弱)

## Stage 1 — dialog (5-Beat 多轮)

| step | qkvo (旧 curr) | without_o      | full           | 趋势                               |
| ---- | -------------- | -------------- | -------------- | ---------------------------------- |
| 1000 | 61.70 / 82.38  | 65.96 / 80.74  | 64.89 / 83.20  | 起点接近                           |
| 2000 | 68.09 / 84.43  | 60.64 / 80.33  | 64.89 / 85.25  | without_o 退步                     |
| 3000 | 69.15 / 85.25  | 59.57 / 79.92  | 65.96 / 84.43  | qkvo 上行 / without_o 续跌 / full 持平 |
| 4000 | (没跑到)        | 60.64 / 80.74  | 67.02 / 84.43  | full 微升,without_o 收尾仍 60%    |

**ema_acc 终值** (整体 token 熵):
- qkvo @3190: 43.24% (上行未停)
- without_o @4000: 44.24% (卡 44%)
- full @4000: 44.39% (卡 44%)

三家 ema_acc 都卡 44% 附近,但 dialog_val 趋势分化:
- **qkvo**: 单调上行 (+7.5pp from step1000→3000)
- **without_o**: **单调下行** (-6.4pp)
- **full**: 持平偏微升 (+2pp)

`dialog_val_substream_1B = 0%` 三家全程未启动 → 训练数据没 1B 配比 (`curriculum.yaml mix: {1A: 1.00}`),非 LoRA 问题。

## 机制解释

### 1. o_proj 是分水岭

`without_o` 是唯一不挂 o_proj 的配置,也是唯一 dialog_val **退步** 的配置。VALUE 80% 持平但 dialog_val 跌 6pp:

- W 通路 (受 in_proj_a/b/z LoRA 控制) **写得进、读得出** —— VALUE 持稳为证
- 但**标准注意力的输出端没法把读到的内容塑形成对话语境下合用的 logits**
- o_proj 是 attention 输出投影,**没有它的 LoRA,backbone 原本针对预训练分布的 o_proj 不能适配新对话分布**;训练越久越偏离,dialog_val_overall 反而下行

### 2. 线性注意力侧 LoRA (`in_proj_a/b/z` + `out_proj`) 边际收益小

把 `qkvo` 加上线性注意力 LoRA 得到 `full`:
- skeleton 上 **更差** (step3000 93.81 → 85.84)
- dialog 上接近 (step3000 69.15 → 65.96, curriculum 不可比)
- VALUE 略升 1-2pp

W 写读由 backbone 自带 in_proj 权重 + Hippocampus 可训 plugin (`read_scale`, `beta_bias`) 主导,LoRA 在 in_proj_a/b/z 上是 **冗余优化方向**,只稀释 LR 与更新预算。
对应到既有结论 *"feedback_read_scale_low_ok"* 与 *"feedback_memory_write_nontrainable"* —— 写是声明式快照不需要学,读 scale 已稳态;LoRA 在这条通路上没什么可学的。

### 3. 参数数量不是关键

- qkvo (30.46M) < without_o (31.65M) < full (32.83M)
- 效果排序: **qkvo > full > without_o**
- 多 1.2M 参挂错位置 (without_o 把额度从 o_proj 挪到 in_proj_a/b/z) 直接导致 dialog 退步
- **正确的"少而准"胜过"多而散"**

### 4. 44% ema_acc 平台与 LoRA 配置无关

三家 dialog 阶段 ema_acc 都卡 44%。这是 **数据 / curriculum 层面的瓶颈**:
- val log 显示 `turn_truncation_rate=10.24%` (max_turn_tokens_seen=1103 vs `turn_max_tokens=256`)
- 突破上限要调数据侧 (`turn_max_tokens` ≥1152, 或降 `beat3_min_chars`),不是改 LoRA

## 结论与建议

1. **生产配置选 `qkvo`**: 参数最少、skeleton 最快收敛、dialog 单调上行。Backbone 的线性注意力部分**不需要 LoRA 干预**。
2. **绝对避免 `without_o`**: 移除 o_proj LoRA 会让 dialog 训练越走越歪,VALUE 看似 OK 实际输出端在退化。
3. **`full` 是"加错地方"**: 多出的 in_proj LoRA 没帮 dialog,反而让 skeleton 难子集 (S3/S7) 退步。要扩 LoRA,应优先 MLP 层 (`gate_proj`/`up_proj`/`down_proj`),不是线性注意力的 in/out projection。
4. **44% 平台是数据瓶颈**: 与 LoRA 选择无关,要破上限改数据 (`turn_max_tokens`/`beat3_min_chars`)。

## 备份位置 (远端)

- `backup/lora_qkvo/`         — q,k,v,o,旧 curriculum,stage 0 完整 (`0_atomic_skeletons.pt`),stage 1 半路止
- `backup/lora_full_without_o/` — q,k,v + in_proj_a/b/z,stage 1 完整 (`1_5beat_natural.pt`),stage 0 沿用既有
- `backup/lora_full/`         — q,k,v,o + in_proj_a/b/z + out_proj,stage 0+1 完整,stage 3 未跑
- `backup/novel_lora_full/`   — `full` 配置下小说 recall 实验,`novel_only.pt`
