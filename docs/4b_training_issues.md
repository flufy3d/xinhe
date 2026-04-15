# 4B 基座迁移：实验记录与失败分析

## 背景

0.8B 基础记忆课程（14 阶段）已训练成功，ema_acc 稳定 95%+。目标：将 0.8B 训好的 plugin core（灵魂）迁移到 4B backbone。

核心假设：plugin core 是 backbone-agnostic 的，只需重新训 proj（维度桥接）和 LoRA（backbone 适配），core 冻结即可。

## 实验记录

### 实验 0: 4B 从零直接训练

**结论：失败**

直接在 4B 上训练 state plugin，准确率上不去。这是开发整套迁移机制的动机。

失败原因：state 初始随机 → backbone 发现用 state 有害 → gate 学会"永远不写" → 局部最优。这和 0.8B 不用课程学习直接训练的失败模式一致。

---

### 实验 1: 单阶段迁移 M0_adapt（冻结 core，proj + LoRA 同时训）

**配置**：
- freeze_plugin_core: true
- grad_accum_steps: 2 → 后改 8
- batch_size: 1
- 全能力数据（5 facts, entity, recall, overwrite）

**结果**：

| 指标 | 值 |
|------|-----|
| ema_acc 范围 | 84-88% |
| scale | 0.5653（完全不动，符合预期） |
| 训练步数 | ~1500 步 |

**观察**：
- ema_acc 快速到 87% 后进入平台期
- 增大 grad_accum 从 2→8 没有改善（有效 batch 从 2→8，每步更慢但收敛速度无变化）
- 单步 acc 在 63-100% 大幅波动（batch=1 噪声）

**结论**：proj 和 LoRA 同时训练，互相追逐（proj 的目标是 LoRA 修改后的 hidden states，LoRA 的目标是 proj 映射后的 state output），形成移动目标，收敛困难。

---

### 实验 2: 两阶段迁移 M0_proj_align → M1_joint_adapt

**设计思路**：先让 proj 单独对齐，再联合训练。

**M0_proj_align（500 步）**：
- freeze_plugin_core: true + freeze_lora: true（只训 proj）
- 最简数据：name only, 2 turns, distance=0
- 结果：ema_acc 64% → 82%，proj 找到正确方向

**M1_joint_adapt（5000 步）**：
- freeze_plugin_core: true + freeze_lora: false（proj + LoRA）
- 全能力数据
- 结果：

| 阶段 | ema_acc 范围 | 趋势 |
|------|-------------|------|
| step 0-300 | 67% → 82% | 快速上升 |
| step 300-600 | 82% → 88.7% | 缓慢上升 |
| step 600-5000 | 84-88% | 平台震荡 |

**改进**：收敛速度显著提升（300 步到 82% vs 实验 1 需要 1000+ 步）。
**问题**：天花板不变，仍然是 87-88%。

**结论**：两阶段解决了收敛速度问题，但没有解决上限问题。瓶颈不在训练顺序，而在冻结 core 本身。

---

### 实验 3: M2_core_finetune（core 0.1x lr 微调）

**设计思路**：在 M1 基础上，解冻 core 用极小 lr 适配 4B 的表示几何。

**配置**：
- freeze_plugin_core: false
- plugin_core_lr_multiplier: 0.1
- 实际 core lr = 3e-4 × 2.0 × 0.1 = 6e-5（峰值）

**结果（950 步）**：

| 指标 | M1（冻结 core） | M2（core 0.1x lr） |
|------|----------------|-------------------|
| ema_acc 范围 | 84-88% | 85-89% |
| scale 变化 | 0（冻结） | 0.0004（0.5653→0.5649） |

**结论**：0.1x lr 等于没解冻。core 的有效 lr 只有 6e-5，950 步 scale 才动了万分之四。需要更大 lr 才能让 core 真正适配。

---

## 核心发现

### 1. 冻结 core 的天花板是 87-88%

三个实验（单阶段 / 两阶段 / 微调 core）反复验证了这个上限。

原因分析：0.8B core 的 `gate_proj` 在 0.8B 的表示空间中训练，学会了"tell 位置的 hidden state 长什么样"。4B 的 hidden state 几何结构不同，即使通过 proj_up 线性映射，也无法完全还原 0.8B 的分布特征。proj 是线性变换，但两个 backbone 的表示差异可能是非线性的。

### 2. grad_accum 增大没有改善

grad_accum 从 2→8，理论上减少梯度噪声，实际 ema_acc 无变化。说明瓶颈不是梯度噪声，而是模型容量/表示限制。

### 3. 两阶段加速收敛但不改上限

先 proj → 再 proj+LoRA 的分阶段策略有效提升收敛速度（避免互相追逐），但上限不变。说明最终瓶颈和训练动力学无关，是冻结 core 的表示能力上限。

---

## 待尝试

### A. 提高 core lr 到 0.5x

让 core 真正适配 4B，代价是可能破坏已学好的 state 读写模式。
从 M1 checkpoint 出发，core 已有 0.8B 的知识，不太可能回到"放弃 state"。

### B. 4B 从零跑完整 14 阶段课程

0.8B 的成功是课程学习的功劳。4B 之前失败可能是因为没用课程。
如果 4B + 14 阶段课程能训到 95%+，则迁移方案不再必要。
代价：训练时间远长于迁移。

### C. 非线性 proj

当前 proj_up/proj_down 是单层线性。如果 0.8B→4B 的表示差异是非线性的，
加一层 MLP + activation 可能帮助。但增加了模型复杂度。
