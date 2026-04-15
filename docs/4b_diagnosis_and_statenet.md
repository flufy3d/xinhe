# 4B 训练瓶颈诊断与解决方案

## 背景

0.8B 基础记忆课程（14 阶段）训练成功，ema_acc 稳定 95%+。
目标：升级到 4B backbone 获得更强的语言能力，同时保留 state plugin 的记忆能力。

4B 上尝试了多条路径，全部卡在 ~87% 准确率。本文档记录完整的排查过程、根因分析、方案演进，以及最终的简化方案。

---

## 走过的路（全部失败）

### 路径 1: 4B + 14 阶段课程从零训练

直接在 4B 上跑完整课程。前几阶段（单 entity、简单 recall）可以到 95%，但到 entity 区分（stage 9a/9b）准确率怎么也上不去，卡在 ~87%。

### 路径 2: 4B + state_dim=2560 原生对接（无 proj）

将 state_dim 设为 2560 匹配 hidden_size，去掉 proj_up/proj_down，state token 直接在 backbone 的表示空间里。结果：仍然卡住。

**排除了 proj 瓶颈假说。**

### 路径 3: 迁移方案（冻结 0.8B core）

先在 0.8B 训好 core，冻结后迁移到 4B，只训 proj + LoRA。详见 `4b_training_issues.md`。

结果：天花板同样是 87-88%。尝试了单阶段、两阶段、core 微调，均无法突破。

**排除了训练策略问题。**

### 路径 4: 增大 LoRA rank

怀疑 LoRA 容量不足，增大 rank。结果：反而更难训练，效果更差。

**排除了 LoRA 容量不足假说。**

---

## 根因分析

### 关键观察

| 实验条件 | 单 entity 准确率 | entity 区分准确率 |
|---------|-----------------|-----------------|
| 0.8B + 课程 | 95%+ | 95%+ |
| 4B + 课程 | **95%+** | **~87%** |
| 4B + 无 proj | 同上 | ~87% |
| 4B + 迁移 | — | ~87% |
| 4B + 大 LoRA | 同上 | 更差 |

**4B 能学会简单 state 读写（95%），只在 entity 区分上失败。**

这说明：
1. state plugin 的基本设计没有问题
2. 4B backbone 能通过 LoRA 学会使用 state token
3. 瓶颈精确定位在：**LoRA 无法让 state token 之间产生足够的分化，导致 entity 路由失败**

### 为什么 0.8B 能到 99% 而 4B 卡在 87%

LoRA 在 0.8B 和 4B 上面临同样的"全局共享"限制，但结果不同：

- **0.8B 小模型没什么"主见"**：注意力模式简单、弱，LoRA rank=4 就能大幅改变行为。"保护语言能力"的压力小（本来就不强），LoRA 可以大胆改 → 一个 LoRA 兼顾 state 路由 + 语言适配，够用。
- **4B 大模型有强烈的预训练模式**：注意力模式复杂、顽固。LoRA 必须小心翼翼 —— 改多了语言能力崩，改少了 state 路由学不会 → 走钢丝，rank 大了更难平衡。

类比：教小孩（0.8B）新技能容易，没有固化的习惯。教专家（4B）新技能难，强烈的既有模式和新行为互相干扰。

### 为什么 LoRA 做不到 entity 路由

entity 区分需要：
- "我叫张三" → 写入 state slot A（不是 B）
- "他叫李四" → 写入 state slot B（不是 A）
- "他叫什么" → 精确从 slot B 读取

这要求 state token 在 K/V 投影后产生**不同的特征**（slot A 的 key 带"我"特征，slot B 的 key 带"他"特征），以便 content 的 query 能区分它们。

但 LoRA 是**全局低秩扰动**：同一组 A、B 矩阵作用于所有位置（content + state），无法产生 position-specific 的行为。

同一个 LoRA 必须同时满足两个互相冲突的目标：
- **目标 A**：不破坏 content 的语言能力
- **目标 B**：让 state token 之间产生分化

rank 越大，冲突的自由度越多，优化 landscape 越复杂，训练越难收敛。这就是增大 LoRA rank 反而更差的原因。

### 附加因素：Qwen 3.5 的 linear attention

Qwen 3.5 4B 有 32 层：8 层 standard attention + 24 层 linear attention。

linear attention 无法做精确的 token-to-token 注意力。state token 的 entity 路由只能在 8 层 standard attention 里发生。单 entity recall 对精度要求低，8 层够用；entity 区分需要更高精度，8 层可能不够。

这是一个额外的制约因素，但在验证 StateNet 方案之前无法确定其影响程度。

---

## 方案演进

### 方案 A（初版）：LoRA 层内加 StateNet

最初的想法是在每个 LoRA 层内为 state token 位置加一个独立的全参 MLP 适配器：

```
Content 位置:  W(x) + LoRA(x)                ← 只管语言
State 位置:    W(x) + LoRA(x) + StateNet(x)  ← LoRA管语言 + StateNet管路由
```

**优点**：直接在 attention 投影层解决 K/V 分化问题，精准命中瓶颈。

**缺点**：
- StateNet 在 hidden_size 空间（0.8B=1024, 4B=2560），**无法跨 backbone 迁移**
- 每层 LoRA 都要加 StateNet + 传递 state_mask，改动侵入 backbone 内部
- 4B 训练 entity 阶段很贵，无法先在 0.8B 上验证再迁移

### 方案 B（改进）：State Plugin 内加 StateEnricher

将 slot 间交互放到 state plugin 内部，在 state_dim 空间（backbone 无关）：

```python
class StateEnricher(nn.Module):
    """state tokens 之间的自注意力，产生 entity 分化"""
    def __init__(self, state_dim, n_heads=4):
        self.self_attn = nn.MultiheadAttention(state_dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)

    def forward(self, state):
        # state: (B, 32, state_dim)  各 slot 互相看，形成分化表示
        state = state + self.self_attn(self.norm1(state), state, state)[0]
        state = state + self.mlp(self.norm2(state))
        return state
```

**优点**：state_dim 空间 = backbone 无关 → 可先 0.8B 训好再迁移到 4B。

**缺点**：新增独立模块，增加了架构复杂度。

### 方案 C（最终）：奥卡姆剃刀 —— 先减后加

审视现有 state plugin 的组件：

```python
# 现有 plugin 的所有处理层：
state_emb          # 初始状态
read_pos / write_pos / write_emb  # 位置编码
state_out_proj     # 输出投影（per-token）
gate_bias          # 静态门控偏置（per-token, per-dim）  ← 需要重新审视
gate_proj          # 动态门控投影（per-token）            ← 需要重新审视
```

**关键发现：现有层全部是 per-token 操作，slot 之间完全看不见彼此。**

gate_proj 处理 `[state_old_i, state_new_i]` 时，slot #5 完全不知道 slot #10 存了什么。slot 之间唯一的交互发生在 backbone 内部（frozen + LoRA），4B 上做不好 → 这就是 entity 路由失败的架构根源。

同时，gate_bias 的设计初衷需要重新审视：

```
gate_bias 的设计初衷:
  高 gate_bias → 慢维度（长期记忆）
  低 gate_bias → 快维度（工作记忆）
  期望：训练中自然涌现长短期分化

实际情况:
  1. 所有课程都是 16 轮以内的短 episode
     → 没有"记住 A 100 轮同时不断更新 B"的训练信号
     → gate_bias 没有分化的梯度压力，大概率学了近似均匀值
  2. 后续计划 sleep 固化机制（MLP 绑定长期记忆）
     → 长期记忆的职责交给 MLP，state 只需当好工作记忆
     → gate_bias 的长短期分化设计变得多余
```

**最终方案：减掉 gate_bias，在 gate 中加入 slot 间自注意力。**

```python
# 改动前（per-token，slot 间无交互）:
state_new = state_out_proj(write_raw)
gate = sigmoid(gate_bias + gate_proj(cat([state_old, state_new])))
state_next = gate * state_old + (1 - gate) * state_new

# 改动后（slot 间有交互，gate_bias 移除）:
state_new = state_out_proj(write_raw)
state_new = slot_attn(state_new)   # 自注意力：slot 之间互相看，形成分化
gate = sigmoid(gate_proj(cat([state_old, state_new])))
state_next = gate * state_old + (1 - gate) * state_new
```

slot_attn 是一层轻量自注意力（32 个 token 之间），让每个 slot 知道其他 slot 存了什么，从而发展出互补的存储策略。

### 方案 C 的优势

**先减后加，净复杂度可能更低：**

| 组件 | 改动前 | 改动后 |
|------|-------|-------|
| gate_bias | 有（无实际作用） | 移除 |
| gate_proj | per-token | 保留，角色更清晰："此 slot 此刻要不要更新" |
| slot_attn | 无 | 新增：slot 间自注意力 |

**每个组件职责清晰：**
- slot_attn: slot 间通信 → entity 分化 → 解决路由问题
- gate_proj: 动态更新决策（纯内容驱动，不再混入无用的静态偏置）
- state_out_proj: 输出对齐

**在 state_dim 空间操作 → 可迁移：**
- 0.8B 上训好 core（含 slot_attn）→ 直接迁移到 4B
- 迁移时 LoRA 只需学语言适配（不用学 entity 路由）→ 更容易收敛

**附带收益：解放 LoRA**
- LoRA 不再需要同时兼顾语言适配和 state 路由
- 后续 think 课程中，LoRA 专注于推理能力，state 路由由 slot_attn 负责
- 两者优化目标不再冲突

---

## 验证计划

### 最小验证

修改 state_plugin.py（移除 gate_bias，加入 slot_attn），跑到 stage 9a（entity 区分）：
- 如果准确率突破 87% 到 92%+ → 方向正确，继续跑完整课程
- 如果仍然卡住 → 瓶颈在 linear attention 层或其他因素，需要换思路

### 迁移验证

如果 0.8B + slot_attn 跑到 99%：
1. 冻结 core（含 slot_attn），迁移到 4B
2. 只训 proj + LoRA
3. 验证 entity 区分能否突破 87%（LoRA 不再需要学路由，理论上应该更容易）

### 备选方向（如果 slot_attn 不够）

- 换全 standard attention 的 backbone（LLaMA 系列），验证 linear attention 是否是制约因素
- 方案 A（LoRA 层内 StateNet），更直接但无法迁移
- Cross-attention 侧通道方案（代价：失去"带着记忆思考"的深度集成）

---

## 被排除的假说

| 假说 | 排除依据 |
|------|---------|
| proj 瓶颈（state_dim ≠ hidden_size） | state_dim=2560 无 proj 仍然失败 |
| 课程缺失 | 4B 跑了完整 14 阶段课程 |
| LoRA 容量不足 | 增大 rank 反而更差 |
| 训练策略（单阶段/两阶段/微调） | 多种策略同一天花板 |
| 整体架构设计问题 | 4B 前几阶段能到 95%，state 读写机制本身没问题 |
| 迁移 core 不兼容 | 从零训练也卡在同一位置 |

---

## 需要重新审视的设计决策

| 设计 | 原始意图 | 实际情况 | 建议 |
|------|---------|---------|------|
| gate_bias（静态门控偏置） | 涌现长短期记忆分化 | 课程无长短期训练信号，后续 sleep/MLP 接管长期记忆 | 移除 |
| per-token gate（slot 间无交互） | 简洁设计 | entity 路由需要 slot 间通信，是 4B 瓶颈的架构根源 | 加入 slot_attn |

---

## 改动范围

核心改动集中在 `xinhe/model/state_plugin.py`：
- 移除 `gate_bias` 参数
- 在 `extract_and_update` 中 `state_out_proj` 之后、gate 之前加入 slot 间自注意力
- slot_attn 归入 CORE_PARAM_PREFIXES（backbone 无关，可迁移）

不需要改动：
- `lora.py`（LoRA 层不变）
- `qwen_backbone.py`（backbone forward 不变）
- `xinhe_model.py`（model forward 流程不变）
- `curriculum.yaml`（课程不变）
