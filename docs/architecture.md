# Xinhe (心核) — 架构设计

## 核心假设

智能可以从"持续状态 + 可塑性 + 多时间尺度"这种统一动力系统中自然涌现。

不做 RAG、不做模块拼装、不扩大 context window。用最小结构（冻结 transformer + 持久状态矩阵 + LoRA），通过持续训练让系统内部自发分化出记忆、抽象、快慢变量。

---

## 整体架构

```
┌──────────────────────────────────────────────────┐
│                   XinheModel                      │
│                                                   │
│   输入 content (B,T,D)                             │
│        ↓ embed                                    │
│   ┌────────────────────────────────────────┐      │
│   │ for each hook layer:                   │      │
│   │   hidden → q_proj[L] → q (B,H,T,d_k)   │      │
│   │   q = F.normalize(q, dim=-1)           │      │
│   │   read = einsum(q, W^T) → (B,H,T,d_v)  │      │
│   │   out = o_proj[L](merge_heads(read))   │      │
│   │   hidden += sigmoid(read_scale) * out  │ ← 读  │
│   └────────────────────────────────────────┘      │
│        ↓ backbone 各层（frozen + LoRA）            │
│   content_output (B,T,D)                          │
│        ↓ write_from_content                       │
│   ┌────────────────────────────────────────┐      │
│   │ k = normalize(k_proj(content))         │      │
│   │ v = v_proj(content)                    │      │
│   │ β = sigmoid(beta_proj(content))        │      │
│   │ W_new = W + β·(v - W·k) ⊗ k^T  (Delta) │ ← 写  │
│   └────────────────────────────────────────┘      │
│        ↓                                           │
│   logits (via lm_head) + W_new                    │
└──────────────────────────────────────────────────┘
```

### 核心组件

**StateInterface**（`xinhe/model/state_plugin.py`）—— 唯一的可训练状态机制。

```python
class StateInterface(nn.Module):
    # 读侧：每层独立 q/o 投影（per-layer hook 语义）
    self.q_projs = nn.ModuleList([Linear(hidden, H*d_k)])    # n_layers
    self.o_projs = nn.ModuleList([Linear(H*d_v, hidden)])    # n_layers
    self.read_scale = Parameter(tensor(-3.0))                # sigmoid(-3)≈0.047

    # 写侧：全局共享（作用于 content_output 全序列）
    self.k_proj = Linear(hidden, H*d_k)
    self.v_proj = Linear(hidden, H*d_v)
    self.beta_proj = Linear(hidden, H)  # per-head 写强度
```

**状态形状**：`W: (B, n_heads=16, d_v=128, d_k=128)`
- 每样本 262144 floats ≈ 1MB（bf16）
- 多头独立的外积矩阵 —— 同一 (k,v) 由 k_proj 分配到不同 head
- 每个 head 是一个独立的联想记忆空间

---

## Delta Rule 读写机制

### 读：纯线性，零 softmax

```python
def read_layer(hidden_states, W, layer_idx):
    q = self.q_projs[layer_idx](hidden_states)       # (B,T,H*d_k)
    q = q.view(B,T,H,d_k).transpose(1,2)             # (B,H,T,d_k)
    q = F.normalize(q, dim=-1)                       # L2 归一化，匹配写侧
    read = einsum("bhtd,bhvd->bhtv", q, W)           # (B,H,T,d_v)
    merged = read.transpose(1,2).reshape(B,T,H*d_v)
    out = self.o_projs[layer_idx](merged)            # (B,T,D)
    return hidden_states + sigmoid(read_scale) * out
```

**为什么纯线性**：
- 无 softmax 意味着所有 head 同时贡献读出，不会"赢者通吃"
- `q @ W^T` 本质上是 key 空间的相似度加权 value 求和 —— 和 softmax attention 信息等价，但数学更简洁
- Identity-preserving：`W=0` 时 `read=0`，`read_scale` 初始 -3 → 初始读贡献≈0.05 × 0 = 0，backbone 行为完全保留

### 写：Delta Rule（外积更新）

```python
def write_from_content(W_old, content):
    k = normalize(self.k_proj(content))              # (B,H,T,d_k)
    v = self.v_proj(content)                         # (B,H,T,d_v)
    β = sigmoid(self.beta_proj(content))             # (B,H,T) per-head per-token 写强度
    
    # 对序列每个 token 依次更新（chunkwise parallel 实现）
    for t in range(T):
        W_new = W + β[t] * (v[t] - W·k[t]) ⊗ k[t]^T
    return W_new
```

**数学意义**：`W_new·k = W·k + β·(v - W·k) = (1-β)·W·k + β·v`
- 如果 k 是"新"的 key（W·k ≈ 0）→ 写入 β·v
- 如果 k 与已存在的 key 相似（W·k ≈ v_old）→ 用 β 权重从 v_old 过渡到 v_new（**天然覆写**）
- 如果 k 是随机噪声 → 误差 (v - W·k) 的方向与已有 key 正交，对已有 memory 干扰小

**并行实现**（`_delta_parallel`）：通过三角求解代替 Python 级 for 循环，把 GPU 利用率从 20% 提到 80%。详细推导见代码注释。

### Per-token β 的门控特性

`beta_proj(content)` 学习"这个 token 应该写多强"，sigmoid 输出 (0,1)：
- β ≈ 0：跳过写（chat 闲聊 token 不污染 W）
- β ≈ 1：强写入（fact value token）
- 训练良好的 plugin 在 fact token 上 β 达 0.3-0.5，在 chat token 上 β ≈ 0.03

这是**自学的写门控**，不需要额外机制。T2 诊断脚本 `scripts/probe_beta.py` 可量化此行为。

---

## 零侵入 backbone

```python
# QwenBackbone.forward_blocks 的唯一改动：加 layer_hook 调用
for layer_idx, layer in enumerate(layers):
    if layer_hook is not None:
        hidden_states = layer_hook(hidden_states, layer_idx)   # state read
    hidden_states = layer(hidden_states, attention_mask, ...)  # backbone 层不变
```

layer_hook 在每个 full-attention 层之前执行读。DeltaNet（线性 attention）层跳过 hook —— 那些层用 conv_state + recurrent_state 不适合额外 state 注入。

**为什么不用 past_key_values**：Qwen3.5 是混合架构（DeltaNet + full attention 交替），DeltaNet 不支持 KV cache 注入。layer_hook 统一生效。

---

## LoRA 的职责

v2 之前 LoRA 负责 state 路由 + 语言适配，在 4B 上低秩瓶颈暴露。v5c 把 state 路由**完全移出 LoRA**，交给全参数的 `q/k/v/o_proj`。LoRA 回归单一职责：**语言适配**。

```yaml
# 默认配置
lora:
  rank: 8
  alpha: 16
  target_modules: ["q_proj", "v_proj", "in_proj_qkv", "out_proj"]  # 只在 backbone attention
```

LoRA 参数量随 hidden 扩张：
- 0.8b (hidden=1024, 24 层): ~1.6M 参数
- 4b (hidden=2560, 36 层): ~5.9M 参数

---

## .pt = AI 的灵魂

```python
checkpoint = {
    "global_step": ...,
    "plugin_state": ...,           # StateInterface（W 读写投影 + read_scale）
    "lora_state": ...,             # LoRA A/B 矩阵
    "optimizer_state": ...,
    "scheduler_state": ...,
    "config": ...,
    "curriculum_stage": ...,
}
```

每个用户的 AI 从同一预训练起点出发，经历不同对话后 `.pt` 分化。不同的 `.pt` 就是不同的"人"。

---

## 安全启动

三层保护，加入 StateInterface 不破坏 backbone：

1. **LoRA 零初始化**：LoRA B 矩阵全零 → LoRA delta = 0，backbone 行为 = 原始
2. **W 零初始化**：`blank_state` 返回全零 W → `q @ W^T = 0` → 读贡献 = 0
3. **渐进 read_scale**：`sigmoid(-3.0) ≈ 0.047`，即使 W 非零，读的幅度也很小

这三重保护让训练早期 backbone 行为几乎 = 原始 backbone，模型有充分时间学习 state 机制而不破坏语言能力。

---

## Attention Mask 与 Position IDs

### Attention Mask

Backbone 处理纯 content（无 state token），使用标准因果 mask：

```
position 0..T-1: content
mask: 标准因果（位置 i 只看 0..i）+ padding 遮蔽
```

State 读取通过 layer_hook 完成，不需要额外 mask —— `q @ W^T` 是对 W 的线性读，没有位置约束。

### Position IDs

Content 使用 `0..T-1`。State（W 矩阵）不在序列中，没有 RoPE —— 合理的，state 是"外部关联记忆"，不应有位置偏好。

---

## Burn-in：用 prompt 初始化 persona

```python
@torch.no_grad()
def burn_in(self, token_ids_list, batch_size=1):
    state = self.init_state(batch_size)  # 全零 W
    for token_ids in token_ids_list:
        result = self.forward(token_ids, state)
        state = result["state_next"]
    return state
```

不同 system prompt → 写入 W 不同的 (k,v) 对 → 不同初始状态 → 不同 persona。老用户回来从 `.pt` 恢复。

---

## 训练机制

### 课程结构

新的 `persona_unified` 课程只有 2 stage（详见 [curriculum_learning.md](curriculum_learning.md)）：

```
Stage 0: 0_bootstrap (2 轮, freeze_lora, 1 fact/name only)
  → plugin 学会 Delta Rule 基本读写

Stage 1: 1_persona_unified (12-20 轮, 10 种 turn kind + retention patterns)
  → 完整能力（世界知识保留 + 拒答 + 多 fact + 覆写 + retention）
```

### TBPTT

```python
state = init_state()
for seg_idx, segment in enumerate(episode):
    result = forward(segment, state, labels, weights)
    loss += result["loss"]
    state = result["state_next"]
    if (seg_idx + 1) % tbptt_steps == 0:
        loss.backward()
        optimizer.step()
        state = state.detach()   # 截断梯度流
        loss = 0
```

典型 tbptt_steps=8，episode_length=16 → 每 episode 做 2 次 backward。

### 4 指标联合早停

训练期 `_validate` 计算：
- VALUE (`eval_value_breakdown`)：召回 value token argmax 准确率
- WorldQA (`eval_world_qa`)：世界 QA 整体 token 准确率
- Refusal (`eval_refusal`)：拒答 regex 检测率（且无 fabrication）
- Compositional (`eval_compositional`)：多 fact 单句全部 value 命中率

**4 个全部过阈值**才触发早停。典型阈值 98 / 85 / 95 / 95。

### 值加权 loss

```python
# conversation.py: tokenize_turn
weights = [1.0 if label != -100 else 0.0 for label in labels]  # assistant = 1
if value_str:
    for v in values:  # 支持 str | list[str]
        # 用 offset_mapping 定位 v 在 assistant text 中的 token 区间
        weights[value_tokens] = VALUE_WEIGHT  # 5.0
```

Value token 得到 5× 梯度，让模型优先学对具体事实 token 而非 filler。

---

## 参数总览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_heads | 16 | 多头独立的联想记忆 |
| head_dim | 128 | d_v = d_k = 128 |
| W 形状 | (B, 16, 128, 128) | 每样本 262144 floats |
| read_scale_init | -3.0 | sigmoid(-3)≈0.047，初始读贡献小 |
| beta_bias_init | -2.0 | sigmoid(-2)≈0.12，初始写强度 |
| LoRA rank | 8 | 注入 q/v/in_proj/out_proj |
| lora_alpha | 16 | |
| segment_length | 256 | 单 segment token 长度 |
| episode_length | 16 | 单 episode segment 数 |
| tbptt_steps | 8 | 梯度截断窗口 |

---

## 记忆体系分层：短期（W） vs 长期（Memory MLP + Sleep，未来）

心核的记忆是**双层架构**，对应人脑的海马体 vs 皮层分工：

| 层 | 当前状态 | 载体 | 类比 | 更新时机 |
|---|---|---|---|---|
| **短期 / 工作记忆** | ✅ 已实现 | Delta Rule W 矩阵 | 海马体 / 工作记忆 | 对话每轮（推理时动态） |
| **长期 / 固化记忆** | 🚧 未来工作 | Memory MLP（规划中） | 皮层 / synaptic consolidation | Sleep 时批量固化 |

### 当前（v5c）：只有短期记忆

W: `(B, H=16, d_v=128, d_k=128)` 每样本 1MB，Delta Rule 动态演化。能处理单次对话内的记忆（fact 存储、召回、覆写、跨 chat retention）。

**当前的局限**：
- W 容量固定，长对话必然遗忘早期信息
- 对话结束后（`.pt` 保存）W 里的信息是静态的 —— 没被主动"固化"到权重
- 跨 session 的记忆靠 burn_in 重建，不是真正"消化了"

### 未来：长期记忆固化（Memory MLP + Sleep）

**设计方向**（未实现）：
- 引入 Memory MLP 模块（SwiGLU 结构，独立于 backbone）作为长期记忆载体
- Sleep 阶段：replay buffer 回放对话 + 逐步弱化 W（read_scale → 0），迫使信息从 W 转移到 Memory MLP 权重
- 效果：sleep 过的事实即使 W 清空也能回答（权重里记住了）；`.pt` 分化更深刻（MLP 权重随用户独立演化）

**为什么现在先不做**：
- 当前 v5c 的 W 在单次对话内已经充分解决了短期记忆问题
- Memory MLP + Sleep 增加工程复杂度（replay buffer / 两阶段训练 / 灾难遗忘防护）
- 先验证短期记忆机制，再加长期层。分步 de-risk

### 为什么不用 RAG

RAG 检索原文 chunk，每次回忆都要重新理解 —— 信息没被"消化"。心核的 W 存的是 (k,v) 外积对，信息已经消化进联想记忆结构。未来长期化后，Memory MLP 权重更是完全 implicit 的记忆表示。

**可能的融合点**：跨用户的共享知识（公共世界事实）用 RAG 补充，personal / episodic 记忆用 W + Memory MLP。但这是远期考虑。

---

## 设计演进简史

v1 → v5c 演化详见 [design_rationale.md](design_rationale.md)。一句话版本：

- v1 state-as-tokens + shared LoRA → 0.8b 通过，4b 卡 entity 87%
- v2 对称 cross-attention + 专用投影 → 解决 LoRA 瓶颈
- v3 EKS (slot keys) / v4 slot routing → slot 身份伪概念
- v5b slot + contrastive → slot 线性投影 hash 碰撞
- **v5c Delta Rule**：`(v - Wk)` 数学消歧，取消 slot，简洁性最大
- **persona_unified (最新)**：训练数据从窄分布（13-stage memory + think）改为真实使用分布（persona 驱动多轮 + teacher rehearsal + retention patterns）

v5c 是架构终点，persona_unified 是训练范式终点。
