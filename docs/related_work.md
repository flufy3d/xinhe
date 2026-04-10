# 心核 (Xinhe) — 相关工作综述

> 本文档整合了心核与现有研究的对比分析。
> 包含：EM-LLM 详细对比 + 更广泛的相关工作调研。

---

## 一句话定位

- **EM-LLM** = 外存 + 检索：把 KV cache 卸载到 CPU/磁盘，需要时用惊讶度分割 + 相似度检索找回来
- **RMT** = 循环记忆 token：输入前加 `[mem]` token，处理后传给下一个 segment
- **Titans** = 测试时权重学习：用小 MLP 在推理时每 token 梯度更新
- **心核** = 内存 + 演化 + 睡眠固化：state token 持续压缩记忆，sleep 时固化到 LoRA 权重，.pt = 灵魂

---

## 1. 最相似：Recurrent Memory Transformer (RMT)

> Bulatov et al., NeurIPS 2022 / AAAI 2024
> https://arxiv.org/abs/2207.06881

在 self-attention 里加 `[mem]` token，处理一个 segment 后，`[mem]` 的输出作为下一个 segment 的输入，形成循环。

| | RMT | 心核 |
|---|---|---|
| 记忆 token | 读写同一组 | 读写分离（read 在序列头，write 在尾） |
| 更新机制 | 直接覆写 | 双层 gate：σ(static_bias + dynamic_proj) |
| LoRA | 无，全量微调或从零训 | Skill LoRA (attention) + Memory LoRA (MLP) |
| Sleep | 无 | replay buffer + state 弱化 → 权重固化 |
| Per-user 分化 | 无 | .pt 灵魂分化 |

**评价**：RMT 是心核 StatePlugin 最近的架构祖先。心核的读写分离和双层 gate 是关键扩展。

---

## 2. Titans — 测试时记忆学习

> Behrouz & Zhong, Google, 2025
> https://arxiv.org/abs/2501.00663

引入一个小 MLP 作为长期记忆模块，推理时根据"惊讶度"（loss 高 = 重要 = 记住）实时梯度更新 MLP 权重。

| | Titans | 心核 |
|---|---|---|
| 记忆载体 | 独立 MLP，推理时每 token 更新权重 | state token（推理时）+ MLP LoRA（sleep 时） |
| 更新时机 | 持续，每 token 都做梯度 | 分离：推理时只更新 state，sleep 时更新权重 |
| 计算开销 | 每 token 一次反向传播，开销大 | 推理时零额外开销，sleep 时批量更新 |
| 惊讶度门控 | 有 | 无（gate 学习什么该记） |

**评价**：Titans 验证了"记忆应该在权重里"的核心假设。但每 token 做梯度更新对实时对话不实际。心核的 sleep 批量更新更适合部署。

---

## 3. GDWM — 门控可微工作记忆

> Gated Differentiable Working Memory, 2026
> https://arxiv.org/abs/2601.12906

用 write controller 估计"上下文效用"，门控决定哪些信息固化到 **LoRA adapter** 中。

| | GDWM | 心核 |
|---|---|---|
| LoRA 固化 | 有，按信息效用门控写入 | 有，sleep 时 replay + state 弱化 |
| 适用范围 | 单次长上下文内 | 跨 session，持续积累 |
| State token | 无 | 有，32×1024 |
| Sleep 周期 | 无，持续适应 | 有，批量巩固 |

**评价**：GDWM 是最接近心核 sleep 机制的已发表工作——都是往 LoRA 里固化信息。但 GDWM 没有跨 session 持久记忆。

---

## 4. MemoryLLM / M+

> Wang et al., ICML 2024 / ICML 2025
> https://arxiv.org/abs/2402.04624

冻结 Llama2-7B，每层加大量 memory token（~1B 参数）。推理时新信息覆写最旧的 memory slot（FIFO）。

| | MemoryLLM | 心核 |
|---|---|---|
| 记忆规模 | 巨大（~1B 参数，每层 7680 token） | 微小（32×1024 ≈ 32K 参数） |
| 更新策略 | FIFO 替换 | 学习的 gate 混合 |
| 权重学习 | 无 | Sleep 时 MLP LoRA 更新 |
| 冻结 backbone | 是 | 是 |

**评价**：MemoryLLM 用巨大记忆池补偿了没有权重级学习的缺陷。心核的哲学是小 state + 权重固化，更接近生物记忆。

---

## 5. Reactive Transformer (RxT)

> Filipek, 2025
> https://arxiv.org/abs/2510.03561

维护固定大小的 Short-Term Memory (STM) slot，用 GRU 风格的 gate 更新：`STM_t = (1-G) * STM_{t-1} + G * Update`。

| | RxT | 心核 |
|---|---|---|
| 记忆读写 | 跨注意力（cross-attention） | 自注意力（self-attention，token 注入） |
| Gate | GRU 风格 | σ(static_bias + dynamic_proj)，含静态偏置 |
| 额外网络 | 单独的 memory encoder | 无，复用 backbone |

**评价**：gate 哲学非常相似。心核更简单（复用 self-attention），RxT 引入了额外的 cross-attention 机制。

---

## 6. Infini-attention

> Munkhdalai et al., Google, 2024
> https://arxiv.org/abs/2404.07143

每层 attention 内加一个压缩记忆（线性 attention），用 gate 平衡本地注意力和长期记忆检索。114x 压缩率。

| | Infini-attention | 心核 |
|---|---|---|
| 目标 | 单次超长序列 | 跨 session 持久记忆 |
| 记忆形式 | per-layer KV 压缩 | 显式 state token |
| 在线学习 | 无 | Sleep 权重更新 |

**评价**：解决不同问题。Infini-attention 做单次长上下文，心核做跨会话持久记忆。

---

## 7. G-MemLLM

> Gated Latent Memory, 2026
> https://arxiv.org/abs/2602.00015

冻结 LLM + 可训练 Latent Memory Bank，GRU 风格 gate 更新。< 3% 额外参数。

**评价**：gate 机制与心核相似，但没有 sleep、LoRA、跨 session。

---

## 8. 其他相关工作

| 工作 | 年份 | 与心核的关系 |
|------|------|-------------|
| **DualNet** (NeurIPS 2022) | 2022 | 快/慢互补学习系统 + replay buffer，验证了心核 sleep 的理论基础 |
| **Sleep-like Replay** (Nature Comm.) | 2022 | 生物学验证：睡眠 replay 减少灾难遗忘，形成正交记忆表示 |
| **SleepGate** | 2026 | 学习的 sleep 周期 + 遗忘 gate，但操作 KV cache 而非权重 |
| **Larimar** (IBM, ICML 2024) | 2024 | 脑启发的快/慢记忆分离，但用外部 KV 存储而非权重学习 |
| **Compact Recurrent Transformer** (ICML 2025) | 2025 | 单个持久记忆向量 + RNN 压缩，比心核更紧凑但容量更小 |
| **Trained Persistent Memory for Frozen LLMs** | 2026 | 系统测试了 6 种冻结 LLM 加记忆的方法，prefix tuning ≈ 简化版心核 read token |

---

## 9. EM-LLM 详细对比

> EM-LLM: https://github.com/em-llm/EM-LLM-model (ICLR 2025)
> 论文: https://openreview.net/forum?id=BI2int5SAC

### 根本哲学

| | EM-LLM | 心核 |
|---|---|---|
| **核心思路** | 外挂检索系统，从历史 KV cache 中找回相关片段 | 内生状态演化，让模型学会自己"记住"什么 |
| **类比** | 人翻笔记本找信息 | 人脑把经验内化为直觉/记忆 |
| **记忆形态** | 显式存储的 KV 对（episodic memory blocks） | 32 个可学习 state token 的持续演化 |

### 训练 vs 推理

| | EM-LLM | 心核 |
|---|---|---|
| **是否需要训练** | 完全免训练（inference-time monkey-patch） | 需要训练（LoRA + StatePlugin，~500K 参数） |
| **修改权重** | 不动模型权重，运行时替换 attention forward | 冻结 backbone，只训练 state 机制和 LoRA |
| **在线学习** | 无 | 核心设计 — state 随对话持续演化 |
| **持续适应** | 无 — 模型永远是出厂状态 | sleep 时反向传播更新权重，对话经验固化进 .pt |

### 记忆机制

| | EM-LLM | 心核 |
|---|---|---|
| **存储方式** | KV block 卸载到 CPU/磁盘，按需检索回 GPU | state token 始终在 GPU 上，参与每次 forward |
| **记忆容量** | 理论无限（已测到 1000 万 token） | 受 state 维度限制（32×1024），是压缩瓶颈 |
| **分割策略** | 惊讶度阈值 + 图论社区检测 | 无显式分割 — gate 隐式决定保留/遗忘 |
| **检索策略** | 两阶段：相似度 top-k + 时间邻近 buffer | 无检索 — state 本身就是压缩后的记忆 |
| **遗忘机制** | 无主动遗忘，所有 block 永久存储 | dual-layer gate 持续决定更新/保留 |
| **记忆层级** | 单层（KV block 存储） | 双层 — state 短期 + sleep 权重长期 |
| **跨会话持久** | 无 — 会话结束 KV 清空 | .pt 保存全部状态和权重，重启后延续 |

### EM-LLM 核心流程

```
输入 token 流
    ↓
按 chunk (512 token) 处理
    ↓
计算每个 token 的惊讶度 (-log P(x_t))
    ↓
超过阈值的位置 → 事件边界候选
    ↓
图论 modularity 优化 → 精化边界
    ↓
KV 对按事件分块存储 → CPU/磁盘
    ↓
解码时: initial tokens (128) + 检索 tokens (2048) + local window (4096)
    ↓
两阶段检索: 代表向量相似度 top-k + 时间邻近 buffer
```

### 适用场景

| | EM-LLM | 心核 |
|---|---|---|
| **擅长** | 超长文档、精确事实检索 | 多轮对话、人格积累、个性化 AI |
| **模型规模** | 7B-8B | 0.8B 验证 → 4B/9B 迁移 |
| **上下文规模** | 百万级 token | state + 权重跨会话无限积累 |
| **信息保真度** | 高 — 原始 KV 完整保留 | 有损压缩，sleep 固化缓解 |
| **计算开销** | 内存随历史线性增长 | 固定开销（32 token 常驻） |
| **个性化** | 无 | 每个用户独立 .pt |

### 互补关系

```
EM-LLM  →  "图书馆"  →  海量信息的精确存取
心核     →  "大脑"    →  经验的内化与抽象
```

State-level RAG 是两者思路的融合点：不检索原始 token，而是检索已压缩的 state 快照。

---

## 10. 心核的独特性总结

**没有单一论文组合了心核的全部组件。**

| 心核组件 | 最近的先例 | 心核的扩展 |
|----------|-----------|-----------|
| Read/Write state token | RMT (2022) | 读写分离 + 双层 gate（static bias + dynamic proj） |
| LoRA 教 state 交互 | 标准 LoRA | LoRA 专门服务 state 读写，不是任务适配 |
| Sleep replay → MLP LoRA 固化 | GDWM, DualNet | 完整 sleep 周期（state 弱化 100%→0%），无直接先例 |
| .pt = 灵魂 | 无 | per-user 权重分化，checkpoint 即身份 |
| 双 LoRA（Skill + Memory） | 无 | Attention LoRA 教路由，MLP LoRA 存记忆，功能分离 |

### 心核独有的能力维度

| 能力 | 机制 | 意义 |
|------|------|------|
| **灵魂分化** | 同一起点 → 不同用户 sleep → 不同 .pt | AI 个性化是权重级别的分化，不是 prompt 工程 |
| **灵魂遗传** | 从成熟 .pt 初始化新 AI | 类似"师徒"关系，知识和风格可以传承 |
| **多 AI 社交** | 不同 .pt 之间交换 state | AI 之间的"交流"不需要自然语言，直接在状态空间对话 |
| **权重级记忆** | sleep 反向传播更新 LoRA | 反复出现的模式从 state 沉淀为权重，类似肌肉记忆 |
| **心跳主动闲聊** | 空输入 + state → 主动开口 | AI 不只被动回答，能基于记忆主动发起话题 |

---

## 参考文献

- [RMT] Bulatov et al., "Recurrent Memory Transformer", NeurIPS 2022. https://arxiv.org/abs/2207.06881
- [Titans] Behrouz & Zhong, "Titans: Learning to Memorize at Test Time", 2025. https://arxiv.org/abs/2501.00663
- [GDWM] "Gated Differentiable Working Memory", 2026. https://arxiv.org/abs/2601.12906
- [MemoryLLM] Wang et al., "MemoryLLM", ICML 2024. https://arxiv.org/abs/2402.04624
- [RxT] Filipek, "Reactive Transformer", 2025. https://arxiv.org/abs/2510.03561
- [Infini-attention] Munkhdalai et al., Google, 2024. https://arxiv.org/abs/2404.07143
- [G-MemLLM] "Gated Latent Memory", 2026. https://arxiv.org/abs/2602.00015
- [DualNet] Pham et al., "DualNet: Continual Learning, Fast and Slow", NeurIPS 2022. https://arxiv.org/abs/2110.00175
- [Sleep Replay] Tadros et al., Nature Communications, 2022. https://www.nature.com/articles/s41467-022-34938-7
- [SleepGate] Xie, 2026. https://arxiv.org/abs/2603.14517
- [Larimar] Das et al., "Larimar", ICML 2024. https://arxiv.org/abs/2403.11901
- [EM-LLM] ICLR 2025. https://openreview.net/forum?id=BI2int5SAC
