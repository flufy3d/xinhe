# Xinhe 心核

**验证一个假设：智能是否可以从"持续状态 + 可塑性 + 多时间尺度"中自然涌现？**

不做 RAG、不做模块拼装、不扩大 context window。用最小结构 —— 冻结的小 transformer + **Delta Rule 联想记忆** + LoRA —— 让记忆、遗忘、个性从系统内部涌现。

每个心核实例从同一起点出发，经历不同对话后 `.pt` 逐渐分化。权重就是记忆，记忆就是自我。

---

## 核心思路

把**持久状态**实现为 Delta Rule 联想记忆矩阵 `W: (B, H, d_v, d_k)`，每个 batch × head 一个外积矩阵。

```
         ┌─────── 读（每层）──────┐
hidden → q_proj → q            W → o_proj → 加回 residual (identity-preserving)
                   └── q @ W^T ─┘

         ┌─────── 写（segment 末）───────┐
content → k_proj → k                      W_new = W + β · (v - W·k) ⊗ k^T
content → v_proj → v                                    │
content → beta_proj → β (per-head, per-token)           │
                                                        Delta Rule
```

**为什么 Delta Rule**：
- `(v - W·k)` 误差项天然处理"相似 key 的旧值消除 + 新值写入" —— 同类消歧 / 覆写 都不用额外机制
- 纯外积更新，零 softmax，线性复杂度
- 整个 state 是 W 矩阵本身，不是额外 token —— backbone 只处理纯 content

**零侵入 backbone**：通过 `layer_hook` 在每层之前做 `out = hidden + sigmoid(read_scale) · o_proj(q @ W^T)`，sigmoid(-3)≈0.047 初始接近 identity，对 backbone 原有能力完全保留。

**记忆体系分层**：

| 层 | 当前状态 | 载体 | 类比 | 时间尺度 |
|---|---|---|---|---|
| 短期工作记忆 | ✅ 已实现 | Delta Rule W | 海马体 / 松果体 | 单次对话动态演化 |
| 长期固化记忆 | 🚧 未来 | Memory MLP + Sleep | 皮层 synaptic consolidation | 跨 session 权重固化 |

当前 v5c 实现了短期层（W 矩阵），长期层（Memory MLP + Sleep 机制）是未来工作。详见 [docs/architecture.md](docs/architecture.md) 和 [docs/roadmap.md](docs/roadmap.md)。

---

## 架构

```
┌─────────────────────────────────┐
│   StateInterface (Delta Rule)   │  ← 可训练
│   W: (B, H=16, d_v=128, d_k=128)│
│   q/o_projs × n_layers (读)      │
│   k/v/beta_proj 全局共享 (写)    │
├─────────────────────────────────┤
│    Backbone (可切换，冻结 + LoRA) │  ← 只管语言适配
│    Qwen3.5-0.8B / 4B / 9B       │
└─────────────────────────────────┘
```

---

## 当前能力（v5c + persona_unified）

通过 `persona_unified` 统一分布训练后，心核同时具备：

| 能力 | Q 8 example | A |
|---|---|---|
| 世界知识 | "巴黎在哪里" | "巴黎是法国首都，位于塞纳河畔" |
| 拒答未告知 | "我叫什么？"（空状态）| "你还没告诉我你的名字呢" |
| 多 fact 单句记忆 | "我叫陈杰，35 岁，爱弹吉他" | ack，后续可分别召回 |
| 跨 chat 保留 | 告知 → N 轮世界 QA → 召回 | 值保留 |
| 覆写 | "不对我叫 X"→"不我叫 Y" | 最新值 Y |
| 多槽同时召回 | 告知名字+城市+爱好 → 依次问 | 全对 |

4b 从 backbone scratch 单 course 训 10000 步即达：VALUE 98.12 / WorldQA 98.31 / Refusal 98 / Compositional 100。

---

## 项目结构

```
configs/
  base.yaml                       # 训练默认参数
  qwen3.5-{0.8b,4b,9b}.yaml       # backbone 配置
  curriculum_persona.yaml         # ★ 2-stage 共享课程（bootstrap + unified）
  persona_unified.yaml            # 0.8b 训练入口
  persona_unified_4b.yaml         # 4b 训练入口
  curriculum.yaml, curriculum_*.yaml, think_*.yaml, migrate_*.yaml
                                  # legacy：旧 13-stage + think + migrate 课程，保留作参考

xinhe/
  model/
    state_plugin.py               # StateInterface：Delta Rule 读写
    xinhe_model.py                # 组装：backbone + layer_hook + plugin
    backbone.py, qwen_backbone.py # 可切换 backbone
    lora.py                       # LoRA 注入
  data/
    conversation.py               # 多轮对话数据集（tokenize + 多 value weight）
    persona.py                    # Persona 采样 + 槽位 + reveal 顺序
    generate_persona_data.py      # 主 orchestrator（10 种 turn kind + retention patterns）
    generate_memory_data.py       # 老的 memory 生成器（bootstrap 复用）
    refusal_templates.py          # 8 槽 × 8 surface 拒答库
    multi_fact_templates.py       # 多 fact 一句话模板
    deepseek_sampler.py           # DeepSeek V3 teacher cache 采样
  evaluation/
    persona_joint.py              # 3 新指标（WorldQA / Refusal / Compositional）
    metrics.py, probes.py         # legacy 记忆保留 / wipe 分析
  training/
    trainer.py                    # TBPTT + 4 指标联合早停
scripts/
  train.py                        # 训练入口
  chat.py                         # 交互式聊天
  chat_smoke.py                   # 批量人工验收脚本
  generate_data.py                # 统一数据分发（memory / persona）
  build_chat_cache.py             # DeepSeek teacher cache 构建
  build_val_sets.py               # 4 val 集生成
  eval_value_breakdown.py         # VALUE/FRAME/TELL 分解评估
  probe_beta.py                   # β 分布诊断
  shift_beta_bias.py              # 可选 bias 平移工具
  remote.py                       # 远端 VM deploy/start/logs 管理
```

---

## 快速开始

### 环境

```bash
uv sync
```

### 下载 backbone 权重

```bash
# 把 model.safetensors 放到 ./models/qwen3.5-0.8b/ 或 ./models/qwen3.5-4b/
```

### 准备 teacher cache + val 集（一次性）

```bash
# 1. DEEPSEEK_API_KEY 加到 .env 或 export
echo "DEEPSEEK_API_KEY=sk-..." >> .env

# 2. 采 teacher cache（~10 min, off-peak 更便宜，约 ¥10-15）
python scripts/build_chat_cache.py --n-chat 6000 --n-qa 4000

# 3. 建 4 val 集
python scripts/build_val_sets.py
```

### 训练

```bash
# 0.8b 端到端（bootstrap 1500 步 → main 3000-5000 步）
python scripts/train.py --config configs/persona_unified.yaml

# 4b 端到端（bootstrap ~1500 步 → main ~3000-5000 步）
python scripts/train.py --config configs/persona_unified_4b.yaml

# 从指定阶段开始
python scripts/train.py --config configs/persona_unified_4b.yaml --from-stage 1_persona_unified
```

### 聊天验证

```bash
python scripts/chat.py --checkpoint checkpoints/curriculum/persona_unified_4b.pt

# 命令
/save <name>   # 保存 .pt
/load <name>   # 加载
/wipe          # 清空状态（对比实验）
/stats         # state 分析（W 范数、有效秩等）
/burnin <text> # 用文本初始化 persona
```

### 批量验收

```bash
python scripts/chat_smoke.py --ckpt checkpoints/curriculum/persona_unified_4b.pt
# 跑 6 类场景（世界知识 / 拒答 / 多 fact / 覆写 / 单槽穿插 / 多槽 retention）
```

---

## 训练机制（persona_unified）

**单一课程，两个 stage**：

```
Stage 0: bootstrap
  2 轮 tell + recall，只 name 一类，freeze_lora
  → plugin 学会 Delta Rule 读写原语
  
Stage 1: persona_unified
  Persona 驱动多轮对话（12-20 turn），10 种 turn kind 混合：
    34% general_chat（DeepSeek teacher 采样，防 LoRA 漂移）
    10% world_qa（事实知识 rehearsal）
    14% reveal_single（个人信息披露）
    8%  reveal_multi（多 fact 一句话）
    15% recall（召回已披露）
    10% refusal（未披露 → 应拒答）
    4%  overwrite（Delta Rule 原生强项）
    3%  third_party（第三方人物）
    2%  compositional（跨槽组合）
  + 10% stress_retention（reveal → chat → recall）
  + 10% multi_slot_retention（多槽同时）
  → 联合 4 指标早停（VALUE 98 / WorldQA 85 / Refusal 95 / Compositional 95）
```

**关键哲学**：生成器是 permanent infrastructure。新发现的能力缺失 → 往 turn mix 加一个 kind + 一个比例，不改训练流程。

---

## 为什么这个方案

**v1-v4 的演化**（历史，详见 [docs/design_rationale.md](docs/design_rationale.md)）：
- v1 state-as-tokens：0.8b 跑通但 4b LoRA 全局共享瓶颈
- v2 对称 cross-attention：解决 LoRA 瓶颈但 entity 消歧 87%
- v3 EKS / v4 slot + key 路由：slot 身份成了伪概念
- v5b slot + contrastive：slot 线性投影 hash 碰撞
- **v5c Delta Rule**：`(v - Wk)` 误差项数学消歧，取消 slot 概念

**persona_unified 的来源**：v5c 跑完 13-stage memory + think 课程后 VALUE 98% 但 LoRA 漂到模板，问"巴黎在哪"答"新加拿地"。三个失败（拒答 / 多 fact / 世界知识）共享一个根因 —— 训练分布是真实使用分布的空集。

**解法**：替换 13-stage + think 窄分布 curriculum 为单一 persona_unified，含 teacher 采样的 chat/world_qa 作为 rehearsal，含 refusal / overwrite / retention 的结构化 patterns。用严阈值 4 指标联合早停。

---

## 相关工作

心核每个组件都有先例，但 "Delta Rule 联想记忆 + LoRA rehearsal 防漂 + 结构化 retention patterns + 统一分布" 这个组合是独特的。详见 [docs/related_work.md](docs/related_work.md)。

---

## 硬件

| Backbone | 训练显存（batch=1 + grad_checkpointing） |
|---|---|
| Qwen3.5-0.8B | ~10-12 GB |
| Qwen3.5-4B | ~18-22 GB |

远端训练通过 `scripts/remote.py` 管理（deploy / start / logs -f / stop）。

---

## 许可

研究项目，仅供学术探索。
