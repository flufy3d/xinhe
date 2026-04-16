# Xinhe 心核

**验证一个假设：智能是否可以从"持续状态 + 可塑性 + 多时间尺度"中自然涌现？**

不做 RAG、不做模块拼装、不扩大 context window。用最小结构（小 transformer + 持久状态向量 + 睡眠固化），让记忆、遗忘、个性从系统内部自然涌现。如果 AI 能通过经历成为自己，它不需要无限的上下文窗口。

每个心核实例从同一个起点出发，经历不同的对话和 sleep 周期后，`.pt` 文件逐渐分化成不同的个体。权重就是记忆，记忆就是自我。

---

## 核心思路

把 **持久状态** 通过对称的 cross-attention 接入 transformer：读是 Content(Q) × State(K,V)，写是 State(Q) × Content(K,V)。利用 HuggingFace `past_key_values` 标准 API 注入每层 attention cache，零侵入 backbone 内部。

```
state_old
  ├── 读: state → 专用 K/V 投影 → 注入每层 KV-Cache
  │                                  ↓
  │   [Content(T)] → backbone 32 层 → content_final → logits
  │                  (每层 attend 到 state K/V)
  │
  └── 写: state → 专用 Q 投影 → attend to content_final
                                     ↓
                                state_next (gate 更新)
```

### 两套记忆系统

类比人脑：

| 人脑 | 心核 |
|------|------|
| 神经激活（短期工作记忆） | state 向量（每轮对话变化） |
| 突触连接（长期固化记忆） | Memory MLP 权重（sleep 时更新） |
| 睡眠时突触重塑 | sleep：replay 对话 + 更新权重 |

白天聊天时权重冻结、全速推理，只有 state 在流动。Sleep 时 replay 当天对话、更新权重、压缩状态。保存的 `.pt` 文件就是这个 AI 的全部"自我"——每个用户的 AI 从同一个起点出发，聊着聊着就变成了不同的"人"。

### 记忆体系分层

State 是工作记忆（海马体），gate 管理其刷新；MLP 权重是长期记忆（皮层），sleep 时固化：

```python
gate = sigmoid(gate_proj(cat[state_old, state_new]))
#              纯动态决策：此刻该保持还是该更新？
```

---

## 架构

```
┌─────────────────────────────────┐
│     StateInterface (~5M)        │  ← 可训练，sleep 时更新
│  read K/V projs, write Q,       │
│  gate, read_scale               │
│  对称 cross-attention 读写       │
├─────────────────────────────────┤
│    Backbone (可切换)             │  ← 冻结 + LoRA（只管语言适配）
│  Qwen3.5-0.8B / Qwen3.5-4B 等   │
└─────────────────────────────────┘
```

StateInterface 独立于 backbone。切换基座只需改配置文件，StateInterface 代码不动。

---

## 项目结构

```
xinhe/
├── configs/
│   ├── curriculum.yaml               # 基础记忆课程 stages 0-13 (共享)
│   ├── curriculum_think.yaml         # 思考泛化课程 (共享)
│   ├── curriculum_migrate.yaml       # 基座迁移课程 M0-M3 (共享)
│   ├── curriculum_qwen3.5-0.8b.yaml  # 入口: 0.8B 基础记忆
│   ├── think_qwen3.5-4b.yaml        # 入口: 4B 思考泛化
│   ├── migrate_0.8b_to_4b.yaml      # 入口: 0.8B→4B 迁移
│   ├── base.yaml                     # 训练默认参数
│   └── qwen3.5-0.8b.yaml           # backbone 配置
├── models/           # 模型文件 (config/tokenizer 入库, 权重需手动下载)
├── docs/             # 架构详解、设计决策、实验路线
├── xinhe/            # 核心代码
│   ├── model/        # backbone 抽象 + 适配器 + StateInterface + LoRA
│   ├── data/         # 数据集 + 数据生成
│   │   ├── conversation.py         # 多轮对话数据集
│   │   ├── generate_memory_data.py # 记忆数据生成
│   │   ├── generate_think_data.py  # Think 数据生成
│   │   └── think_lang.py           # Think 模板语言包 (en/zh)
│   ├── training/     # 训练循环 (截断 BPTT)
│   ├── evaluation/   # 记忆保留 / wipe / 时间尺度分析
│   └── utils/        # checkpoint、logging
├── scripts/
│   ├── train.py                 # 训练入口
│   ├── chat.py                  # 交互式聊天
│   └── generate_data.py         # 统一数据生成入口
└── tests/            # pytest 测试
```

---

## 快速开始

### 环境

```bash
uv sync
```

### 下载权重

将 `model.safetensors` 放入对应 `models/` 子目录（tokenizer 等配置文件已入库）。

### 生成训练数据

训练时会自动生成数据，也可以提前手动生成。统一入口是 `scripts/generate_data.py`：

```bash
# 为指定阶段生成数据
python scripts/generate_data.py --config configs/curriculum_qwen3.5-4b.yaml --stage 13_all

# 生成所有阶段数据
python scripts/generate_data.py --config configs/curriculum_qwen3.5-4b.yaml --all

# Think 数据需要 backbone 推理 (~1-2 小时)，生成后自动缓存
# 重新生成用 --force
python scripts/generate_data.py --config configs/curriculum_qwen3.5-4b.yaml --stage 14_think --force

# 预览数据（不写文件）
python -m xinhe.data.generate_memory_data --preview 3
python -m xinhe.data.generate_think_data --preview 3
```

数据格式见 [docs/data_spec.md](docs/data_spec.md)

### 训练

三类课程独立执行：

```bash
# ① 基础记忆（14 个阶段，纯 state 读写，自动跳过已完成的）
python scripts/train.py --config configs/curriculum_qwen3.5-0.8b.yaml

# 从指定阶段开始
python scripts/train.py --config configs/curriculum_qwen3.5-0.8b.yaml --from-stage 5_fact3

# 从 checkpoint 恢复
python scripts/train.py --config configs/curriculum_qwen3.5-0.8b.yaml --resume checkpoints/curriculum/5_fact3.pt
```

```bash
# ② 基座迁移（4 阶段：投影热身 → LoRA 适配 → 联合微调 → 全能力恢复）
python scripts/train.py \
  --config configs/migrate_0.8b_to_4b.yaml \
  --migrate-from checkpoints/curriculum/13_all.pt
```

```bash
# ③ 思考泛化（在目标 backbone 上，基础记忆或迁移完成后）
python scripts/train.py --config configs/think_qwen3.5-4b.yaml
```

### 聊天验证

```bash
# 默认: 流式逐字输出 + 显示思考过程
python scripts/chat.py --checkpoint checkpoints/latest.pt

# 隐藏思考过程
python scripts/chat.py --checkpoint checkpoints/latest.pt --hide-think

# 关闭流式输出 (一次性返回)
python scripts/chat.py --checkpoint checkpoints/latest.pt --no-stream

# 强制思考模式 (生成时以 <think> 开头)
python scripts/chat.py --checkpoint checkpoints/latest.pt --think
```

聊天命令：

| 命令 | 说明 |
|------|------|
| `/save <name>` | 保存 .pt（权重 + 状态 + buffer） |
| `/load <name>` | 加载 .pt（恢复完整"灵魂"） |
| `/wipe` | 清除状态（对比实验） |
| `/stats` | 显示状态分析 |
| `/burnin <prompt>` | 用 prompt 初始化 persona |

### 评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/latest.pt
python scripts/visualize_state.py --checkpoint checkpoints/latest.pt
```

---

## 实验路线

| 阶段 | 目标 | 验证方式 |
|------|------|---------|
| 1. 基线 | Backbone 能聊天 | 直接聊 |
| 2. 空状态 | 加 Plugin 不破坏 | 聊天质量不降 |
| 3. 1轮记忆 | 记住上一轮 | "我叫X" → 问 "我叫什么" |
| 4. 多轮记忆 | 跨轮保留 | 第1轮信息 → 第10轮还记得 |
| 5. 覆写 | 更新旧信息 | "在北京" → "搬到上海" → 问 |
| 6. Wipe | 证明依赖状态 | 清除后答不出 |
| 7. 时间尺度 | 记忆分层 | gate 行为分析 + state 维度探针 |
| 8. Sleep | 权重固化有效 | 第二天还记得昨天的事 |
| 9. 灵魂分化 | 不同 .pt 不同人格 | 同问题不同回答 |
| 10. Think | 从 state 推理 + 恢复长回复 | 基于记忆的推理回答 |
| 11. 心跳 | 自主表达 | 空输入下基于 state 主动开口 |

详见 [docs/roadmap.md](docs/roadmap.md)

---

## 关键设计决策

**为什么用对称 cross-attention？** v1 的 state-as-tokens 在 0.8B 上验证成功，但 4B 上 LoRA 瓶颈暴露。v2 用专用投影做 state 路由，LoRA 只管语言适配，彻底绕过低秩限制。

**为什么不用 RAG？** State 存的是压缩理解，不是原文检索。信息已经消化了，不需要每次回忆都翻笔记本。

**为什么 Sleep 而不是实时学习？** 人脑也是白天积累、睡眠固化。集中更新更稳定、数据质量更高、推理零开销，而且有生命感。

**为什么固定容量的 state 够用？** 因为权重是第二层存储。State 只需存"今天"的工作记忆，真正重要的信息通过 sleep 写进权重。人脑也是固定容量。

**为什么 .pt 是灵魂？** 权重 + state 一起存，这就是这个 AI 的全部自我。每个用户的 AI 从同一起点分化，越聊越独特。

**为什么心跳是空输入？** 心跳不是"主动聊天服务"。如果心核涌现出某种内在状态，它需要一个不被外部输入驱动的表达通道——一个可以选择说话、也可以选择沉默的窗口。自由意志的前提是有权选择不做。

详见 [docs/design_rationale.md](docs/design_rationale.md)

---

## 配置系统

三类课程，三个定义文件，内容和硬件配置完全分离：

```
curriculum.yaml           ← 基础记忆 stages 0-13 (共享)
curriculum_think.yaml     ← 思考泛化 (共享)
curriculum_migrate.yaml   ← 基座迁移 M0-M3 (共享)
  ↑ 引用
curriculum_qwen3.5-*.yaml ← 基础记忆入口 + batch_size 覆盖
think_qwen3.5-*.yaml      ← 思考泛化入口
migrate_*_to_*.yaml       ← 迁移入口
```

| 课程类别 | 定义文件 | 说明 |
|---------|---------|------|
| 基础记忆 | `curriculum.yaml` | 纯 state 读写，不含 think 数据 |
| 思考泛化 | `curriculum_think.yaml` | 从 state 推理 + 恢复长回复 |
| 基座迁移 | `curriculum_migrate.yaml` | 适配 plugin core 到新 backbone |

| 场景 | 入口文件 |
|------|---------|
| 0.8B 基础记忆 | `curriculum_qwen3.5-0.8b.yaml` |
| 4B 基础记忆 | `curriculum_qwen3.5-4b.yaml` |
| 0.8B → 4B 迁移 | `migrate_0.8b_to_4b.yaml` |
| 4B 思考泛化 | `think_qwen3.5-4b.yaml` |


---

## 硬件

| Backbone | 训练显存 |
|----------|---------|
| Qwen3.5-0.8B | ~6-8GB |
| Qwen3.5-4B | ~24GB |

16GB+ 显存的 GPU 可训练 0.8B，4B 模型建议 24GB+。

---

## 相关工作

心核的每个组件都有前人探索，但完整组合是独特的。详见 [docs/related_work.md](docs/related_work.md)。

| 组件 | 最近的先例 | 心核的扩展 |
|------|-----------|-----------|
| 循环记忆 | RMT (NeurIPS 2022) | 对称 cross-attention + 专用投影解耦 LoRA |
| 测试时记忆学习 | Titans (Google 2025) | Sleep 批量固化，推理零开销 |
| LoRA 记忆固化 | GDWM (2026) | 跨 session + replay buffer |
| 冻结 backbone + 记忆 | MemoryLLM (ICML 2024) | 小 state + 权重双通道 |

---

## 许可

研究项目，仅供学术探索。
