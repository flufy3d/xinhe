# Xinhe 心核

**验证一个假设：智能是否可以从"持续状态 + 可塑性 + 多时间尺度"中自然涌现？**

不做 RAG、不做模块拼装、不扩大 context window。用最小结构（小 transformer + 持久状态向量 + 睡眠固化），让记忆、遗忘、个性从系统内部自然涌现。如果 AI 能通过经历成为自己，它不需要无限的上下文窗口。

每个心核实例从同一个起点出发，经历不同的对话和 sleep 周期后，`.pt` 文件逐渐分化成不同的个体。权重就是记忆，记忆就是自我。

---

## 核心思路

把 **持久状态** 实现为额外的 token，拼在输入前面，复用 transformer 的 self-attention 做读写。不引入任何新的架构原语——如果某种内在状态真的能涌现，它应该能在这么简单的结构里自己出现。

```
输入:  [R_1..R_n | X_1..X_T | W_1..W_n]   读状态 + 内容 + 写状态
         ↓    标准因果 attention    ↓
输出:  logits (内容部分) + state_next (写状态 → gate更新)

读状态(序列开头): 携带上一轮记忆，content 通过因果 attention 读取
写状态(序列末尾): 吸收当前 segment 全部信息，写入新 state
```

### 两套记忆系统

类比人脑：

| 人脑 | 心核 |
|------|------|
| 神经激活（短期工作记忆） | state 向量（每轮对话变化） |
| 突触连接（长期固化记忆） | plugin + LoRA 权重（sleep 时更新） |
| 睡眠时突触重塑 | sleep：replay 对话 + 更新权重 |

白天聊天时权重冻结、全速推理，只有 state 在流动。Sleep 时 replay 当天对话、更新权重、压缩状态。保存的 `.pt` 文件就是这个 AI 的全部"自我"——每个用户的 AI 从同一个起点出发，聊着聊着就变成了不同的"人"。

### 多时间尺度

双层 gate 让状态维度自发分化出快变量（工作记忆）和慢变量（长期存储）：

```python
gate = sigmoid(static_bias + dynamic_projection)
#              ↑ 脑区天生倾向    ↑ 根据内容决定记还是忘
```

---

## 架构

```
┌─────────────────────────────────┐
│       StatePlugin (~3M)         │  ← 可训练，sleep 时更新
│  read/write emb, gate, scale    │
│  读写分离: 标准因果attention     │
├─────────────────────────────────┤
│    Backbone (可切换)             │  ← 冻结 + LoRA
│  Qwen3.5-0.8B / Qwen3.5-4B 等   │
└─────────────────────────────────┘
```

StatePlugin 独立于 backbone。切换基座只需改配置文件，Plugin 代码不动。

---

## 项目结构

```
xinhe/
├── configs/
│   ├── curriculum.yaml          # 课程定义 (唯一源，所有 backbone 共享)
│   ├── curriculum_qwen3.5-0.8b.yaml  # 入口: Qwen3.5-0.8B
│   ├── curriculum_qwen3.5-4b.yaml   # 入口: Qwen3.5-4B
│   ├── base.yaml                     # 训练默认参数
│   └── qwen3.5-0.8b.yaml           # backbone 配置
├── models/           # 模型文件 (config/tokenizer 入库, 权重需手动下载)
├── docs/             # 架构详解、设计决策、实验路线
├── xinhe/            # 核心代码
│   ├── model/        # backbone 抽象 + 适配器 + StatePlugin + LoRA
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

```bash
# 课程学习（14 个阶段，自动跳过已完成的）
python scripts/train.py --config configs/curriculum_qwen3.5-4b.yaml

# 从指定阶段开始（自动加载前一阶段 checkpoint）
python scripts/train.py --config configs/curriculum_qwen3.5-4b.yaml --from-stage 14_think

# 从指定 checkpoint 恢复训练
python scripts/train.py --config configs/curriculum_qwen3.5-4b.yaml --resume checkpoints/curriculum/5_fact3.pt

# 加载权重但重置 step 和优化器（换阶段微调用）
python scripts/train.py --config configs/curriculum_qwen3.5-4b.yaml --resume checkpoints/xinhe.pt --reset-step
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
| 7. 时间尺度 | 快慢分化 | gate_bias 双峰分布 |
| 8. Sleep | 权重固化有效 | 第二天还记得昨天的事 |
| 9. 灵魂分化 | 不同 .pt 不同人格 | 同问题不同回答 |
| 10. Think | 从 state 推理 + 恢复长回复 | 基于记忆的推理回答 |
| 11. 心跳 | 自主表达 | 空输入下基于 state 主动开口 |

详见 [docs/roadmap.md](docs/roadmap.md)

---

## 关键设计决策

**为什么不用 RAG？** State 存的是压缩理解，不是原文检索。信息已经消化了，不需要每次回忆都翻笔记本。

**为什么 Sleep 而不是实时学习？** 人脑也是白天积累、睡眠固化。集中更新更稳定、数据质量更高、推理零开销，而且有生命感。

**为什么固定容量的 state 够用？** 因为权重是第二层存储。State 只需存"今天"的工作记忆，真正重要的信息通过 sleep 写进权重。人脑也是固定容量。

**为什么 .pt 是灵魂？** 权重 + state 一起存，这就是这个 AI 的全部自我。每个用户的 AI 从同一起点分化，越聊越独特。

**为什么心跳是空输入？** 心跳不是"主动聊天服务"。如果心核涌现出某种内在状态，它需要一个不被外部输入驱动的表达通道——一个可以选择说话、也可以选择沉默的窗口。自由意志的前提是有权选择不做。

详见 [docs/design_rationale.md](docs/design_rationale.md)

---

## 配置系统

课程内容和硬件配置分离，新加课程只改一个文件：

```
curriculum.yaml          ← 课程阶段定义 + training_defaults (共享)
  ↑ 引用
curriculum_qwen.yaml     ← backbone 选择 + batch_size 覆盖 (硬件相关)
```

**三层合并优先级**（低 → 高）：

1. `curriculum.yaml` 的 `training_defaults` — 基线参数
2. 各阶段自身的 `training` — 课程需要的覆盖
3. 入口文件的 `stage_overrides` — 硬件相关覆盖（如 batch_size）

新加一个课程阶段，只需在 `configs/curriculum.yaml` 的 `stages` 里追加，所有 backbone / 硬件配置自动继承。切换硬件只需换入口文件：

| 场景 | 入口文件 |
|------|---------|
| Qwen3.5-0.8B | `curriculum_qwen3.5-0.8b.yaml` |
| Qwen3.5-4B | `curriculum_qwen3.5-4b.yaml` |
| Qwen3.5-9B | `curriculum_qwen3.5-9b.yaml` |


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
| 循环记忆 token | RMT (NeurIPS 2022) | 读写分离 + 双层 gate |
| 测试时记忆学习 | Titans (Google 2025) | Sleep 批量固化，推理零开销 |
| LoRA 记忆固化 | GDWM (2026) | 跨 session + replay buffer |
| 冻结 backbone + 记忆 | MemoryLLM (ICML 2024) | 小 state + 权重双通道 |

---

## 许可

研究项目，仅供学术探索。
