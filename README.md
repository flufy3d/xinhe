# Xinhe 心核

**验证一个假设：智能是否可以从"持续状态 + 可塑性 + 多时间尺度"中自然涌现？**

不做 RAG、不做模块拼装、不扩大 context window。用最小结构（小 transformer + 持久状态向量），让系统内部自发分化出记忆、抽象、快慢变量。

最终目标：一个没有上下文长度限制、能聊一辈子的 AI 朋友。

---

## 核心思路

把 **持久状态** 实现为额外的 token，拼在输入前面，复用 transformer 的 self-attention 做读写。不引入任何新的架构原语——如果智能真的能涌现，它应该能在这么简单的结构里自己出现。

```
输入:  [S_1..S_n | X_1..X_T]       状态token + 内容token
         ↓  transformer  ↓
输出:  [S'_1..S'_n | Y_1..Y_T]     更新后状态 + 预测

状态更新: state_next = gate * state_old + (1-gate) * S'
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
│       StatePlugin (~2M)         │  ← 可训练，sleep 时更新
│  state_emb, gate, scale         │
├─────────────────────────────────┤
│    Backbone (可切换)             │  ← 冻结 + LoRA
│  MiniMind 64M / Qwen3-0.6B 等   │
└─────────────────────────────────┘
```

StatePlugin 独立于 backbone。切换基座只需改配置文件，Plugin 代码不动。

---

## 项目结构

```
xinhe/
├── configs/          # 超参数配置 (base + 各 backbone)
├── models/           # 模型文件 (config/tokenizer 入库, 权重需手动下载)
├── docs/             # 架构详解、设计决策、实验路线
├── xinhe/            # 核心代码
│   ├── model/        # backbone 抽象 + 适配器 + StatePlugin + LoRA
│   ├── data/         # 多轮对话数据集
│   ├── training/     # 训练循环 (截断 BPTT)
│   ├── evaluation/   # 记忆保留 / wipe / 时间尺度分析
│   └── utils/        # checkpoint、logging
├── scripts/          # train / chat / evaluate / visualize
└── tests/            # pytest 测试
```

---

## 快速开始

### 环境

```bash
pip install -e .
```

### 下载权重

将 `model.safetensors` 放入对应 `models/` 子目录（tokenizer 等配置文件已入库）。

### 训练

```bash
python scripts/train.py --config configs/minimind.yaml
```

切换 backbone 只需换配置文件，如 `configs/qwen3-0.6b.yaml`。

### 聊天验证

```bash
python scripts/chat.py --checkpoint checkpoints/latest.pt
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
| 1. 基线 | MiniMind 能聊天 | 直接聊 |
| 2. 空状态 | 加 Plugin 不破坏 | 聊天质量不降 |
| 3. 1轮记忆 | 记住上一轮 | "我叫X" → 问 "我叫什么" |
| 4. 多轮记忆 | 跨轮保留 | 第1轮信息 → 第10轮还记得 |
| 5. 覆写 | 更新旧信息 | "在北京" → "搬到上海" → 问 |
| 6. Wipe | 证明依赖状态 | 清除后答不出 |
| 7. 时间尺度 | 快慢分化 | gate_bias 双峰分布 |
| 8. Sleep | 权重固化有效 | 第二天还记得昨天的事 |
| 9. 灵魂分化 | 不同 .pt 不同人格 | 同问题不同回答 |

详见 [docs/roadmap.md](docs/roadmap.md)

---

## 关键设计决策

**为什么不用 RAG？** State 存的是压缩理解，不是原文检索。信息已经消化了，不需要每次回忆都翻笔记本。

**为什么 Sleep 而不是实时学习？** 人脑也是白天积累、睡眠固化。集中更新更稳定、数据质量更高、推理零开销，而且有生命感。

**为什么固定容量的 state 够用？** 因为权重是第二层存储。State 只需存"今天"的工作记忆，真正重要的信息通过 sleep 写进权重。人脑也是固定容量。

**为什么 .pt 是灵魂？** 权重 + state 一起存，这就是这个 AI 的全部自我。每个用户的 AI 从同一起点分化，越聊越独特。

详见 [docs/design_rationale.md](docs/design_rationale.md)

---

## 硬件

| Backbone | 权重大小 | 训练显存 |
|----------|---------|---------|
| MiniMind 64M | 122MB | ~4-6GB |
| Qwen3-0.6B | 1.4GB | ~6-8GB |

RTX 5080 16GB 两个 backbone 都无压力。

---

## 许可

研究项目，仅供学术探索。
