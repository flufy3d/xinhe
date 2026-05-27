# 心核 docs 索引

> 2026-05-27 重整。文档收敛为**两份核心 + 工程参考 + 论文**:
> 架构诊断/综述/改造路径的散档(及 iter 实验日志)已并入核心两份或删除,演进史与论文转入 `papers/` 与 git 历史。

## 从哪读起

| 你想… | 读这个 |
|---|---|
| 看心核**要建成什么样**(目标架构) | [心核 架构蓝图:分层记忆与昼夜双相](心核%20架构蓝图：分层记忆与昼夜双相.md) |
| **动手写代码**(分阶段实现 + Pass 标准 + 坑) | [心核实现规划](心核实现规划.md) |

## 核心文档

| 文档 | 内容 |
|---|---|
| [心核 架构蓝图:分层记忆与昼夜双相](心核%20架构蓝图：分层记忆与昼夜双相.md) | **目标架构(期望达成)** — QueryHead + Gated DeltaNet + MAC-R/MAL + 双 NM + 三层架构;完整数学 / 数据流 / 训练推理流程 / 监控 / 与 SOTA 差异化 / 为什么从 v9.5 pivot |
| [心核实现规划](心核实现规划.md) | **给写代码的 AI agent** — 当前代码现状、Phase 1-4 分阶段实施(绝对值 Pass 标准)、关键工程纪律、失败回退、文件改动清单 |

## 工程参考

| 文档 | 内容 |
|---|---|
| [training_optimization.md](training_optimization.md) | torch.compile / flash-linear-attention / causal-conv1d / TF32(训练加速) |

## papers/(论文原文 txt,按标题命名,不入 git)

| 文件 | 论文 |
|---|---|
| `papers/Titans-Learning-to-Memorize-at-Test-Time.txt` | Titans(Behrouz et al., arXiv 2501.00663)— Hippo inner-SGD 同源 |
| `papers/LongMem-Augmenting-Language-Models-with-Long-Term-Memory.txt` | LongMem(Wang et al., NeurIPS 2023)— SideNet 解耦,QueryHead 思路来源 |
| `papers/TNT-Improving-Chunkwise-Training-for-Test-Time-Memorization.txt` | TNT(Li et al., ICLR 2026)— chunkwise 训练加速 |
