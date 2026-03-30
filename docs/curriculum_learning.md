# 课程学习：心核训练的核心策略

## 发现过程

M4 阶段我们尝试一步到位训练多轮记忆（多样 filler + 随机 tell 位置 + 长距离），连续失败 4 次。通过隔离实验发现：每个变量单独加入都可能导致失败，但从已有 checkpoint 出发逐步加难度，每一步都能在几百步内收敛。

**核心矛盾**：模型需要同时学会"state 怎么用"和"什么时候用"，这两个目标互相依赖。从零开始训练时，state 初始是噪声，模型发现用 state 反而有害，于是学会了"不用 state"——陷入局部最优，永远无法发现 state 的价值。

## 为什么课程学习有效

课程学习将一个不可解的端到端问题拆成多个可解的子问题：

1. 先在最简单的场景下让模型发现 state 有用（scale 上升）
2. 建立基本读写能力后，每次只增加一个维度的难度
3. 模型在每个阶段已经掌握了 state 机制，只需在新条件下微调

这不是心核特有的问题。AI 领域中，当系统有一个"可选组件"（state / tool / 新模态），且该组件需要先学会才能体现价值时，直接端到端训练几乎必然失败。大模型的分阶段训练（预训练→SFT→RLHF）、强化学习的课程设计、Progressive GAN，本质上都是同一个道理。

## 失败的教训

1. **不要一次改多个变量**。我们最初同时改了 filler 数量、tell 位置、distance、episode 长度、batch size，浪费了大量算力却无法定位问题
2. **Scale 是最重要的监控指标**。Scale 持续下降 = 模型在放弃 state，不需要等 loss 收敛就能判定失败
3. **Gate bias 调参无法解决根本问题**。我们试了 gate_bias=2.0，完全无效。问题不是"记不住"，而是"根本没学会写入"。参数调优解决不了学习路径的问题
4. **Batch size 不是万能解释**。我们一度怀疑 batch=1 的梯度噪声是元凶，实验证明不是

## 对后续训练的指导

**每次引入新能力都应该用课程学习**，包括但不限于：
- M5 覆写能力：先在当前 checkpoint 上加入覆写数据
- 新的事实类别：先少量混入，确认 scale 稳定后再增加比例
- 更长的对话距离：逐步拉长，而非直接跳到目标距离

**课程阶段之间的参数**：
- 小学习率（1e-4）
- 自动重置 optimizer，保留模型权重
- 每阶段设置 `early_stop_loss` + `early_stop_patience`，收敛后自动进入下一阶段

## 使用方法

课程学习已集成到训练主流程，在一个 YAML 中定义所有阶段：

```bash
# 从头训练完所有阶段
python scripts/train.py --config configs/curriculum_qwen.yaml

# 从指定阶段开始（自动加载前一阶段 checkpoint）
python scripts/train.py --config configs/curriculum_qwen.yaml --from-stage 3_distance
```

每个阶段自动生成数据、训练、保存 checkpoint 到 `checkpoints/curriculum/{stage_name}.pt`。已完成的阶段会自动跳过。

扩展新能力（如 M5 覆写）只需在 `curriculum_qwen.yaml` 末尾追加新阶段。
