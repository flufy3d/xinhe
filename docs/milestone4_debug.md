# Milestone 4 调试记录

## 问题

M3（distance=1, 16 种固定 filler, tell 固定在第一轮）训练成功：verify_state 33/33，scale 先降后涨（0.488→0.506），loss 收敛到 0.0003。

但扩展到多轮记忆后，**所有尝试都失败**——scale 持续下降，不出现拐点，模型没学会用 state。

## 实际聊天暴露的问题

用 M3 checkpoint 聊天发现：
1. **先说"你好"再说"我叫陈杰"** → 回忆失败。模型学到的是"记住第一轮"而非"记住事实"
2. **用训练外的话题闲聊**（电脑、回锅肉等）→ state 被污染，distance=1 也失败
3. **结论**：M3 的成功是过拟合到训练分布，不是真正的记忆泛化

## 已做的尝试

### 尝试 1：distance 1~10 + 100+ filler + 随机 tell 位置 + tbptt=7

**改动**：
- FILLERS 从 16 扩充到 100+（天气、美食、科技、旅行、哲学等）
- tell 前加 0~3 轮随机 filler（tell 位置随机化）
- 20% 覆写 episode（M5 数据）
- episode_length=14, tbptt_steps=7, batch_size=4

**结果**：
- Loss 呈双峰：短距离 episode 接近 0，长距离 >1，scale 持续下降
- 原因：tbptt=7 把 episode 切成 2 个窗口，tell 和 recall 跨窗口，梯度被 detach 切断

### 尝试 2：同上但 tbptt=14（整个 episode 不切断）

**改动**：tbptt_steps=14, batch_size=1（显存限制）

**结果**：
- 从头训到 step 5100，scale 持续下降（0.50→0.47），loss 在 0.3~1.0 波动
- verify_state: 有 state 和无 state 结果几乎一样 → 模型完全没学会用 state

### 尝试 3：缩小距离到 1~3，episode_length=8

**改动**：
- max-distance=3, max-turns=8, episode_length=8, tbptt_steps=8
- batch_size=1

**结果**（机器 1, rank=4）：
- Step 6650: scale=0.4628，持续下降，loss ~0.5，无拐点

### 尝试 4：同上但 LoRA rank 16（机器 2）

**改动**：lora rank=16, alpha=32（可训练参数 3.88M → 5.60M）

**结果**：
- Step 5250: scale=0.4720，同样持续下降，和 rank=4 表现一样
- 结论：**模型容量不是瓶颈**

## 对比：M3 为什么成功

| 参数 | M3（成功） | M4 尝试（失败） |
|------|-----------|----------------|
| distance | 1 | 1~3 或 1~10 |
| filler 种类 | 16 种固定 | 100+ 种多样 |
| tell 位置 | 固定第一轮 | 随机（0~3 轮前置 filler） |
| episode_length | 4 | 8 或 14 |
| tbptt_steps | 4（=episode_length） | 7, 8, 14 |
| batch_size | 4 | 1~2 |
| 覆写数据 | 无 | 20% |

一次改了太多变量，不知道是哪个（或哪几个组合）导致失败。

## 下一步：隔离实验

逐个变量测试，找到导致失败的关键因素：

### 实验 A：M3 参数 + 多样 filler

- episode_length=4, tbptt_steps=4, batch_size=4
- distance=1, tell 固定第一轮（去掉 pre-filler）
- 100+ 种 filler
- 无覆写数据（覆写是 M5 的事，先不混入）
- **如果成功**：filler 多样性不是问题，问题在 distance / tell 位置
- **如果失败**：filler 多样性本身就导致学不会

### 实验 B：M3 参数 + 随机 tell 位置

- episode_length=4, tbptt_steps=4, batch_size=4
- distance=1, 16 种固定 filler
- tell 前加 0~1 轮 filler（位置随机化）
- **如果成功**：tell 位置随机化不是问题
- **如果失败**：tell 位置随机化导致学不会

### 实验 C：M3 参数 + distance 1~3

- episode_length=6, tbptt_steps=6, batch_size=2
- 16 种固定 filler, tell 固定第一轮
- distance=1~3
- **如果成功**：距离增大不是问题（只要 tbptt 覆盖全 episode）
- **如果失败**：距离本身就是问题

所有隔离实验都不包含覆写数据——覆写是 M5 的目标，M4 先只解决多轮记忆。等 M4 通过后再加入覆写数据。

根据实验结果，逐步组合成功的变量，最终达成 M4 目标。

## 代码改动备注

需要修改 `generate_memory_data.py` 中的 `generate_episode()` 来控制 pre-filler：
- 当前：`pre_filler_count = rng.randint(0, 3)` 固定存在
- 实验 A/C 需要：`pre_filler_count = 0`（禁用前置 filler）
- 建议加命令行参数 `--no-pre-filler` 控制

`configs/base.yaml` 需要对应调整 episode_length, tbptt_steps, batch_size。
