# 实验路线与里程碑

---

## 总览

```
v1 架构验证（已完成）:
  阶段 A: 基础验证 ✅       阶段 B: 记忆涌现 ✅ (0.8B 达到 99%)
  1. 基线聊天 ✅             3. 1轮记忆 ✅
  2. 空状态不破坏 ✅          4. 多轮记忆 ✅
                           5. 信息覆写 ✅
                           6. Wipe对比 ✅
  → 4B entity 区分卡在 87%，确认 LoRA 瓶颈 → 触发 v2 架构重设计

v2 架构验证（进行中）:
  Phase 1: 0.8B 验证         Phase 2: 4B 验证         Phase 3: 迁移验证
  ──────────────            ──────────────           ──────────────
  v2 跑完整 14 阶段课程       4B 从零跑课程             0.8B core → 4B
  目标: ema_acc 95%+         目标: entity 突破 87%      冻结 core, 重训 proj+LoRA
  重点: stage 9a/9b          验证扩展瓶颈解决           验证迁移加速收敛

阶段 C: 在线学习（v2 验证后）:
  8. Sleep (Memory MLP + state弱化回放)
  9. 灵魂分化
  10. 消融实验
```

**关键发现**：
- 课程学习是训练 state 机制的核心策略（详见 `docs/curriculum_learning.md`）
- v1 在 4B 上暴露 LoRA 全局共享瓶颈 → v2 用对称 cross-attention + 专用投影解决

**课程三大类**: 基础记忆 (stages 0-13) / 思考泛化 (stage 14) / 基座迁移 (M0-M3)。
迁移时只携带 core（灵魂：state_emb + gate_proj），投影层和 LoRA 重新训练。

---

## 阶段 A：基础验证 ✅ (v1 完成)

### 1. 基线 ✅

**目标**：Backbone 正常加载能聊天

**通过标准**：对话流畅，回答合理

### 2. 空状态 ✅

**目标**：加 StateInterface (scale≈0) 后仍能正常聊天

**通过标准**：聊天质量与基线无明显差异

**验证了什么**：三层安全保护有效（LoRA 零初始化 + 状态近零 + 渐进 scale）

---

## 阶段 B：记忆涌现

### v1 验证 ✅ (0.8B 达到 99%)

v1 架构在 0.8B 上通过了全部 14 阶段课程：
- 1轮记忆 ✅、多轮记忆 ✅、信息覆写 ✅、Wipe 对比 ✅
- 4B 上 entity 区分卡在 87% → 确认 LoRA 瓶颈 → 触发 v2

### v2 验证（进行中）

#### Phase 1: 0.8B 验证

**目标**：v2 架构在 0.8B 上跑完整 14 阶段课程

**操作**：
- 实现 v2 StateInterface（对称 cross-attention）
- 从零训练，跑完整课程

**通过标准**：
- ema_acc 95%+（至少和 v1 持平）
- 重点关注 stage 9a/9b（entity 区分）的表现
- 如果 0.8B 上 v2 不如 v1 → 架构有回归，需排查

#### Phase 2: 4B 验证

**目标**：验证 v2 解决了 4B 扩展瓶颈

**操作**：
- 在 4B 上从零跑课程

**通过标准**：
- entity 区分突破 87%
- 如果成功 → v2 架构验证通过

#### Phase 3: 迁移验证

**目标**：验证 core 参数跨 backbone 迁移

**操作**：
- 0.8B core（state_emb, gate_proj）迁移到 4B
- 冻结 core，重训 projections + LoRA

**通过标准**：
- 迁移比从零训更快收敛

---

## 阶段 C：在线学习

### 8. Sleep — Memory MLP + State 弱化回放

**目标**：sleep 后对话信息固化进 Memory MLP，state 清空也能回答

**架构**：
- 专用 Memory MLP 叠加在 backbone 输出后（SwiGLU，零初始化）
- 对话时保存 (state_in, content, labels) 序列到 replay buffer
- Sleep 时回放序列，逐步弱化 state KV 注入（read_scale → 0），标准交叉熵 loss
- Memory MLP lr=1e-4，LoRA + StateInterface 冻结

**操作**：
- 聊天一天 → `/sleep`（回放 state 序列，弱化 state，更新 LoRA）
- 第二天加载 .pt → state 空白 → 验证还记得昨天的事
- 对比：不 sleep vs sleep 后

**通过标准**：
- sleep 后 Memory MLP 权重有变化（weight diff > 0）
- state 清空后仍能回忆 sleep 过的事实
- 新事实仍靠 state 正常记忆（不受 Memory MLP 干扰）

**Replay Buffer 策略**：
- Buffer 不在 sleep 后清空，作为长期档案持续积累
- Sleep 采样：70% 近期对话 + 30% 历史随机采样
- 混合回放巩固旧记忆 + 发现跨时间关联，防止只记今天忘昨天

**验证了什么**：
- 记忆从 state 转移到 Memory MLP 的通路有效
- 残差叠加不破坏原有能力
- 历史混合回放防止旧记忆被覆盖

### 9. 灵魂分化

**目标**：不同用户的 AI 变得不同

**操作**：
- 从同一个预训练 .pt 出发
- 用户 A 聊技术话题，经历多次 sleep
- 用户 B 聊生活话题，经历多次 sleep
- 用同样的问题测试两个 .pt

**通过标准**：
- 两个 .pt 的权重 diff 显著
- 对同样问题的回答风格/内容不同
- 各自记住各自用户的信息

**验证了什么**：.pt 确实在分化，"灵魂"在成长

### 10. 消融实验

逐个关闭组件，测量影响：

| 消融项 | 关闭方式 | 预期影响 |
|--------|---------|---------|
| state 数量 | n_state: 32→16→8→4 | 对话内 retention 下降 |
| gate 动态部分 | 去掉 gate_proj，固定 gate=0.5 | 覆写能力下降 |
| read_scale | 固定为 1.0（不渐进） | 训练初期不稳定，可能塌缩 |
| per-layer projection | 所有层共享一组 K/V 投影 | 层间记忆利用效率下降 |
| 写侧 cross-attention | 去掉写侧，直接用 content mean 更新 state | state 信息提取精度下降 |
| LoRA | lora_rank: 4→2→0 | backbone 语言适配能力下降 |
| Memory MLP | 去掉 Memory MLP | 跨天记忆消失，只能靠 state |
| state 弱化 | sleep 时不弱化 state | Memory MLP 学不到东西（state 兜底了） |
| replay 混合比例 | 100% 近期 / 0% 历史 | 旧记忆快速遗忘 |
| TBPTT 窗口 | tbptt_steps: 4→2→1 | 跨 segment 梯度断裂 |

---

## 未来方向（验证成功后）

### 近期
- ~~**Qwen 迁移**~~：已完成 QwenBackbone（支持 Qwen3.5 系列），切换只需改 yaml
- **更多事实类别**：从 3 类（name/city/number）扩展到 8 类（+food/job/hobby/age/pet），验证 state 容量上限
- **LLM 生成训练数据**：用大模型生成自然对话替代手写模板，提升泛化能力
- **更大 backbone**：Qwen3-4B 等更强模型（需量化或更大显存）
- **多模态状态**：图片、语音信息也消化进 state

### 中期 — 主动性涌现
- **空闲自转**：用户沉默时 state 不停，由轻量 MLP 持续 tick 演化，类似大脑默认模式网络
- **驱力维度**：state 中自然分化出"内稳态变量"，空闲时漂移、交互后回落，积累到阈值产生说话冲动
- **Initiative Head**：小分类器读 state 输出"该不该说话"，由 sleep 时用户反馈（回应=正、忽视=负）自动校准
- 三者合一：不同 .pt 经不同用户塑造后，涌现出不同的主动性人格——话痨、安静、只在特定时段搭话等

### 远期
- **元认知涌现**：模型能感知自身 state 变化并用语言描述（"我好像记错了"、"这个我不确定"），自省能力从 state 自注意力中自然长出
- **自主目标形成**：不靠外部指令，由 state 慢维度积累出稳定意图，驱动跨对话的持续行为（主动学某个话题、持续关注某件事）
- **情感回路闭环**：state 中的情绪维度不只影响生成风格，还反向调节学习率和记忆门控——开心时更容易记住，低落时更倾向回忆

---

## 硬件需求

| 阶段 | 显存 | 时间估算 |
|------|------|---------|
| 基线验证 | ~2GB | 几分钟 |
| 预训练 | ~4-6GB | 几小时（10K steps） |
| 聊天验证 | ~2GB | 实时 |
| Sleep (replay+权重更新) | ~4GB | 几分钟/次 |
| 评估 | ~2GB | 几十分钟 |

16GB+ GPU 全程无压力（4B 模型建议 24GB+）。
