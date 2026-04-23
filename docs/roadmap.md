# 实验路线与里程碑

---

## 总览（按时间轴）

```
阶段 A: 架构定型（v1 → v5c）✅
  v1 state-as-tokens:      0.8b 通过，4b 87% 卡 → LoRA 瓶颈暴露
  v2 对称 cross-attention: LoRA 瓶颈解决，同类消歧 93% 卡
  v3/v4 slot routing:      slot 身份伪概念
  v5b slot+contrastive:    hash 碰撞无解
  v5c Delta Rule (2026-04):  (v-Wk) 数学消歧，架构定型

阶段 B: 训练范式定型（persona_unified）✅
  v5c 13-stage memory + think 课程后 VALUE 98% 但 LoRA 漂，
    真实 chat 答错"新加拿地"/乱编名字 → 诊断为训练分布窄化
  persona_unified (2026-04-22): 替换为 persona 驱动对话 + DeepSeek rehearsal
  persona_retention_v2: 加 stress + multi_slot retention patterns
  4b single-course: 4b 从 scratch 单课 VALUE 98%+，chat_smoke 23/27
  2-stage bootstrap refactor (2026-04-23): curriculum_persona.yaml 定型

阶段 C: 未开启（未来）
  在线学习: 当前只在训练阶段 fine-tune LoRA，部署时权重冻结
  灵魂分化: 不同用户 .pt 的权重差异分析
  消融实验: 各组件的边际贡献
  心跳 / 主动表达: 空输入下基于 W 的自发生成
```

---

## 已完成（核心三大问题解决）

### ✅ 问题 1: 不知道要说不知道

- v5c 后 chat 测试: "我叫什么？"（空状态）→ 编造一个名字
- 根因: 训练数据 100% 假设 state 里有 answer，没有"未告知"样本
- 修复: persona_unified 的 refusal turn kind（10%）+ 8 槽 × 8 variant 拒答模板
- 验证: 4b chat_smoke [B] 拒答 5/5，真实 chat "我叫什么"（空状态）→ "你还没告诉我你的名字呢" ✅

### ✅ 问题 2: 多 fact 一句话崩盘

- v5c 后: "我叫陈杰 35 岁爱弹吉他" → 年龄/爱好能召回，名字不行
- 根因: 训练里 FACT_TEMPLATES 一句 1 个 fact，k/v_proj 没学过单 utterance 多语义事件
- 修复: reveal_multi turn kind（8%）+ `value: list[str]` 多 value 权重 + multi_fact_templates
- 验证: chat_smoke [C] 多 fact 3/3，真实 chat 4b 一句话 4 fact ack 完美 ✅

### ✅ 问题 3: 世界知识遗忘

- v5c 后: "巴黎在哪"答"新加拿地"
- 根因: 13-stage 窄分布训练 LoRA 漂到"模板填空"模式
- 修复: DeepSeek V3 采样 world_qa + general_chat 作为 34%+10% rehearsal
- 验证: 4b chat_smoke [A] 世界知识 5/5（巴黎 → 法国北部塞纳河畔、周杰伦 1979 台湾、四大发明正确、瑞利散射） ✅

### ✅ 问题 4: 跨 chat 保留

- v5c 后: 告知名字 + 世界 QA 穿插 → 忘了名字
- 根因: 训练 episode 没有足够"reveal → chat → recall"结构化样本
- 修复: stress_retention (10%) 和 multi_slot_retention (10%) 结构化 episode
- 验证: chat_smoke [E] 单槽穿插 5/5 (0.8b), 4/5 (4b), real chat retention 良好 ✅

### ✅ 问题 5: Think 课程

- v5c 后: think 课程 TELL 66%，推理质量差
- 解决方案: 彻底删除 think 课程。reasoning 能力由 4b backbone 自带，不需要单独训
- 验证: 4b chat 含 `<think>` 的问题（瑞利散射等）自然生成详细解释 ✅

---

## 当前架构能力矩阵

| 能力 | 0.8b (persona_retention_v2) | 4b (persona_unified_4b, single stage) |
|---|---|---|
| 世界知识 | 4/5 | **5/5** |
| 拒答 | 3/5 | **5/5** |
| 多 fact 单句 | 3/3 | **3/3** |
| 覆写 | 1/1 | **1/1** |
| 单槽穿插召回 | 5/5 | 4/5 |
| 多槽 retention | 6/8 | 5/8 |
| **合计** | 22/27 (81.5%) | **23/27 (85.2%)** |

4b 的 backbone 原生能力（世界知识、指令跟随、多语言）让它全面碾压 0.8b。0.8b 在 retention 稍强是因为经过了多轮 retention-specific 微调。

---

## 未来方向

### 近期（按优先级）

**1. 4b 端到端 2-stage 验证（bootstrap + main）**
- 当前 4b 用的是 single-stage (10000 步)
- 2-stage 预期 ~5000-6000 步达同等水平（省时间）
- 配置已就绪：`configs/persona_unified_4b.yaml` + `curriculum_persona.yaml`

**2. 剩余 chat_smoke 失败场景**
- [E] 单槽穿插 4/5（偶尔失败）：加更多高 chat-turn 数的 retention pattern
- [F] 多槽 retention 5/8：加更长 reveal chain (4-5 槽)
- 英文 query / 中文 value retention：加 bilingual_ratio turn kind

**3. 多种 backbone 验证**
- Qwen3.5-9b（configs 已就绪）
- 其他架构：Llama / Mistral 等，看 Delta Rule 能否泛化

### 中期：长期记忆固化（Memory MLP + Sleep）

**4. 实现长期记忆层（当前最优先的未来工作）**
- **定位**：当前 W 矩阵是**短期工作记忆**（海马体 / 单次对话内动态演化）。Memory MLP + Sleep 是**长期记忆**（皮层 / 跨 session 权重级固化）
- **架构**：Memory MLP（SwiGLU 独立模块）叠加在 backbone 输出后
- **Sleep 机制**：replay buffer 回放对话 + 逐步弱化 W 注入（read_scale → 0）→ 迫使信息从 W 转移到 MLP 权重
- **采样策略**：70% 近期 + 30% 历史混合，防旧记忆覆盖
- **效果**：sleep 过的事实 W 清空也能回答；`.pt` 分化更深刻（MLP 权重随用户独立）
- **拆分**：先做离线 sleep（固定间隔触发），后做在线 sleep（用户使用中后台触发）

**5. 在线 fine-tune**
- 和 #4 配合：Memory MLP 权重在 sleep 时更新
- 需要 replay buffer / sanity check / 灾难遗忘防护
- Sleep-style 批量 fine-tune 避免每轮梯度

**6. 心跳 / 主动表达**
- 空输入 + 非零 W → 模型自发生成
- 需要专门的 heartbeat turn kind 训练
- 哲学问题：AI 应不应该有主动表达权
- Memory MLP 存在后更有意义（即使 W 清空，权重里仍有长期记忆）

### 远期：哲学 / 人格

**6. 灵魂分化验证**
- 同起点 .pt → 不同用户 → 不同对话轨迹 → 权重 diff 显著
- 测试：对同问题两个 `.pt` 回答风格 / 具体记忆明显不同
- 需要在线 fine-tune 先实现

**7. 元认知涌现**
- 模型能感知自身 W 状态并用语言描述（"我记不太清了"、"这个不确定"）
- 自省能力从 W 的内部结构自然长出？
- 探索性，没明确路径

**8. 消融实验**
- Delta Rule vs softmax attention state
- Per-layer q/o_proj vs 共享投影
- `value: list[str]` 加权 vs 均匀权重
- 各 retention pattern 的边际贡献

---

## 已停止的方向（legacy，保留在代码库作参考）

| 方向 | 停止原因 |
|---|---|
| State-as-tokens（v1） | LoRA 全局共享瓶颈 |
| 对称 cross-attention（v2） | 同类消歧瓶颈 |
| Slot routing（v3/v4） | slot 身份伪概念 |
| Contrastive value head（v5b） | hash 碰撞无解 |
| Think 课程 | TELL 66% 失败，4b backbone 自带 |
| 13-stage memory curriculum | 窄分布，LoRA 漂 |
| 基座迁移（migrate_*） | 暂未在 v5c 验证 |

**注**：Memory MLP + Sleep 不是 retired —— 是**未来工作**（见 "中期：长期记忆固化"）。当前 W 是短期记忆，MLP + Sleep 是规划中的长期记忆层。

---

## 硬件需求

| 场景 | 显存 | 时间估算 |
|---|---|---|
| 0.8b 训练（bootstrap + main） | ~10-12 GB | ~2-3 小时 |
| 4b 训练（bootstrap + main） | ~18-22 GB | ~4-6 小时 |
| 4b 单 stage 从 scratch | ~18-22 GB | ~8-10 小时（无 bootstrap 加成） |
| 聊天验证 | ~2-3 GB (0.8b) / ~10 GB (4b) | 实时 |
| chat_smoke 批量评测 | 同上 | ~5-10 分钟 |

16GB+ GPU 足够跑 0.8b，4b 建议 24GB。训练全部用 bf16 + gradient_checkpointing。
