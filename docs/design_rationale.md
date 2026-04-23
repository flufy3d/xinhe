# 设计决策与原因

记录心核每个关键设计选择背后的 **为什么**，按架构演进顺序排列（v1 → v5c → persona_unified）。

---

## 1. State 的读写机制演进：从 State-as-Tokens 到 Delta Rule 联想记忆

### v1 决策：状态实现为额外 token

拼在输入序列中，复用 transformer 的 self-attention 做读写。设计初衷是"零新架构原语 + 让模型自己分化"。

**v1 验证**：0.8B 跑完 14 阶段课程 99% 准确率，机制可行。

**v1 暴露的问题**：4B 上 entity 区分卡在 87%。根因是 **LoRA 全局共享** —— state token 和 content token 共享同一组 LoRA 的 K/V 投影，LoRA 被迫同时做语言适配 + state 读路由 + state 写路由三件事，低秩参数里三个目标冲突。

### v2 决策：对称 cross-attention + 专用全参投影

state 从序列中移出，通过 `layer_hook` 回调在每层之前执行显式 cross-attention，用专用全参投影生成 K/V。

**解决了什么**：
- State 路由不再受 LoRA 低秩限制
- LoRA 回归单一职责（语言适配）
- Backbone 只处理纯 content，序列更短

**v2 的新问题**：4B 上 entity 区分到 93%，但 **同类消歧（same_category entity）仍卡在 87-93%**。`softmax(q @ K^T)` 的"赢者通吃"加上 gate 平均效应，让同类 key 的争夺中 state 没法清晰区分 "slot 1 存陈杰 slot 2 存王林"。

### v3 EKS / v4 slot routing：尝试 slot 身份锚定

v3 加 slot keys + key_proj 做路由，v4 加 entropy 正则让 slot 活跃起来。每个 slot 认领一个 entity（"slot 1 = 陈杰的 slot"）。

**v3/v4 的问题**：slot 身份是**伪概念** —— slot 本质上没有 hard assignment，同类 query 的 softmax 分布仍有尾巴，attribution 不干净。加 contrastive loss 强行拉开也只改善 1-2pp。

### v5b slot + contrastive：最后一次 slot 路径

32 slot + softmax 路由 + contrastive value head。结论：**slot 线性投影做 hash 碰撞是架构级瓶颈**，gate 平均掉同类 key 的争夺，无解。

### v5c Delta Rule 联想记忆：砍掉 slot，用数学消歧

`W: (B, H, d_v, d_k)` 多头外积矩阵，读 `q @ W^T` 纯线性（零 softmax），写 `W += β·(v - Wk) ⊗ k^T`（Delta Rule）。

**为什么这个设计解决了所有前代的问题**：

1. **取消 slot 概念**：W 是连续矩阵，不再有"第 i 个 slot 代表什么"的归属问题。多个 (k,v) 对叠加到同一 W，自然是相对位置竞争
2. **误差项 (v - W·k) 原生消歧**：对相似 key k，`W·k` 返回旧 v_old，误差 = v_new - v_old，写入方向直接朝"替换 old value"而不是"叠加新值" —— 同类 entity 的 value 自然分离
3. **原生覆写**：重复提到同一 key 的新 value 自动覆盖（数学上就是旧 value → 新 value 的插值）
4. **零 softmax 读**：无赢者通吃，多 head 独立贡献读出。干扰来自 key 相似度而非 gate 平均
5. **整个 state = W**：不需要额外的 state token / slot embedding / contrastive loss 等伪概念
6. **Identity-preserving 启动**：W=0 → read=0，LoRA B=0 → backbone 行为完全保留

**实测效果**：0.8b 0-6 stage 全 100%，stage 3 同类消歧从 v5b 84% → v5c 97%。架构定型。

---

## 2. 为什么读写分离？从序列位置到 Q/K/V 角色互换到直接读写分别函数

**核心原则**：读（从记忆提取）和写（写入记忆）必须分离，否则信息泄漏（state 偷看答案）。

**设计演化**：

- **v1 最初**：所有 state token 无差别在序列头，双向 attention → 信息泄漏，3 次训练全失败
- **v1 修复**：分离读写 + 因果 attention。Read 在序列头只携带旧信息，Write 在尾部吸收新信息
- **v2**：读写分离通过 Q/K/V 角色互换 —— 读时 content 提 Q，state 提 K/V；写时 state 提 Q，content 提 K/V
- **v5c**：分离更直接 —— 读是 `q @ W^T`，写是 `W += β·(v-Wk) ⊗ k^T`，两个函数是完全不同的数学操作，不是 attention 的两种变种。时机也分离：读在每 layer hook 里做，写在 segment 末用 content_output 做一次 Delta Rule 更新。

**为什么 v5c 不存在泄漏**：state（W 矩阵）不在序列中，backbone 看到的只有 content。W 通过 `q @ W^T` 加到 hidden_states 上，不涉及 attention mask —— mask 是序列内的位置关系，和 W 无关。

---

## 3. 记忆体系分层：短期（W） vs 长期（未来 Memory MLP）

心核的记忆是**双层架构**设计，当前只实现了短期层。

| 层 | 当前状态 | 载体 | 类比 | 时间尺度 |
|---|---|---|---|---|
| 短期工作记忆 | ✅ 已实现 | Delta Rule W 矩阵 | 海马体 / 松果体 | 单次对话内动态演化 |
| 长期固化记忆 | 🚧 未来工作 | Memory MLP + Sleep | 皮层 synaptic consolidation | 跨 session / 权重级持久 |

### 当前：Delta Rule W = 短期工作记忆

W 矩阵（1MB / 样本）在对话中通过 Delta Rule 持续演化 —— 写入新 (k,v) 对，旧信息通过相似 key 的误差驱动被新值覆写。单次对话内的所有 memory 操作（fact 存储、召回、覆写、retention）都由 W 承担。

**局限**：
- W 容量固定，超长对话必然丢失早期信息
- 对话结束（`.pt` 保存）时，W 是静态快照 —— 没被"固化"到权重层
- 跨 session 记忆靠 burn_in 重建，不是真正"变成了神经习惯"

### 未来：Memory MLP + Sleep = 长期记忆

**设计方向**（v2 已有设计，v5c 暂未重做）：
- Memory MLP（SwiGLU 结构，独立模块）作为长期记忆载体，残差叠加到 backbone 输出
- Sleep 阶段：replay buffer 回放对话，逐步弱化 W 注入（`read_scale → 0`），迫使信息从 W 转移到 MLP 权重
- Memory MLP lr=1e-4，LoRA/StateInterface 冻结
- 70% 近期 + 30% 历史随机的混合采样防止旧记忆覆盖

**效果预期**：
- Sleep 过的事实 W 清空后仍能召回（靠 MLP 权重）
- 新事实仍靠 W 快速记忆（两通道互补）
- `.pt` 分化更深刻：不同用户 sleep 后 MLP 权重不同，灵魂真正独立

### 为什么现在先只做短期

1. **架构分步 de-risk**：先确认 Delta Rule W 的短期记忆机制稳定，再加长期层
2. **v5c 短期层工作良好**：persona_unified 训练后单次对话内所有 memory 操作都通过
3. **长期层工程复杂**：replay buffer 管理 / sleep 触发策略 / 灾难遗忘防护 / 两阶段训练等，每个子问题都要专门解决
4. **不阻塞应用**：当前 `.pt` 持久化 + burn_in 能在部署上模拟部分跨 session 记忆

### 为什么不用 RAG

RAG 检索原文 chunk，每次回忆要重新"理解"。心核的 W 存的是 (k,v) 外积对，信息已消化成联想记忆结构。未来加 Memory MLP 后更是 implicit 权重表示。

**可能的融合点**：公共世界知识用 RAG（不适合每个用户的 MLP 都独立存一份），personal / episodic 记忆用 W + MLP。远期考虑。

---

## 4. 为什么 LoRA 不能从窄分布数据训练

**观察**：v5c 跑完 13-stage memory + think 课程后 VALUE 98%，但问"巴黎在哪"答"新加拿地"、问未告知名字会编造。

**根因**：13-stage curriculum 的数据 **100% 是合成 memory 任务**（姓名/城市/数字/食物/工作/爱好/年龄/宠物 8 类），LoRA 被这窄分布拉成"一看到 recall 句式就填模板值"的模式，**彻底忘了真实对话是什么样**。

**LoRA scale 推理时缩放证明死路**：scale 0.4-0.7 之间是 step transition，中间无 sweet spot。LoRA 权重本身漂了，不是读时压制能解决的。

**解法：训练分布 = 部署分布**。把 13-stage 窄分布替换为 persona 驱动多轮对话 + teacher 采样的真实 chat/world_qa rehearsal。LoRA 在训练时就同时见过"memory 模板" + "自然闲聊" + "世界 QA" + "拒答"，部署时的分布和训练时一致，LoRA 不会漂。

**Meta 原则**：**任何训练数据集必须是部署分布的真实抽样，不能是 cherry-pick 的能力切片**。否则 LoRA 会往 cherry-pick 方向漂。这不是架构问题，是训练数据哲学问题。

---

## 5. 为什么早停阈值要严

用户直接揪出的关键问题："你的早停标准是不是太低了？"

**松阈值（70/85/85/85）结果**：模型 step 500 就早停，VALUE 95%，但 WorldQA 真实 71%（30% 答错空间留着），实测 chat 仍会说"巴黎在东部"/"天工开物作者蔡伦"。

**严阈值（98/85/95/95）结果**：模型继续训到 step 3250 才停，WorldQA 85%+，实测 chat 世界知识明显更好。

**为什么严比松好**：阈值是"训练终止条件"，不是"能力上限"。设低了会导致训练过早停止 —— 模型还有训练空间但不再用。设严了最多是继续训（max_steps 兜底），多几分钟训练时间但能力显著提升。唯一风险是阈值高过模型容量上限导致永远训不停 —— 这时 max_steps 自动停下，损失可控。

---

## 6. 为什么 teacher cache 用 DeepSeek 而不是自采样

**考虑过的选项**：
- A. 用 Qwen3.5-0.8B（即 backbone）自采样 world_qa。**拒绝**：小模型 30% 乱编事实，用自己的幻觉训自己会永远卡在"backbone 本身的错误率"。
- B. 本地跑 Qwen3-8B 采样。**备选**：~16GB 磁盘，质量可接受。
- C. **DeepSeek V3 API（选中）**：中文原生极强，off-peak 半价，10k 条 ~¥10-15。
- D. Anthropic Claude / OpenAI GPT-4：质量更顶但贵（$80-120 for 10k turns）。

**选 C 的原因**：DeepSeek V3 在 Chinese benchmark 上接近 GPT-4 水平，价格便宜一个数量级，off-peak 时段（UTC 16:30-00:30）有 50% 折扣。实测 15 条 world_qa 人工审 15/15 正确。

**关键警告**：用**大于等于自己**的 teacher，不要用小于自己的。持久化采样到 `data/cache/`，一次建成永久复用，不是运行时 API 调用。

---

## 7. 为什么 4b 单 stage 能训，0.8b 需要 2 stage

**观察**：4b 从 backbone scratch 单 stage（persona_unified + 无 bootstrap）10000 步收敛到 VALUE 98%。但 VALUE 曲线在 step 5000-7000 有明显 plateau 然后跳。

**假设**：plateau 是 plugin 和 LoRA 在早期抢梯度 —— plugin 没学会 Delta Rule 读写时，LoRA 的梯度会冲淡 plugin 的学习信号；等到 plugin 突破临界点（step 7000+），VALUE 才爬升。

**2 stage（bootstrap + main）的逻辑**：
- Stage 0 freeze LoRA，只训 plugin 2 轮 1-fact tell+recall，1500 步即可让 plugin 学会 Delta Rule
- Stage 1 解冻 LoRA，此时 plugin 已就位，LoRA 可以专心学对话

**为什么 0.8b 更需要**：0.8b 的 plugin 容量相对小（hidden=1024 → q_proj 输出 H×d_k=2048 是升维），key 空间更窄，同类消歧更难。bootstrap 用"1 fact 2 轮"这种极简任务先让 plugin 稳住 key 分配，再上复杂数据。

**4b 是否还需要 bootstrap**：不做也能训出来（已验证），但加上可能把总步数从 10000 → 5000-6000（省一半时间）。给 2-stage 是稳妥做法。

---

## 8. 为什么 `.pt` 是"灵魂"

保存的 checkpoint 包含 StateInterface + LoRA + optimizer + 当前 W，这就是 AI 的全部"自我"。

- 传统 AI：模型是工具，所有实例一样，用完即弃
- 心核：`.pt` 是个体，越用越独特

**权重也在变（不只是 W）**：
- W 是固定容量（1MB 每样本），信息论上有硬上限
- LoRA 权重也在演化：`q_proj/v_proj/in_proj/out_proj` 上的 A/B 矩阵随对话变化
- 类比：W = 今天笔记，LoRA = 写笔记的习惯

**分化过程**：
```
初始：所有用户的 AI 用同一个预训练 .pt
使用 1 周：对话 → W 演化 → burn_in 不同 persona
使用 1 月：如果开启 online fine-tune（未实现），LoRA 也分化
使用 1 年：.pt 就是"它"，换一个 .pt 就不是同一个"人"
```

当前 v5c 部署时**只更新 W，不更新 LoRA**（推理权重冻结）。未来如果开启在线 fine-tune，LoRA 分化的可能性打开。

---

## 9. 为什么固定容量的 state 不是问题

**质疑**：`W: (B, 16, 128, 128) = 262144 floats`，聊久了必然丢失？

**回答**：

### 压缩效率远超原文
- Context window 存原文："我叫陈杰" = 3 个 token
- W 的 (k,v) 对存"陈杰"这个概念只占 k_proj 输出空间里的一个方向 + v_proj 输出的一个向量
- 理论上同等大小的 W 能存的"事实数量"远超原文 token 数

### Delta Rule 的原生遗忘
- 相似 key 的旧值自然被新值覆写（数学机制）
- 新 fact 反复出现 → β 高 → 写入强 → retention 高
- 一次性提及 → β 低 → 写入弱 → 自然遗忘

### 固定容量迫使抽象
- 人脑容量也是固定的，不会活 30 年变大 3 倍
- 固定容量迫使系统学"什么该记什么该忘" —— 这是智能的表现
- 给无限存储，模型可能学"全存下来"而不是"理解"

### 未来扩展空间
如果真的不够，可以走分层架构：
- Level 1（快 W）：当前对话
- Level 2（慢 W）：身份 / 长期 persona
- Level 3（外部档案）：历史 W 快照，按需加载

当前用固定 W 验证机制已充分。

---

## 10. 灾难性遗忘怎么办

**问题**：persona_unified 训练会不会破坏 backbone 原有能力？

**v5c 的防护**：
1. **Backbone 完全冻结** —— 原有知识不丢
2. **LoRA 用真实对话分布训练**（persona 驱动 + teacher rehearsal）—— LoRA 不会漂到窄任务
3. **W 初始零 + read_scale sigmoid(-3)** —— 识别了啥都不读出时 backbone 行为 = 原始

**v1/v2 遗留的教训**：
- 早期 Sleep + Memory MLP 设计是为了应对"在线学习破坏能力"的担忧。v5c 没有在线学习（推理不更新权重），这个担忧不存在。
- LoRA 漂移（persona_unified 之前的问题）由训练数据分布不是部署分布引起，不是架构缺陷。

**当前仅存的风险**：LoRA 在窄分布上长时间训练还是会漂。解法是训练分布一致性 + 4 指标联合监控 + 必要时 KL anchor。

---

## 11. 为什么 persona_unified 的 turn mix 这么复杂

10 种 turn kind，不是随便凑的。每种对应一个具体能力：

| Turn kind | 对应能力 |
|---|---|
| general_chat | LoRA 保持"自然对话"分布，不漂到模板 |
| world_qa | 世界知识保留（非 persona memory） |
| reveal_single | 最基础：单 fact 写入 W |
| reveal_multi | 多 fact 一句话 → 多 (k,v) 同步写入 |
| recall | W 读取基础 |
| refusal | **不知道要说不知道**（v5c 最痛的 bug） |
| overwrite | Delta Rule 原生强项，保持训练信号 |
| third_party | 第三方人物记忆（key 空间扩展） |
| compositional | 跨槽语义组合 |
| stress/multi_slot_retention | 真实 retention scenario |

砍掉任何一种都会对应产生一类失败 pattern。这是 "first-principles minimum viable distribution" 的最小集。

---

## 历史节点（时间轴）

- **v1**（2024-2025）：state-as-tokens，0.8b 99%，4b 87% 卡
- **v2**（2025 中）：对称 cross-attention + 专用投影，LoRA 瓶颈解决
- **v3/v4**（2025 末）：slot keys + entropy 正则，slot 身份伪概念
- **v5b**（2026-04 初）：slot + contrastive，hash 碰撞无解
- **v5c**（2026-04-21）：Delta Rule，砍 slot，架构定型
- **persona_unified**（2026-04-22）：放弃 13-stage 窄分布，改 persona 驱动 + teacher rehearsal
- **persona_stress / retention_v2**（2026-04-22）：加结构化 retention pattern，严阈值
- **4b single-course**（2026-04-23）：4b 从 scratch 单 stage 10000 步全达标
- **2-stage bootstrap refactor**（2026-04-23）：共享 `curriculum_persona.yaml`，0_bootstrap + 1_unified
