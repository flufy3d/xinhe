# W_turn 多相位共振架构失败复盘（2026-04-24）

本文档记录 v6 双流架构里 W_turn 读侧的三次迭代（v1 / v4 / v4.1）全部失败的经验，供后续"大调架构"时不要再踩相同的坑。

---

## 架构回顾

双流持久状态 `DualState(W_fact, W_turn)`：
- **W_fact**：Delta Rule 联想记忆，写侧 β-selective，读侧单点内积 —— fact 0a bootstrap 已收敛到 98%，没问题
- **W_turn**：自旋时序罗盘，写侧每轮 `γ·R·W + mean(v⊗k)`（程序性，无可学参数），读侧"多相位共振搜索"：
  ```
  对 τ ∈ {0..phase_max} 分别正旋转 q_τ = R^τ · q
  并行内积 r_τ = q_τ @ W_turn.T  → (B,H,T,P,d_v)
  score_τ = ||r_τ||_2 / √d_v × K     # K = turn_phase_temperature
  α = softmax(score, dim=phase)
  read = Σ_τ α_τ · r_τ
  out = hidden + sigmoid(read_scale_turn) · o_proj(read)
  ```

设计直觉：RoPE 正交性保证 `<R^τ q, R^age k>` 在 τ=age 时最大，**不用"预测"age，直接"搜索并挑赢家"**。

---

## 三次迭代的失败轨迹

### v1（100% verbatim_recall，freeze_lora=true）

**数据**：3-6 轮 "记住这句: <8-12字随机短语>" → distractor × 0-2 → "刚才那句是？" 原样复述。

**结果**：VALUE token 级 62%，其中字符间空格贡献 ~50%，实际 alnum 字符召回 ~18%。softmax 塌平、phase_ent ≈ log(P+1) ≈ 均匀、turn_scale 反跌。

**根因**：**单条目 W_turn 里任何 τ 读出的都是同一条内容在不同旋转下的投影**。score 差距完全依赖 `⟨q, R^(a-τ) k⟩` 衰减曲线，q_proj_turn 随机初值下曲线平 → 无"选错就答错"的压力 → α 永远均匀。

### v4（60% 对抗集 + 20% verbatim + 20% meta_recall，freeze_lora=false）

**改动**：加入时序碰撞对抗集（3 条 random alnum phrase + 按位置/距离指定查询其中一条），强制 phase 选择才能答对。放开 LoRA 让 backbone 适应 turn 输入。

**结果**（step 3000 val VALUE=24.91%）：
```
step 500:  phase_ent=1.725  turn_scale=0.0436
step 1500: phase_ent=1.505  turn_scale=0.0392  ← 最低点
step 2350: phase_ent=1.67   turn_scale=0.0381  ← 冻结
step 3000: phase_ent=1.637  turn_scale=0.0381  ← 回退且冻结
```

**根因**：
1. **LoRA 放开给了 backbone 捷径**。对抗集里 60-70% 的 query 是"最后那句"（dtau=0/1）attention 看得到；"最早那句"（dtau=4/5）attention 看不到。backbone 通过适应前者的模式拉平均 loss。
2. **turn_scale 被压扁自锁**。错 phase 选择 → 读出是噪音 → 梯度压 scale → scale 小 → 读出对 loss 影响小 → 无法通过 loss 推 phase 学习 → α 保持均匀 → 读出还是噪音。**chicken-and-egg 死锁**。

### v4.1（K=30, turn_lr_multiplier=2.0，其他同 v4）

**动机**：假设 v4 的 K=10 softmax 温度不够，尝试把锐化力度加 3 倍 + turn 读侧 LR 翻倍，撬开对称性。

**结果**（step 4000 val VALUE=46.7%）：
```
step 500:  phase_ent=1.388  turn_scale=0.0403  (loss 初期更差，K 锐化的代价)
step 1000: phase_ent=1.349  turn_scale=0.0349  (反超 v4：loss 2.44 vs 2.58)
step 2000: phase_ent=1.185  turn_scale=0.0290  (loss 1.83)
step 3000: phase_ent=1.355  turn_scale=0.0269  (loss 0.73，但 phase_ent 反弹)
step 4000: phase_ent=1.358  turn_scale=0.0266  (loss 0.59，val VALUE=46.7%)
```

**表面看**：VALUE 从 v4 的 24.91% 翻倍到 46.70%，看起来在收敛。

**实际**：**phase_mode = 2.99 ≈ dtau 均值 2.5**，即 argmax 相位分布和数据里 target_dtau 的边缘分布一致。argmax **没有条件化到具体 query**，是群体均值 —— **没有真正学到相位选择**。loss 下降全靠 LoRA+backbone 吸收"最后那句 + 中间那句"的易 query，hard queries（最早那句/N=5 轮前）依然全错，只是数量不够拉垮平均分。

**turn_scale 继续被压扁到 0.0266**（sigmoid(-3.6)，实际贡献 ~2.7%）印证了这点 —— 模型判定 turn 读侧是噪音。

**stop 在 step 4000** 因为判断再训 2000 步也只能爬到 VALUE ~60%、phase_mode 仍在均值附近，机制没启用而是被绕过，喂给 Stage 1 大概率被 refusal/pronoun 梯度覆盖 W_turn 读侧（v6.2 实测 chat_smoke G=0/3 H=0/2 的复现）。

---

## 核心失败模式

### 1. Multi-phase softmax 选择需要"已对齐"的 q/k 投影

读侧 softmax 挑赢家的前提是 **score 分布有一个显著峰**。但训练初期：
- `q_projs_turn` 随机初值
- `k_proj_turn`、`v_proj_turn` 随机初值（v6.2 加的独立投影，0a ckpt 不包含）
- `||r_τ||` 各 τ 基本同阶
- softmax 输出接近均匀 → gradient 到每个 score_τ 都很小 → 学习信号被稀释到 P+1 份

**K=30 治标不治本**：加大 K 让"刚好最大的那个"权重更高，但在 score 没真区分度时，选的那个是 **随机**。反而把梯度信号集中砸在错相位上，加快收敛到局部最优（phase_mode = 均值）。

### 2. turn_scale 的 sigmoid(-3) 初值让读路径在训练初期就是"信号死区"

`sigmoid(-3) ≈ 0.047` 意味着 `out = hidden + 0.047 · turn_read`。即使 turn_read 完全对，对 output 的贡献也只有 5%；对 loss 的影响比 LoRA+backbone 小一个量级。梯度 `∂L/∂(read_scale_turn)` 在早期主要来自"turn_read 是噪音"的惩罚 → scale 被压得更小 → turn 路径进一步失效。

### 3. "有捷径就走捷径"定律

`freeze_lora=false` 让 backbone + LoRA 能学会通过 attention 在当前 segment 内解决 60-70% 的 query（latest/middle 系 dtau=0/1）。于是：
- 这类 query loss 下降 → 平均 loss 下降 → val VALUE 上升
- 这类 query 不需要 phase 选择 → 不提供 phase 选择的监督信号
- 剩下 30-40% 的 hard query（earliest）需要 phase 选择，但提供的梯度占比低，斗不过 LoRA 方向的梯度
- 最终：**backbone 绕过 W_turn 解决容易的事，难的事直接放弃；W_turn 读路径永远学不会**

### 4. v1 vs v4 的 freeze_lora 对比

| 版本 | LoRA | 数据 | 失败模式 |
|---|---|---|---|
| v1 | freeze | 单条目 verbatim | W_turn 是唯一通路，但单条目不逼 phase 选择 → flat α 也能读 → 学到的是"不管 α 读什么一样" |
| v4 | open  | 多条目对抗集 | 数据逼 phase 选择，但 LoRA 提供捷径 → 模型走捷径 |

**两者的共同点**：都缺少"必须学会 phase 选择才能活"的训练信号闭环。

---

## 对下一版架构设计的启示

**核心问题是 score-based soft selection + 低初始读强度 + 有捷径的组合拳。** 任何新架构应该破掉其中至少两项：

1. **去掉"挑赢家"的 softmax，改成"已知答案的预测 + 监督"**。比如加一个 Δτ head（先前 dtau_head 设计）直接预测 target_dtau，用 `target_dtau` 字段做 supervised loss。风险：dtau_head 曾经塌成数据均值。但新版可以：
   - 输入不只 recall turn 的 hidden，还 cross-attend 历史轮的 key 来选
   - 或 hard-argmax + 直通梯度 `StraightThroughEstimator`

2. **读路径初始就"压过" hidden**。turn_read_scale_init=0（sigmoid=0.5）或更高。读错的代价在 day 0 就全量作用于 loss，模型不能假装它不存在。

3. **关掉所有捷径**。0b 整段 freeze_lora=true，甚至考虑 attention mask 屏蔽 recall turn 对历史轮的 attention（强制只能通过 memory 读）。

4. **用"程序性正确性可验证"的任务而非"token-level loss"**。比如构造：给定 3 个 phrase + 一个明确 dtau，恰好只有一个 phrase 匹配。用 MRR（平均倒数排名）或 Recall@1 作为 hard metric，直接优化；token loss 作为辅助。

5. **别迷信"端到端学习 phase 选择"**。写侧程序性（γ·R 每轮自转）是对的；读侧强行"自己学会挑哪个 τ"是过度泛化。可以考虑：
   - Δτ = 当前 hidden 投影 + recall prompt cross attend 得到的一个标量 / logit
   - 或直接让 writer 在写时留一个 episodic pointer，reader 按 pointer 读

---

## 事后诊断指标使用经验

训练时下列指标能**快速辨别"真学会"vs"走捷径"**：

| 指标 | 真学会 | 走捷径 |
|---|---|---|
| phase_ent | 显著 < log(P+1) 且单调降 | 波动在 log(P+1) 附近 |
| phase_mode 分布 | 与 data target_dtau 的**条件分布**匹配 | 与 target_dtau 的**边缘**匹配（即 ≈ 均值） |
| turn_read_scale (sigmoid) | 从 init 持续爬升 | 冻结或下降 |
| val VALUE on hard subset (dtau≥3) | 随训练上升 | 恒定低 |

`phase_mode == 数据均值` 是比 `phase_ent 卡住` 更确定的失败判据 —— 前者说明"argmax 没条件化到 query"，架构**定性**失败。下次建议每次 val breakdown 把 VALUE 按 target_dtau 分桶打印。

---

## 清理范围

本次失败后清理的废弃内容：
- `configs/curriculum{,_migrate,_think,_qwen3.5-*}.yaml`、`configs/migrate_*.yaml`、`configs/think_*.yaml` — 11 个 legacy config
- `xinhe/data/generate_think_data.py` / `xinhe/data/think_lang.py` — think 课程代码
- `scripts/debug_value_errors.py` / `scripts/diagnose_turn.py` — 针对旧/失败架构的诊断工具
- `generate_memory_data.py` / `chat.py` / `config.py` / `generate_data.py` 里的 think 参数和分发

W_turn 读侧核心代码（`xinhe/model/turn_plugin.py`、`xinhe/model/dual_state.py`、对抗集数据生成器）**保留**，作为下次大调架构时的参考基线。

---

**作者**：James Chen / Claude
**日期**：2026-04-24
