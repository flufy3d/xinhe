# Xinhe (心核) — 架构设计

## 核心假设

智能可以从"持续状态 + 可塑性 + 多时间尺度"这种统一动力系统中自然涌现。

不做 RAG、不做模块拼装、不扩大 context window。用最小结构（小 transformer + 持久状态向量），通过持续训练让系统内部自发分化出记忆、抽象、快慢变量。

---

## 整体架构

```
┌──────────────────────────────────────────────────┐
│                  XinheModel                       │
│                                                   │
│   输入: [R_1..R_n | X_1..X_T | W_1..W_n]         │
│          读状态      内容token    写状态            │
│              ↓                                    │
│   ┌─────────────────────────────────┐             │
│   │       StatePlugin (~3M)         │  ← 可训练    │
│   │  state_emb   状态初始嵌入        │             │
│   │  read_pos    读状态位置编码      │             │
│   │  write_emb   写状态查询嵌入      │             │
│   │  write_pos   写状态位置编码      │             │
│   │  state_scale 渐进影响力          │             │
│   │  gate_bias   快慢区静态偏置      │             │
│   │  gate_proj   动态门控网络        │             │
│   │  state_out   输出投影            │             │
│   └─────────────────────────────────┘             │
│              ↓ inject (拼接读/写状态)               │
│   ┌─────────────────────────────────┐             │
│   │   Backbone (可切换)              │  ← 冻结+LoRA │
│   │  Qwen3-0.6B / MiniMind 64M 等   │             │
│   │  LoRA 注入 q_proj / v_proj      │             │
│   │  标准因果 attention (无自定义mask)│             │
│   └─────────────────────────────────┘             │
│              ↓ extract_and_update                  │
│   输出: logits (content部分) + state_next (写部分)  │
└──────────────────────────────────────────────────┘
```

### 插件化设计

StatePlugin 完全独立于 backbone。backbone 只需实现 `BackboneBase` 接口：

```python
class BackboneBase(ABC):
    def embed(self, input_ids) -> Tensor:         # token → 嵌入
    def forward_blocks(self, x, mask) -> Tensor:  # transformer 层
    def get_lm_head(self) -> nn.Module:           # 输出头
    def get_hidden_size(self) -> int:              # 隐藏维度
```

已实现的 backbone：
- `QwenBackbone`：Qwen3-0.6B (600M)，当前主力
- `MiniMindBackbone`：64M 参数，早期流程验证用

切换 backbone 只需改 yaml 配置，StatePlugin 代码不变。

---

## State-as-Tokens（读写分离架构）

持久状态实现为额外的 token，复用 transformer 的 self-attention 作为读写机制。关键设计：**读状态放在序列开头，写状态放在序列末尾**，利用因果 attention 的方向性天然防止信息泄漏。

### 序列结构

```
[Read-State(旧) | Content | Write-State(新)]
  pos 0..31       pos 32..T+31   pos T+32..T+63
```

### 前向传播

```
输入:  [R_1..R_n | X_1..X_T | W_1..W_n]   读状态 + 内容 + 写状态
         ↓        transformer (因果 attention)        ↓
输出:  [R'_1..R'_n | Y_1..Y_T | W'_1..W'_n]

logits = lm_head(Y_1..Y_T)                          只用内容部分的输出
状态更新: state_next = gate * state_old + (1-gate) * proj(W')   用写状态的输出
```

### Attention Mask（标准因果）

```
               Read列      Content列     Write列
Read行:      [ 因果自身    | 不可见      | 不可见    ]  ← 只携带旧记忆
Content行:   [ 全可见      | 因果遮蔽    | 不可见    ]  ← 从旧state读 + 因果看content
Write行:     [ 全可见      | 全可见      | 因果自身  ]  ← 吸收所有信息，写入新state
```

标准因果 attention，无需自定义 mask。每个位置只能看到自己和之前的位置。

### 为什么读写分离

早期设计中所有 state token 放在序列开头且双向可见。这导致**信息泄漏**：state token 在训练时偷看同一 segment 的答案，模型不需要跨 segment 记忆就能正确预测。读写分离后：

- **Read-State** 在最前面，看不到任何 content → 只能携带上一轮的旧信息
- **Content** 能看到 Read-State → 记忆融入思考过程
- **Write-State** 在最后面，能看到所有 content → 吸收当前信息供下一轮使用

这保证了模型降低 recall loss 的唯一方式是学会跨 segment 使用 state。

### 状态容量

| 配置 | 值 |
|------|-----|
| n_state | 32 tokens (读写各 32) |
| state_dim | 1024 (= hidden_size) |
| 原始大小 | 32 × 1024 = 32,768 floats ≈ 128KB |
| 有效信息 | ~40KB（压缩表示，非原文） |

固定容量是特性，不是缺陷——迫使系统学会选择性记忆和抽象压缩。

---

## 双层 Gate：多时间尺度

类比大脑：脑区分化是固定的（海马=快，皮层=慢），但区域内"记什么忘什么"是内容决定的。

```python
# 层1 — 静态偏置：维度天生的快慢倾向（类似脑区分化）
self.gate_bias = nn.Parameter(torch.zeros(n_state, state_dim))

# 层2 — 动态投影：根据内容决定此刻该记还是该忘
self.gate_proj = nn.Linear(2 * state_dim, state_dim)

# 合并
dynamic_logit = self.gate_proj(cat[state_old, state_new])
gate = sigmoid(self.gate_bias + dynamic_logit)

state_next = gate * state_old + (1 - gate) * state_new
```

- `gate_bias` 大 → 该维度天生偏慢（长期存储区），但极端内容仍可覆写
- `gate_bias` 小 → 该维度天生偏快（工作记忆区），但内容仍有选择权
- 两层合一个 sigmoid，只多一个线性层，无额外架构复杂度

---

## Sleep 机制：记忆固化 + 在线学习

类比人类睡眠：白天积累经验（state），睡眠时突触重塑（权重更新）。

### 两阶段运行模式

```
白天（推理）：                         Sleep（学习）：
  权重完全冻结                           Memory LoRA + Skill LoRA 更新
  只有 state 在流动                      回放 (state_in, content, labels) 序列
  全速生成，无额外开销                    逐步弱化 state 迫使记忆转移到权重
  保存 (state_in, content, labels) 到 buffer
```

### 双 LoRA 架构

Transformer 内部有功能分离：Attention 层控制信息路由，MLP 层存储事实知识。

```
Transformer Layer:
├── Attention
│   ├── q_proj + Skill LoRA    ← 教 attention 怎么读写 state
│   └── v_proj + Skill LoRA
├── MLP
│   ├── up_proj + Memory LoRA   ← 存储长期记忆（零初始化，sleep 时更新）
│   ├── down_proj + Memory LoRA
│   └── gate_proj + Memory LoRA
```

Memory LoRA 零初始化 → 加上去不影响任何现有行为，直到 sleep 更新它。

### Sleep 流程

1. 回放保存的 (state_in, content, labels) 序列
2. 逐步弱化 state 输入（100% → 50% → 0%）
3. 标准交叉熵 loss，模型必须靠权重而非 state 答对
4. 分层学习率更新：

```
Memory LoRA (MLP):      lr = 1e-4   快速学习记忆
Skill LoRA (attention):  lr = 1e-6   极缓慢演化
StatePlugin:             lr = 0      完全冻结
```

### 醒来后的效果

Memory LoRA 通过残差连接**叠加**到 backbone 输出上（加法，非替换）：
- 新事实（未 sleep）：靠 state，和以前一样
- Sleep 过的事实：state + 权重双通道增强
- State 清空后：权重兜底，仍能回答

### 记忆系统对应

```
人脑                         Xinhe
──────────                   ─────
神经激活（工作记忆）           state（当前对话的工作台）
突触连接（长期记忆）           LoRA 权重（sleep 后积累的知识）
睡眠时海马体→皮层转移          sleep 时弱化 state → 迫使权重学习
```

---

## .pt = AI 的灵魂

```python
checkpoint = {
    "plugin_state_dict": ...,        # StatePlugin（gate/scale/state_emb）
    "skill_lora_state": ...,         # Skill LoRA（attention 层，读写技能）
    "memory_lora_state": ...,        # Memory LoRA（MLP 层，长期记忆）
    "state": ...,                    # 当前 state（工作记忆）
    "replay_buffer": ...,            # 待 sleep 的对话序列
    "sleep_count": ...,              # AI 的"年龄"
}
```

每个用户的 AI 从同一个预训练起点出发，经历不同的对话和 sleep 周期后 .pt 逐渐分化。不同的 .pt 就是不同的"人"。

---

## 安全启动（不破坏聊天能力）

三层保护，确保加入 StatePlugin 不破坏 backbone 原有能力：

1. **LoRA 零初始化**：B 矩阵全零 → 训练开始时 backbone 行为 = 原始模型
2. **状态嵌入近零初始化**：read/write token 的位置编码和初始嵌入都近零 (std=0.01)
3. **渐进影响力 state_scale**：控制 state token 值的缩放，从小到大

读写分离架构的额外优势：标准因果 attention，backbone 不需要处理任何自定义 mask，降低了启动干扰。

---

## Burn-in：用 prompt 初始化 persona

```python
def burn_in(self, token_ids_list):
    state = self.init_state(batch_size=1)
    with torch.no_grad():
        for seg in token_ids_list:
            result = self.forward(seg, state)
            state = result["state_next"]
    return state
```

不同 system prompt → 不同初始状态 → 不同 persona。老用户回来 → 从 .pt 恢复。

---

## 训练

### 课程学习（Curriculum Learning）

State 机制无法一步到位训练——同时学"怎么用 state"和"什么时候用"会陷入死锁（scale 持续下降，模型放弃 state）。必须分阶段递增难度：

```
阶段 1: 固定 tell + d=1        → 学会 state 基本读写
阶段 2: 随机 tell 位置          → 学会内容级写入判断
阶段 3: distance 1-3           → 学会跨距离保持
阶段 4: distance 1-10 + 多 filler → 完整多轮记忆
阶段 5+: 多事实 / 覆写          → 区分类别 / 更新能力
```

课程配置定义在单个 YAML 中，一条命令自动执行所有阶段。详见 `docs/curriculum_learning.md`。

| 项目 | 值 |
|------|-----|
| 训练对象 | StatePlugin (~3M) + Skill LoRA (注入 q/v_proj) |
| 冻结 | Backbone 全部原始权重 |
| 数据 | 合成记忆对话，按课程阶段自动生成 |
| 状态传递 | state 跨 segment 持续传递 |
| 梯度截断 | 每 tbptt_steps 个 segment 做 detach + backward |
| 收敛检测 | EMA loss 低于阈值自动进入下一阶段 |

### Sleep 阶段

| 项目 | 值 |
|------|-----|
| 触发 | 用户手动 `/sleep` 或定时触发 |
| 数据 | 对话时保存的 (state_in, content, labels) 序列回放 |
| State 弱化 | 逐步 100% → 50% → 0%，迫使记忆转移到权重 |
| 更新对象 | Memory LoRA (MLP, lr=1e-4) + Skill LoRA (attention, lr=1e-6) |
| 冻结 | StatePlugin（保护 gate/scale） |
| 安全措施 | sleep 前保存 checkpoint，失败可回滚 |

---

## 参数总览

| 参数 | 值 | 说明 |
|------|-----|------|
| n_state | 32 | 状态 token 数（读写各 32） |
| state_dim | 1024 | 状态维度 = backbone hidden_size |
| state_scale_init | 0.0 | 初始影响力 sigmoid(0)=0.5 |
| Skill LoRA rank | 4 | 注入 q_proj, v_proj |
| Memory LoRA rank | 4 | 注入 up/down/gate_proj（sleep 时更新） |
| lora_alpha | 8 | LoRA 缩放 |
| tbptt_steps | 可变 | 课程学习中随 episode_length 调整 |
| sleep memory_lr | 1e-4 | Memory LoRA 学习率 |
| sleep skill_lr | 1e-6 | Skill LoRA 学习率 |
