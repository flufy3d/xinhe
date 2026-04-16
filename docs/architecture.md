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
│   state_old                                       │
│     ├── 读侧: state → 专用 K/V 投影 → 注入每层 KV-Cache │
│     │                    ↓                        │
│     │   [Content(T)] → backbone 32 层 → content   │
│     │                  (每层 attend 到 state K/V)  │
│     │                    ↓                        │
│     │                  logits                     │
│     │                    │                        │
│     └── 写侧: state → 专用 Q 投影 → attend to content │
│                                      ↓            │
│                                 state_next        │
│                                                   │
│   ┌─────────────────────────────────┐             │
│   │     StateInterface (~5M)        │  ← 可训练    │
│   │  state_emb    状态初始嵌入       │             │
│   │  read_k/v_projs 每层K/V投影     │             │
│   │  read_scale   渐进影响力         │             │
│   │  write_q      写侧Q投影         │             │
│   │  write_out    写侧输出投影       │             │
│   │  gate_proj    动态门控网络       │             │
│   └─────────────────────────────────┘             │
│              ↓ past_key_values 注入                │
│   ┌─────────────────────────────────┐             │
│   │   Backbone (可切换)              │  ← 冻结+LoRA │
│   │  Qwen3.5-0.8B / Qwen3.5-4B 等   │             │
│   │  LoRA 注入 q_proj / v_proj      │             │
│   │  LoRA 只负责语言适配             │             │
│   └─────────────────────────────────┘             │
│              ↓                                     │
│   输出: logits (content部分) + state_next (写侧更新) │
└──────────────────────────────────────────────────┘
```

### 插件化设计

StateInterface 完全独立于 backbone。backbone 只需实现 `BackboneBase` 接口：

```python
class BackboneBase(ABC):
    def embed(self, input_ids) -> Tensor:                          # token → 嵌入
    def forward_blocks(self, x, mask, pos, past_kv=None) -> Tensor: # transformer 层
    def get_lm_head(self) -> nn.Module:                            # 输出头
    def get_hidden_size(self) -> int:                               # 隐藏维度
```

已实现的 backbone：
- `QwenBackbone`：支持 Qwen3.5 系列 (0.8B / 4B / 9B)

切换 backbone 只需改 yaml 配置，StateInterface 代码不变。

---

## 对称 Cross-Attention（读写分离架构）

持久状态通过 **对称的 cross-attention** 实现读写：读是 Content(Q) × State(K,V)，写是 State(Q) × Content(K,V)。Q 和 K/V 角色互换，机制完全对称。

### 核心对称性

```
读: Content(Q)  ×  State(K,V)  →  记忆融入思考    (每层)
写: State(Q)    ×  Content(K,V) →  信息写入记忆    (最后一层后)
```

| | 读 | 写 |
|---|---|---|
| Q 来自 | content（backbone 内部） | state（专用投影） |
| K/V 来自 | state（专用投影） | content（backbone 输出） |
| 机制 | cross-attention（通过 KV-Cache） | cross-attention（独立模块） |
| 时机 | 每层 | 最后一层输出后 |
| 意义 | 带着记忆思考 | 从思考中提取记忆 |

### 读侧：state → K/V 注入

利用 HuggingFace `past_key_values` API，零侵入 backbone：

```python
# Qwen attention forward（已有代码，不需要修改）
Q = q_proj(hidden_states)    # content 的 Q
K = k_proj(hidden_states)    # content 的 K
V = v_proj(hidden_states)    # content 的 V

if past_key_values is not None:
    K = cat([past_key_values[0], K])   # state K/V 自然拼到前面
    V = cat([past_key_values[1], V])

attn = softmax(Q @ K.T / sqrt(d)) @ V  # content 自然 attend 到 state
```

每层有独立的 K/V 投影（全参数，不经过 LoRA）：

```python
# StateInterface 读侧
for layer_i, (k_proj, v_proj) in enumerate(zip(read_k_projs, read_v_projs)):
    K = k_proj(state) * sigmoid(read_scale)   # (B, n_state, hidden_size)
    V = v_proj(state) * sigmoid(read_scale)
    kv_pairs[layer_i] = (K, V)
```

### 写侧：state 向 content 提问

最后一层输出后，state 通过专用 Q 投影向 content 提取信息：

```python
# StateInterface 写侧
Q = write_q(state_old)           # (B, n_state, hidden_size)
K = content_hidden               # (B, T, hidden_size)
V = content_hidden

attn = softmax(Q @ K.T / sqrt(d)) @ V
state_new = write_out(attn)      # (B, n_state, state_dim)

# gate 决定更新
gate = sigmoid(gate_proj(cat[state_old, state_new]))
state_next = gate * state_old + (1 - gate) * state_new
```

### 为什么读写分离

v1 架构中，读写分离利用序列位置和因果 attention 方向性（Read 在序列头部只携带旧信息，Write 在尾部吸收新信息）。

v2 架构中，读写分离通过 Q/K/V 角色互换实现更优雅的对称性：
- **读侧**：state 提供 K/V，content 的 Q 去查询 → 记忆自然融入每层思考
- **写侧**：state 提供 Q，content 提供 K/V → 从思考中提取要记住的信息
- **信息防泄漏**：state 不在序列中，backbone 只处理纯 content，不存在 state 偷看答案的问题

### 状态容量

| 配置 | 值 |
|------|-----|
| n_state | 32 tokens |
| state_dim | 1024 (可独立于 hidden_size) |
| 原始大小 | 32 × 1024 = 32,768 floats ≈ 128KB |
| 有效信息 | ~40KB（压缩表示，非原文） |

固定容量是特性，不是缺陷——迫使系统学会选择性记忆和抽象压缩。

---

## Gate：工作记忆刷新器

Gate 是纯动态决策，只回答一个问题："这个 slot 此刻要不要更新？"

```python
# 纯动态 gate——根据内容决定此刻该记还是该忘
gate = sigmoid(gate_proj(cat[state_old, state_new]))

state_next = gate * state_old + (1 - gate) * state_new
```

- `gate ≈ 1` → 保持（这条信息还有用，别覆盖）
- `gate ≈ 0` → 更新（有新信息要写入）
- 不再有快慢之分，不再试图涌现长短期分化

### 设计演进

v1 使用双层 gate：`sigmoid(static_bias + dynamic_proj)`，期望静态偏置涌现"慢维度=长期记忆，快维度=工作记忆"。但训练中发现：
1. 课程全是 16 轮以内的短 episode，没有长短期分化的训练信号
2. gate_bias 大概率学了近似均匀值，等于没用
3. State 被迫同时承担工作记忆和长期记忆两个角色

v2 明确了记忆体系的分层：

```
┌────────────────────────────────────────────────────┐
│                    心核记忆体系                       │
│                                                    │
│   State（工作记忆）          Sleep → MLP（长期记忆）  │
│   ┌──────────────┐          ┌──────────────────┐   │
│   │ gate_proj 管理 │          │ sleep 机制固化     │   │
│   │ 随时更新       │   ──→   │ 缓慢沉淀          │   │
│   │ 只管当前对话   │  固化    │ 跨对话持久         │   │
│   │ = 海马体       │          │ = 皮层            │   │
│   └──────────────┘          └──────────────────┘   │
│                                                    │
│   各管各的，gate 只做"海马体的刷新控制器"              │
└────────────────────────────────────────────────────┘
```

---

## Sleep 机制：记忆固化 + 在线学习

类比人类睡眠：白天积累经验（state），睡眠时突触重塑（权重更新）。

### 两阶段运行模式

```
白天（推理）：                         Sleep（学习）：
  权重完全冻结                           Memory MLP 更新
  只有 state 在流动                      回放 (state_in, content, labels) 序列
  Memory MLP 输出叠加（已固化的记忆）      逐步弱化 state KV 注入
  全速生成，无额外开销                    迫使记忆转移到 Memory MLP
  保存 (state_in, content, labels) 到 buffer
```

### Memory MLP：专用长期记忆模块

v2 的设计原则是"专用模块做专用事"——state 路由用专用投影而非 LoRA，长期记忆同样用专用 MLP 而非 LoRA 挂在 backbone 上。

```
backbone output (B, T, hidden_size)
        ↓
  + Memory MLP(output) * memory_scale    ← sleep 时训练，零初始化启动
        ↓
  final_output → LM head
              → write_from_content(state, final_output)
```

Memory MLP 是 xinhe 系统自己的模块（SwiGLU 结构），输出层零初始化 + 渐进 scale，加上不影响原有行为。Sleep 时训练它编码长期记忆。

### Sleep 流程

1. 从 replay buffer 采样：**70% 近期对话 + 30% 历史对话**
2. 逐步弱化 state KV 注入（read_scale: 当前值 → 0）
3. 标准交叉熵 loss，模型必须靠 Memory MLP 而非 state 答对

```
更新:  Memory MLP           lr = 1e-4   专用全参模块，快速学习
冻结:  LoRA (attention)     lr = 0      只做语言适配，sleep 无需更新
冻结:  StateInterface       lr = 0      保护 gate + read/write 投影
```

### Replay Buffer：不清空，持续积累

Buffer 不在 sleep 后清空，而是作为长期记忆档案持续保存。Sleep 时按比例混合采样：
- **近期对话**（70%）：学习新记忆
- **历史采样**（30%）：巩固旧记忆 + 发现跨时间的关联

### 醒来后的效果

Memory MLP 通过残差连接**叠加**到 backbone 输出上（加法，非替换）：
- 新事实（未 sleep）：靠 state，和以前一样
- Sleep 过的事实：state + Memory MLP 双通道增强
- State 清空后：Memory MLP 兜底，仍能回答

### 记忆系统对应

```
人脑                         Xinhe
──────────                   ─────
神经激活（工作记忆）           state（当前对话的工作台）
突触连接（长期记忆）           Memory MLP（sleep 后积累的知识）
睡眠时海马体→皮层转移          sleep 时弱化 state → 迫使 Memory MLP 学习
```

---

## 参数分类与基座迁移

StateInterface 的参数分为两类，迁移时只保留核心参数（灵魂），丢弃身体适配层：

### 参数分类

| 类别 | 参数 | 依赖 backbone? | 迁移时 |
|------|------|---------------|--------|
| **Core（灵魂）** | state_emb, gate_proj | 否 | 保留 |
| **Projection（身体适配）** | read_k/v_projs, read_scale, write_q, write_out | 是（涉及 hidden_size） | 重新初始化 |
| **Memory MLP** | up, down, gate, memory_scale | 是（涉及 hidden_size） | 重新初始化 |
| **LoRA** | attention q/v_proj 上的 LoRA A/B | 是 | 重新初始化 |

Core 参数只在 state_dim 空间操作，编码了"记忆更新决策"的核心技能。这些能力不依赖具体的 backbone。

Projection 参数涉及 hidden_size，是 state_dim 和 backbone 之间的桥接。切换 backbone 时需要重训。

### 迁移流程

```
源 backbone (0.8B):  完成基础记忆课程 → 保存 checkpoint
                         ↓ extract_core()
                     提取 Core 参数 (丢弃 proj/LoRA/optimizer)
                         ↓
目标 backbone (4B):  加载 Core → 新 projections 随机初始化 → 新 LoRA 零初始化
                         ↓
                     迁移课程: core 冻结, 训 projections + LoRA
```

### LoRA 职责变化

v2 中各组件职责清晰分离：

```
专用 K/V 投影: State 读路由（全参数，不受低秩限制）
专用 Q 投影:   State 写路由（全参数，不受低秩限制）
Gate:          更新决策（保留 vs 采纳）
LoRA:          只管语言适配（单一目标，优化更容易）
```

LoRA 不再负责 state 路由，这是 v2 解决 4B 扩展瓶颈的关键。

---

## .pt = AI 的灵魂

```python
checkpoint = {
    "state_interface": ...,          # StateInterface（gate/scale/state_emb/projections）
    "memory_mlp": ...,               # Memory MLP（长期记忆，sleep 时更新）
    "lora_state": ...,               # LoRA（attention 层，语言适配）
    "state": ...,                    # 当前 state（工作记忆）
    "replay_buffer": ...,            # 待 sleep 的对话序列
    "sleep_count": ...,              # AI 的"年龄"
}
```

每个用户的 AI 从同一个预训练起点出发，经历不同的对话和 sleep 周期后 .pt 逐渐分化。不同的 .pt 就是不同的"人"。

---

## 安全启动（不破坏聊天能力）

三层保护，确保加入 StateInterface 不破坏 backbone 原有能力：

1. **LoRA 零初始化**：B 矩阵全零 → 训练开始时 backbone 行为 = 原始模型
2. **状态嵌入近零初始化**：state_emb 近零 (std=0.01)，投影输出接近零
3. **渐进影响力 read_scale**：`sigmoid(-5.0) ≈ 0.007`，注入的 state K/V 几乎消失

v2 的额外优势：state 不在序列中，backbone 处理的是纯 content，不存在额外 token 干扰 attention 分布的问题。

---

## Attention Mask 与 Position IDs

### Attention Mask

State K/V 通过 `past_key_values` 注入后，占据 attention 的 "past" 位置：

```
state 占据 "past" 位置 0..31
content 占据位置 32..32+T-1（在 attention 内部自动偏移）
mask: content 可以 attend 到所有 state 位置 + 因果 content 位置
```

### Position IDs

Content 的 position_ids 从 0 开始（不是 32），确保 RoPE 编码正确。State K/V 没有 RoPE——这是合理的：state 不是序列中的某个位置，而是"外部记忆"，不应有位置偏好。Content 对 state 的 attention 是纯内容匹配。

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
| 训练对象 | StateInterface (~5M) + LoRA (注入 q/v_proj) |
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
| 更新对象 | Memory MLP (lr=1e-4) |
| 冻结 | StateInterface + LoRA |
| 安全措施 | sleep 前保存 checkpoint，失败可回滚 |

---

## 性能影响

### 序列长度

```
v1: 32(read) + T(content) + 32(write) = T + 64
v2: T(content)

backbone 处理的序列短了 64 个 token
```

### 额外开销

| 操作 | 开销 |
|------|------|
| 读侧 K/V 生成 | n_layers × 2 × Linear(state_dim, hidden_size) × 32 tokens ≈ 微秒级 |
| KV-Cache 拼接 | Flash Attention 原生支持，近乎零开销 |
| 写侧 cross-attention | 1 次 (32 × T) attention ≈ 微秒级 |

### 兼容性

| 优化 | 兼容 | 原因 |
|------|------|------|
| torch.compile | 是 | 使用标准 API，无动态控制流 |
| Flash Attention | 是 | past_key_values 原生支持 |
| gradient checkpointing | 是 | backbone forward 不变 |

**净效果：大概率比 v1 更快。**

---

## 参数总览

| 参数 | 值 | 说明 |
|------|-----|------|
| n_state | 32 | 状态 token 数 |
| state_dim | 1024 | 状态维度（可独立于 hidden_size） |
| read_scale_init | -5.0 | 初始影响力 sigmoid(-5)≈0.007 |
| read_k/v_projs | n_layers × Linear(state_dim, hidden_size) | 每层独立 K/V 投影 |
| write_q | Linear(state_dim, hidden_size) | 写侧 Q 投影 |
| write_out | Linear(hidden_size, state_dim) | 写侧输出投影（identity init） |
| gate_proj | Linear(2×state_dim, state_dim) | 动态门控 |
| LoRA rank | 4 | 注入 q_proj, v_proj（语言适配） |
| lora_alpha | 8 | LoRA 缩放 |
| Memory MLP | SwiGLU(hidden_size, hidden_size*2) | 长期记忆（sleep 时更新） |
| tbptt_steps | 可变 | 课程学习中随 episode_length 调整 |
