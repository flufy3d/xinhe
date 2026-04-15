# 心核 v2 架构：对称 Cross-Attention State

## 背景

v1 架构（序列注入）在 0.8B 上训练成功（99%），但在 4B 上 entity 区分卡在 87%。
经过完整的排查和头脑风暴（详见 `4b_diagnosis_and_statenet.md`），确定根因是 LoRA 全局共享无法为 state token 提供独立的 K/V 投影能力。

v2 架构将 state 从"序列里的 token"变为"注意力的对话者"，通过对称的 cross-attention 实现读写，彻底绕过 LoRA 瓶颈。

---

## 设计思路演进

```
v1: state 作为 token 混入序列 → backbone 统一处理 → LoRA 全局限制 → 4B entity 路由失败
     ↓
尝试: plugin 内部加 slot 间交互 → 无法改变 backbone 内部的 LoRA 行为
     ↓
尝试: LoRA 层内加 StateNet → 能解决但侵入 backbone、不可迁移
     ↓
v2: state 移出序列 → 专用网络生成 K/V 注入 attention → 对称 cross-attention 读写
    → 不改 backbone 内部 → 利用 past_key_values 标准 API → entity 路由由专用网络负责
```

---

## v1 vs v2 架构对比

### v1：序列注入

```
[Read-State(32) | Content(T) | Write-State(32)]
                    ↓
        backbone 32 层 (W + LoRA)
    Read/Content/Write 共享同一个 LoRA
                    ↓
        拆分 → content_out → logits
              → write_out  → gate → state_next
```

- State 是序列中的 token，K/V 由 backbone 的 `W_k + LoRA_k` 投影
- LoRA 全局共享 → state 和 content 得到相同的 K/V 变换
- 0.8B 上 LoRA 能撬动小模型 → work
- 4B 上 LoRA 撬不动 → entity 路由失败

### v2：对称 Cross-Attention

```
state_old
  ├── Read: state → 专用 K/V 投影 → 注入每层 attention cache
  │                                     ↓
  │   [Content(T)] → backbone 32 层 → content_final
  │                  (每层 attend 到 state K/V)
  │                                     ↓
  │                                   logits
  │                                     │
  └── Write: state → 专用 Q 投影 → attend to content_final
                                        ↓
                                   state_new → gate → state_next
```

- State 不在序列中，通过 past_key_values API 注入
- K/V 由专用全参网络生成，不经过 LoRA → entity 路由不受限
- 读写对称：读是 Content(Q) × State(K,V)，写是 State(Q) × Content(K,V)
- Backbone 只处理纯 content

---

## 核心对称性

```
读: Content(Q)  ×  State(K,V)  →  记忆融入思考    (每层)
写: State(Q)    ×  Content(K,V) →  信息写入记忆    (最后一层后)

Q 和 K/V 角色互换，机制完全对称
```

| | 读 | 写 |
|---|---|---|
| Q 来自 | content（backbone 内部） | state（专用投影） |
| K/V 来自 | state（专用投影） | content（backbone 输出） |
| 机制 | cross-attention（通过 KV-Cache） | cross-attention（独立模块） |
| 时机 | 每层 | 最后一层输出后 |
| 意义 | 带着记忆思考 | 从思考中提取记忆 |

---

## 实现方案

### 关键：利用 HuggingFace past_key_values API，零侵入 backbone

HuggingFace transformer 的每层 attention 原生支持 past_key_values：

```python
# Qwen attention forward（已有代码，不需要修改）
Q = q_proj(hidden_states)
K = k_proj(hidden_states)
V = v_proj(hidden_states)

if past_key_values is not None:
    K = cat([past_key_values[0], K])   # state K/V 自然拼到前面
    V = cat([past_key_values[1], V])

attn = softmax(Q @ K.T / sqrt(d)) @ V  # content 自然 attend 到 state
```

### StateInterface 模块

```python
class StateInterface(nn.Module):
    """对称的 state 读写接口"""

    def __init__(self, n_state, state_dim, hidden_size, n_heads, n_layers):
        super().__init__()
        self.n_state = n_state
        self.state_dim = state_dim

        # ── 初始状态 ──
        self.state_emb = nn.Parameter(torch.randn(n_state, state_dim) * 0.01)

        # ── 读侧: state → K/V（每层独立投影）──
        self.read_k_projs = nn.ModuleList([
            nn.Linear(state_dim, hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_v_projs = nn.ModuleList([
            nn.Linear(state_dim, hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_scale = nn.Parameter(torch.tensor(-5.0))  # 渐进影响力

        # ── 写侧: state(Q) × content(K,V) → state_new ──
        self.write_q = nn.Linear(state_dim, hidden_size, bias=False)
        self.write_out = nn.Linear(hidden_size, state_dim, bias=False)
        nn.init.eye_(self.write_out.weight[:state_dim, :state_dim])

        # ── Gate ──
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)

    def blank_state(self, batch_size, device=None):
        if device is None:
            device = self.state_emb.device
        return self.state_emb.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def generate_read_kv(self, state):
        """读侧: 生成每层的 state K/V，用于注入 past_key_values"""
        scale = torch.sigmoid(self.read_scale)
        kv_pairs = []
        for k_proj, v_proj in zip(self.read_k_projs, self.read_v_projs):
            K = k_proj(state) * scale   # (B, n_state, hidden_size)
            V = v_proj(state) * scale
            kv_pairs.append((K, V))
        return kv_pairs

    def write_from_content(self, state_old, content_hidden):
        """写侧: state 向 content 提问，提取要记住的信息"""
        Q = self.write_q(state_old)                              # (B, n_state, hidden_size)
        K = content_hidden                                        # (B, T, hidden_size)
        V = content_hidden

        d = Q.shape[-1]
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (d ** 0.5), dim=-1)
        extracted = attn @ V                                      # (B, n_state, hidden_size)

        state_new = self.write_out(extracted)                     # (B, n_state, state_dim)

        gate = torch.sigmoid(self.gate_proj(
            torch.cat([state_old, state_new], dim=-1)
        ))
        state_next = gate * state_old + (1 - gate) * state_new
        return state_next
```

### XinheModel forward 流程

```python
class XinheModel(nn.Module):
    def forward(self, input_ids, state, labels=None):
        # 1. Content embedding
        content_emb = self.backbone.embed(input_ids)

        # 2. 读侧: 生成 state K/V
        state_kv = self.state_interface.generate_read_kv(state)

        # 3. Backbone forward（state 通过 past_key_values 注入）
        content_output = self.backbone.forward_blocks(
            hidden_states=content_emb,
            attention_mask=mask,             # 允许 attend 到 state 位置
            position_ids=position_ids,       # content 从 0 开始
            past_key_values=state_kv,        # state K/V 注入！
        )

        # 4. 写侧: 从 content 提取信息更新 state
        state_next = self.state_interface.write_from_content(state, content_output)

        # 5. 语言模型输出
        logits = self.lm_head(content_output)
        loss = cross_entropy(logits, labels) if labels else None

        return {"logits": logits, "state_next": state_next, "loss": loss}
```

### QwenBackbone 改动

唯一改动：`forward_blocks` 接受并传递 `past_key_values`：

```python
class QwenBackbone:
    def forward_blocks(self, hidden_states, attention_mask, position_ids,
                       past_key_values=None):
        # past_key_values 直接传给 HuggingFace model
        outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,   # 新增这一行
            use_cache=False,
        )
        return outputs.last_hidden_state
```

---

## 参数分类与迁移

### Core 参数（state_dim 空间，backbone 无关，可迁移）

| 参数 | 形状 | 说明 |
|------|------|------|
| state_emb | (n_state, state_dim) | 初始空白状态 |
| gate_proj | (2×state_dim, state_dim) | 动态门控 |

### Projection 参数（涉及 hidden_size，backbone 相关，迁移时重训）

| 参数 | 形状 | 说明 |
|------|------|------|
| read_k_projs | n_layers × (state_dim, hidden_size) | 读侧 K 投影 |
| read_v_projs | n_layers × (state_dim, hidden_size) | 读侧 V 投影 |
| read_scale | scalar | 渐进影响力 |
| write_q | (state_dim, hidden_size) | 写侧 Q 投影 |
| write_out | (hidden_size, state_dim) | 写侧输出投影 |

### 迁移路径

```
0.8B 训练:
  core + projections + LoRA → 全部训练 → 99%

迁移到 4B:
  冻结: core（state_emb, gate_proj）
  重训: read/write projections + LoRA
  
  LoRA 职责变轻: 只管语言适配（state 路由由专用投影负责）
```

---

## 与 v1 的关键区别

### v1 中 LoRA 被迫身兼三职

```
LoRA 的职责:
  1. 语言适配（让 backbone 适应下游任务格式）
  2. State 读路由（让 content 的 Q 能区分不同 state slot 的 K）
  3. State 写路由（让 write token 的 Q 能关注正确的 content）

三个目标在同一组低秩参数里互相冲突
小模型能 hack 过去，大模型不行
```

### v2 中各组件职责清晰

```
专用 K/V 投影: State 读路由（全参数，不受低秩限制）
专用 Q 投影:   State 写路由（全参数，不受低秩限制）
Gate:          更新决策（保留 vs 采纳）
LoRA:          只管语言适配（单一目标，优化更容易）
```

---

## 保留的设计特性

| 特性 | v1 | v2 |
|------|----|----|
| 带着记忆思考（每层可见 state） | 通过序列中的 read token | 通过每层注入的 K/V cache |
| 涌现性（无硬编码 slot/类别） | 是 | 是（所有投影学出来的） |
| 课程学习 | 是 | 是（不变） |
| 渐进 scale | state_scale 控制 token 缩放 | read_scale 控制 K/V 缩放 |
| 在线学习（.pt = 灵魂） | state 持久化 | state 持久化（不变） |

### Gate 的角色变化：从"长短期分离器"到"工作记忆刷新器"

这是 v1 → v2 最重要的设计理念转变之一。

**v1 的 gate 设计：试图一个人干两件事**

```
gate = sigmoid(gate_bias + gate_proj([old, new]))
               ~~~~~~~~~
               静态偏置：期望涌现"慢维度=长期记忆，快维度=工作记忆"

问题:
  1. 课程全是 16 轮以内的短 episode，没有长短期分化的训练信号
  2. gate_bias 大概率学了近似均匀值，等于没用
  3. State 被迫同时承担工作记忆和长期记忆，两个角色都做不好
```

**v2 的 gate 设计：明确只管工作记忆**

```
gate = sigmoid(gate_proj([old, new]))

纯动态决策，只回答一个问题："这个 slot 此刻要不要更新？"

gate ≈ 1 → 保持（这条信息还有用，别覆盖）
gate ≈ 0 → 更新（有新信息要写入）

不再有快慢之分，不再试图涌现长短期分化
```

**记忆体系的明确分工：**

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
│   各管各的，不用一个 gate 硬撑两个角色                  │
└────────────────────────────────────────────────────┘
```

v1 试图让 gate_bias 在 state 内部涌现长短期分化，是因为当时没有 sleep 机制的设计。
v2 明确了记忆体系的分层：state 是海马体（快速暂存），MLP 是皮层（慢速持久），sleep 是连接两者的固化过程。gate 只需做好"海马体的刷新控制器"这一个角色。

### 去掉的设计

| 组件 | 原因 |
|------|------|
| gate_bias（静态门控偏置） | 无训练信号 + sleep/MLP 接管长期记忆 → 不再需要 |
| read_pos / write_pos（位置编码） | 不再需要序列位置区分 |
| write_emb（写查询嵌入） | 写侧改为 state 自身作为 Q |
| proj_up / proj_down | 维度桥接由 read/write 投影承担 |
| state_out_proj | 写侧输出由 write_out 承担 |

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
| 读侧 K/V 生成 | n_layers × 2 × Linear(1024, 2560) × 32 tokens ≈ 微秒级 |
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

## 需要注意的细节

### RoPE 与 state K/V

Backbone 的 K 有 RoPE（旋转位置编码），但注入的 state K/V 没有 RoPE。
这意味着 content 对 state 的 attention 是纯内容匹配，不含位置偏好。
这是**合理的**：state 不是序列中的某个位置，而是"外部记忆"，不应有位置偏好。

### Attention Mask

需要构造允许 content attend 到 state 位置的 mask：

```python
# state 占据 "past" 位置 0..31
# content 占据位置 32..32+T-1（在 attention 内部自动偏移）
# mask: content 可以 attend 到所有 state 位置 + 因果 content 位置
```

### Position IDs

Content 的 position_ids 从 0 开始（不是 32），确保 RoPE 编码正确。
past_key_values 机制下，HuggingFace 可能自动偏移 position，需要显式传递 position_ids 覆盖。

---

## 验证计划

### Phase 1: 0.8B 验证

在 0.8B 上实现 v2 架构，跑完整 14 阶段课程：
- 目标：ema_acc 95%+（至少和 v1 持平）
- 重点关注 stage 9a/9b（entity 区分）的表现
- 如果 0.8B 上 v2 不如 v1 → 架构有回归，需排查

### Phase 2: 4B 验证

在 4B 上从零跑课程：
- 目标：entity 区分突破 87%
- 如果成功 → v2 架构验证通过

### Phase 3: 迁移验证

0.8B core 迁移到 4B：
- 冻结 core，重训 projections + LoRA
- 验证迁移是否比从零训更快收敛

---

## 改动范围

| 文件 | 改动 |
|------|------|
| `xinhe/model/state_plugin.py` | 重写为 StateInterface（对称 cross-attention） |
| `xinhe/model/xinhe_model.py` | forward 流程适配新接口 |
| `xinhe/model/qwen_backbone.py` | forward_blocks 增加 past_key_values 参数传递 |
| `xinhe/model/config.py` | 新增 read/write 投影相关配置 |
| `configs/*.yaml` | 适配新参数名 |

不需要改动：
| 文件 | 原因 |
|------|------|
| `xinhe/model/lora.py` | LoRA 不变，不再负责 state 路由 |
| `xinhe/training/trainer.py` | 训练循环不变（state 接口兼容） |
| `xinhe/data/*` | 数据生成不变 |
| `configs/curriculum.yaml` | 课程不变 |

---

## 后续展望

### Sleep 固化机制

v2 的 state 是纯工作记忆。后续 sleep 机制将重要记忆固化到 MLP 权重中：
- State（工作记忆）：当前对话上下文，v2 架构管理
- MLP（长期记忆）：跨对话持久知识，sleep 机制固化

gate_bias 在 v1 中试图让 state 自己涌现长短期分化，但没有训练信号。
v2 放弃了这个设计，将长期记忆明确交给 sleep/MLP 机制。

### Think 课程

v2 的 LoRA 只负责语言适配，不再兼顾 state 路由。
Think 课程（从记忆出发做推理）的训练应该更容易收敛，因为 LoRA 的优化目标更纯净。
