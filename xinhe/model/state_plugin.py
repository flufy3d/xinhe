"""
StateInterface — 对称 Cross-Attention + Entity-Keyed State (v4)

v4 核心变化 (相对 v3):
- 新增 slot_keys (learnable 身份向量), key_proj (content→key space), temperature
- write 侧 attention matrix 混合: (1-α)*原 Q@K 路径 + α*key@slot_keys 路由
- read 侧 state 混合: (1-α)*state + α*slot_keys → k_proj (V 仍来自动态 state)
- eks_alpha 初始 sigmoid(-5)≈0, 续训开局严格等价 v3
- 删除 slot_attn_read (EKS routing 足够), 保留 slot_attn_write

原 v2 设计:
- 读侧: 每层 backbone 之前，content 通过 cross-attention 读取 state K/V
- 写侧: backbone 输出后，state 通过 cross-attention 从 content 提取信息
- Gate: 纯动态决策, 控制 state 更新
- State 不在序列中, backbone 只处理纯 content
- 通过 layer_hook 回调注入, 对 DeltaNet/full attention 统一生效
"""
import torch
import torch.nn as nn


# 参数分类: Core 是灵魂 (backbone-agnostic), Projection 是身体适配 (backbone-specific)
# key_proj 依赖 hidden_size → 归 Projection (迁移时需重新训)
# slot_keys / temperature / eks_alpha 在 state_dim 空间或标量 → 归 Core (可迁移)
CORE_PARAM_PREFIXES = (
    "state_emb", "gate_proj", "slot_attn_write",
    "slot_keys", "temperature", "eks_alpha",
)
PROJECTION_PARAM_PREFIXES = (
    "read_k_projs", "read_v_projs", "read_scale",
    "write_q", "write_out", "key_proj",
)
SLOT_ATTN_PARAM_PREFIXES = ("slot_attn_write",)
EKS_CORE_PARAM_PREFIXES = ("slot_keys", "temperature", "eks_alpha")
EKS_PARAM_PREFIXES = ("slot_keys", "key_proj", "temperature", "eks_alpha")  # 全部 EKS (含 projection)


class SlotAttn(nn.Module):
    """Slot 间自注意力 + FFN (state_dim 空间的标准 transformer block)。

    作用: 让 state 的 n_state 个 slot 互相感知, 产生角色分化。解决 entity
    路由问题 — v1 时代已诊断的真正瓶颈。

    严格恒等初始化: self_attn.out_proj 和 mlp 输出层的 weight/bias 置零, 配合
    residual 保证初始 SlotAttn(x) = x, 续训时不会冲击已有 checkpoint 行为。
    """

    def __init__(self, state_dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(state_dim)
        self.self_attn = nn.MultiheadAttention(
            state_dim, n_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )

        # 严格恒等初始化: 两个 residual 子层的输出通道清零
        with torch.no_grad():
            nn.init.zeros_(self.self_attn.out_proj.weight)
            if self.self_attn.out_proj.bias is not None:
                nn.init.zeros_(self.self_attn.out_proj.bias)
            nn.init.zeros_(self.mlp[-1].weight)
            if self.mlp[-1].bias is not None:
                nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, n_state, state_dim) → (B, n_state, state_dim)"""
        x = self.norm1(state)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        state = state + attn_out
        state = state + self.mlp(self.norm2(state))
        return state


class StateInterface(nn.Module):
    """
    对称 Cross-Attention 状态管理器。

    参数:
        n_state: 状态 slot 数量
        state_dim: 状态维度 (可独立于 hidden_size)
        hidden_size: backbone 的隐藏维度 (None 时等于 state_dim)
        n_layers: backbone 层数 (每层独立 K/V 投影)
        state_scale_init: 渐进影响力初始值 (sigmoid 前)
    """

    def __init__(
        self,
        n_state: int = 32,
        state_dim: int = 768,
        hidden_size: int = None,
        n_layers: int = 24,
        state_scale_init: float = -5.0,
        temperature_init: float = 1.0,
        eks_alpha_init: float = -5.0,
        eks_enabled: bool = True,
    ):
        super().__init__()
        self.n_state = n_state
        self.state_dim = state_dim
        self.hidden_size = hidden_size if hidden_size is not None else state_dim
        self.eks_enabled = eks_enabled

        # ── 初始状态 ──
        self.state_emb = nn.Parameter(torch.randn(n_state, state_dim) * 0.01)

        # ── 读侧: state → K/V（每层独立投影）──
        self.read_k_projs = nn.ModuleList([
            nn.Linear(state_dim, self.hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_v_projs = nn.ModuleList([
            nn.Linear(state_dim, self.hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_scale = nn.Parameter(torch.tensor(state_scale_init))

        # ── 写侧: state(Q) × content(K,V) → state_new (原 v2 路径) ──
        self.write_q = nn.Linear(state_dim, self.hidden_size, bias=False)
        self.write_out = nn.Linear(self.hidden_size, state_dim, bias=False)
        # identity-like init: 写侧初始接近恒等映射
        with torch.no_grad():
            nn.init.zeros_(self.write_out.weight)
            d = min(state_dim, self.hidden_size)
            self.write_out.weight[:d, :d] = torch.eye(d)

        # ── Gate: 纯动态决策 ──
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)

        # ── Slot 间通信: 只保留写侧 (读侧由 EKS routing 承担) ──
        self.slot_attn_write = SlotAttn(state_dim, n_heads=4)

        # ── EKS (Entity-Keyed State): slot 身份 + key 路由 ──
        # slot_keys: 每个 slot 的 learnable 身份向量 (static, 不随 state 变)
        # key_proj: 把 content hidden 投影到 key/state 空间, 和 slot_keys 点积做 routing
        # temperature: 路由温度 (learnable), clamp(min=0.3) 防止过 sharp
        # eks_alpha: 新旧路径混合权重, sigmoid(-5)≈0 → 开局完全走原 v2 路径, 续训友好
        self.slot_keys = nn.Parameter(torch.randn(n_state, state_dim) * 0.1)
        self.key_proj = nn.Linear(self.hidden_size, state_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        self.eks_alpha = nn.Parameter(torch.tensor(float(eks_alpha_init)))
        # 记录最近一次 write 的 routing 分布 (B, T, n_state), 供 entropy aux loss 读取
        self.last_write_routing: torch.Tensor | None = None

    # ---- 参数分类 API (用于迁移 / 冻结) ----

    def core_parameters(self) -> list[nn.Parameter]:
        """返回 backbone 无关的核心参数 (灵魂)"""
        return [p for n, p in self.named_parameters()
                if any(n.startswith(prefix) for prefix in CORE_PARAM_PREFIXES)]

    def projection_parameters(self) -> list[nn.Parameter]:
        """返回 backbone 相关的投影参数 (身体适配)"""
        return [p for n, p in self.named_parameters()
                if any(n.startswith(prefix) for prefix in PROJECTION_PARAM_PREFIXES)]

    def slot_attn_parameters(self) -> list[nn.Parameter]:
        """返回 slot 间通信模块的参数 (可用独立 LR 加速从恒等激活)"""
        return [p for n, p in self.named_parameters()
                if any(n.startswith(prefix) for prefix in SLOT_ATTN_PARAM_PREFIXES)]

    def freeze_core(self):
        """冻结核心参数 (迁移时只训投影层 + LoRA)"""
        for p in self.core_parameters():
            p.requires_grad = False

    def unfreeze_core(self):
        """解冻核心参数"""
        for p in self.core_parameters():
            p.requires_grad = True

    def core_state_dict(self) -> dict:
        """提取核心参数 state_dict (用于跨 backbone 迁移)"""
        return {k: v for k, v in self.state_dict().items()
                if any(k.startswith(prefix) for prefix in CORE_PARAM_PREFIXES)}

    # ---- 状态操作 ----

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """创建空白初始状态。返回: (B, n_state, state_dim)"""
        if device is None:
            device = self.state_emb.device
        return self.state_emb.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def generate_read_kv(self, state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        读侧: 生成每层的 state K/V。

        EKS 混合: K 由 (1-α)*state + α*slot_keys 投影而来 (α=sigmoid(eks_alpha))。
          - α=0: 纯 v2 路径, K 完全来自动态 state
          - α=1: 纯 EKS, K 完全来自静态 slot 身份
        V 始终来自动态 state (内容本身), 不混合。

        参数:
            state: (B, n_state, state_dim) 当前持久状态

        返回:
            list[(K, V)] 长度 = n_layers, 每个 K/V 形状 (B, n_state, hidden_size)
        """
        scale = torch.sigmoid(self.read_scale)
        if not self.eks_enabled:
            # 纯 v2 路径: K/V 都从动态 state 投影 (无 EKS 混合)
            kv_pairs = []
            for k_proj, v_proj in zip(self.read_k_projs, self.read_v_projs):
                K = k_proj(state) * scale
                V = v_proj(state) * scale
                kv_pairs.append((K, V))
            return kv_pairs

        B = state.shape[0]
        alpha = torch.sigmoid(self.eks_alpha)
        # slot_keys: (n_state, state_dim) → 广播到 (B, n_state, state_dim)
        slot_keys_broadcast = self.slot_keys.to(dtype=state.dtype).unsqueeze(0).expand(B, -1, -1)
        state_for_key = (1.0 - alpha) * state + alpha * slot_keys_broadcast

        kv_pairs = []
        for k_proj, v_proj in zip(self.read_k_projs, self.read_v_projs):
            K = k_proj(state_for_key) * scale  # K: 混合 (state + slot_keys identity)
            V = v_proj(state) * scale           # V: 纯动态内容
            kv_pairs.append((K, V))
        return kv_pairs

    def read_layer(self, hidden_states: torch.Tensor, state_kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        读侧单层: content 通过 cross-attention 读取 state K/V。

        参数:
            hidden_states: (B, T, hidden_size) 当前层的输入
            state_kv: (K, V) 每个形状 (B, n_state, hidden_size)

        返回:
            hidden_states: (B, T, hidden_size) 融入 state 信息后的输出
        """
        K, V = state_kv
        # 确保 dtype 一致
        K = K.to(dtype=hidden_states.dtype, device=hidden_states.device)
        V = V.to(dtype=hidden_states.dtype, device=hidden_states.device)

        d = hidden_states.shape[-1]
        attn_weights = torch.softmax(
            hidden_states @ K.transpose(-2, -1) / (d ** 0.5), dim=-1
        )  # (B, T, n_state)
        cross_attn_out = attn_weights @ V  # (B, T, hidden_size)

        return hidden_states + cross_attn_out

    def write_from_content(self, state_old: torch.Tensor, content_hidden: torch.Tensor) -> torch.Tensor:
        """
        写侧: state 向 content 提问，提取要记住的信息。

        EKS 混合 (attention matrix 层面):
          - 原路径 attn_old: softmax(write_q(state_old) @ content.T / sqrt(d)) → (B, n, T)
            含义: 每个 slot 主动向 content 提问, 靠 state_old 内容决定路由
          - EKS 路径 attn_new: softmax(key_proj(content) @ slot_keys.T / τ).T → (B, n, T)
            含义: 每个 content token 按 key 投影匹配 slot 身份, 决定写入哪个 slot
          - 混合: attn = (1-α) * attn_old + α * attn_new, α = sigmoid(eks_alpha)

        记录 routing (B, T, n_state) 到 self.last_write_routing, 供外层计算 entropy 正则。

        参数:
            state_old: (B, n_state, state_dim) 旧状态
            content_hidden: (B, T, hidden_size) backbone 最终输出

        返回:
            state_next: (B, n_state, state_dim) 更新后的状态
        """
        # 原 v2 路径: slot(Q) × content(K,V)
        Q = self.write_q(state_old)                          # (B, n_state, hidden_size)
        K_v2 = content_hidden.to(dtype=Q.dtype)              # (B, T, hidden_size)
        V = K_v2                                              # V 始终 = content
        d_h = Q.shape[-1]
        attn_old = torch.softmax(Q @ K_v2.transpose(-2, -1) / (d_h ** 0.5), dim=-1)  # (B, n, T)

        if not self.eks_enabled:
            # 纯 v2 写路径, 不计算 EKS / routing / aux
            self.last_write_routing = None
            extracted = attn_old @ V
            state_new = self.write_out(extracted)
            state_new = self.slot_attn_write(state_new)
            gate = torch.sigmoid(self.gate_proj(
                torch.cat([state_old, state_new], dim=-1)
            ))
            return gate * state_old + (1 - gate) * state_new

        # EKS 路径: content token 按 key 投影 → softmax 路由到 slot
        key = self.key_proj(content_hidden.to(dtype=self.key_proj.weight.dtype))  # (B, T, state_dim)
        slot_keys = self.slot_keys.to(dtype=key.dtype)        # (n_state, state_dim)
        tau = self.temperature.clamp(min=0.3)
        d_k = key.shape[-1]
        routing = torch.softmax(
            key @ slot_keys.transpose(0, 1) / (tau * (d_k ** 0.5)),
            dim=-1,
        )                                                     # (B, T, n_state) — 每个 token 的 slot 分布
        self.last_write_routing = routing
        attn_new = routing.transpose(-2, -1).to(dtype=attn_old.dtype)  # (B, n_state, T)

        # 混合 attention matrix
        alpha = torch.sigmoid(self.eks_alpha).to(dtype=attn_old.dtype)
        attn = (1.0 - alpha) * attn_old + alpha * attn_new    # (B, n_state, T)

        extracted = attn @ V                                   # (B, n_state, hidden_size)
        state_new = self.write_out(extracted)                  # (B, n_state, state_dim)
        state_new = self.slot_attn_write(state_new)

        gate = torch.sigmoid(self.gate_proj(
            torch.cat([state_old, state_new], dim=-1)
        ))
        state_next = gate * state_old + (1 - gate) * state_new
        return state_next

    def get_state_stats(self, state: torch.Tensor) -> dict:
        """
        获取状态分析统计信息。

        参数:
            state: (B, n_state, D) 或 (n_state, D)

        返回:
            dict: 包含 read_scale、状态范数、有效秩等
        """
        if state.dim() == 3:
            state = state[0]

        U, S, V = torch.linalg.svd(state.float())
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        effective_rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm))).item()

        return {
            "read_scale": torch.sigmoid(self.read_scale).item(),
            "state_norm": state.norm().item(),
            "effective_rank": effective_rank,
        }
