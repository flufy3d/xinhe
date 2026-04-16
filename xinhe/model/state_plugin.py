"""
StateInterface — 对称 Cross-Attention 状态机制 (v2)

核心设计:
- 读侧: 每层 backbone 之前，content 通过 cross-attention 读取 state K/V（专用全参投影）
- 写侧: backbone 输出后，state 通过 cross-attention 从 content 提取信息
- 对称性: 读是 Content(Q) × State(K,V)，写是 State(Q) × Content(K,V)
- Gate: 纯动态决策，控制 state 更新
- State 不在序列中，backbone 只处理纯 content
- 通过 layer_hook 回调注入，对 DeltaNet/full attention 统一生效
"""
import torch
import torch.nn as nn


# 参数分类: Core 是灵魂 (backbone-agnostic), Projection 是身体适配 (backbone-specific)
CORE_PARAM_PREFIXES = ("state_emb", "gate_proj")
PROJECTION_PARAM_PREFIXES = ("read_k_projs", "read_v_projs", "read_scale", "write_q", "write_out")


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
    ):
        super().__init__()
        self.n_state = n_state
        self.state_dim = state_dim
        self.hidden_size = hidden_size if hidden_size is not None else state_dim

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

        # ── 写侧: state(Q) × content(K,V) → state_new ──
        self.write_q = nn.Linear(state_dim, self.hidden_size, bias=False)
        self.write_out = nn.Linear(self.hidden_size, state_dim, bias=False)
        # identity-like init: 写侧初始接近恒等映射
        with torch.no_grad():
            nn.init.zeros_(self.write_out.weight)
            d = min(state_dim, self.hidden_size)
            self.write_out.weight[:d, :d] = torch.eye(d)

        # ── Gate: 纯动态决策 ──
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)

    # ---- 参数分类 API (用于迁移 / 冻结) ----

    def core_parameters(self) -> list[nn.Parameter]:
        """返回 backbone 无关的核心参数 (灵魂)"""
        return [p for n, p in self.named_parameters()
                if any(n.startswith(prefix) for prefix in CORE_PARAM_PREFIXES)]

    def projection_parameters(self) -> list[nn.Parameter]:
        """返回 backbone 相关的投影参数 (身体适配)"""
        return [p for n, p in self.named_parameters()
                if any(n.startswith(prefix) for prefix in PROJECTION_PARAM_PREFIXES)]

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

        参数:
            state: (B, n_state, state_dim) 当前持久状态

        返回:
            list[(K, V)] 长度 = n_layers，每个 K/V 形状 (B, n_state, hidden_size)
        """
        scale = torch.sigmoid(self.read_scale)
        kv_pairs = []
        for k_proj, v_proj in zip(self.read_k_projs, self.read_v_projs):
            K = k_proj(state) * scale
            V = v_proj(state) * scale
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

        参数:
            state_old: (B, n_state, state_dim) 旧状态
            content_hidden: (B, T, hidden_size) backbone 最终输出

        返回:
            state_next: (B, n_state, state_dim) 更新后的状态
        """
        Q = self.write_q(state_old)      # (B, n_state, hidden_size)
        K = content_hidden               # (B, T, hidden_size)
        V = content_hidden

        d = Q.shape[-1]
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (d ** 0.5), dim=-1)
        extracted = attn @ V             # (B, n_state, hidden_size)

        state_new = self.write_out(extracted)  # (B, n_state, state_dim)

        # 纯动态 gate
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
