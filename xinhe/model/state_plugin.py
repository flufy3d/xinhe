"""
StateInterface (v5b) — 对称 Cross-Attention + 动态 Gate + Contrastive Value Head

v5 减法 (v5a):
  删除 EKS/SlotAttn/对角单位 init/参数分组 API

v5b 新增: value_head 支持 contrastive (InfoNCE) loss, objective 监督 slot 身份

参数量 (1024 state_dim, 1024 hidden_size, n_layers=6):
  ~17.86M
"""
import torch
import torch.nn as nn


class StateInterface(nn.Module):
    """对称 Cross-Attention 状态管理器 (v5b)。"""

    def __init__(
        self,
        n_state: int = 32,
        state_dim: int = 768,
        hidden_size: int = None,
        n_layers: int = 24,
        state_scale_init: float = -5.0,
        write_iterations: int = 1,
    ):
        super().__init__()
        self.n_state = n_state
        self.state_dim = state_dim
        self.hidden_size = hidden_size if hidden_size is not None else state_dim
        self.write_iterations = max(1, int(write_iterations))

        # ── 初始空白状态 ──
        self.state_emb = nn.Parameter(torch.randn(n_state, state_dim) * 0.01)

        # ── 读侧: state → K/V (每层独立投影) ──
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
        nn.init.xavier_uniform_(self.write_q.weight)
        nn.init.xavier_uniform_(self.write_out.weight)

        # ── Gate: 纯动态决策 ──
        self.gate_proj = nn.Linear(2 * state_dim, state_dim, bias=False)
        nn.init.xavier_uniform_(self.gate_proj.weight)

        # ── v5b: value_head (contrastive loss 时使用) ──
        self.value_head = nn.Linear(state_dim, self.hidden_size, bias=False)
        nn.init.xavier_uniform_(self.value_head.weight)

    # ---- 状态操作 ----

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """创建空白初始状态。返回: (B, n_state, state_dim)"""
        if device is None:
            device = self.state_emb.device
        return self.state_emb.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def generate_read_kv(self, state: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """读侧: 生成每层的 state K/V。"""
        scale = torch.sigmoid(self.read_scale)
        kv_pairs = []
        for k_proj, v_proj in zip(self.read_k_projs, self.read_v_projs):
            K = k_proj(state) * scale
            V = v_proj(state) * scale
            kv_pairs.append((K, V))
        return kv_pairs

    def read_layer(
        self,
        hidden_states: torch.Tensor,
        state_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """读侧单层 cross-attention。"""
        K, V = state_kv
        K = K.to(dtype=hidden_states.dtype, device=hidden_states.device)
        V = V.to(dtype=hidden_states.dtype, device=hidden_states.device)
        d = hidden_states.shape[-1]
        attn_weights = torch.softmax(
            hidden_states @ K.transpose(-2, -1) / (d ** 0.5), dim=-1
        )
        cross_attn_out = attn_weights @ V
        return hidden_states + cross_attn_out

    def write_from_content(
        self,
        state_old: torch.Tensor,
        content_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写侧 (v5b): state(Q) × content(K,V), gate 决策融合。"""
        K = content_hidden.to(dtype=state_old.dtype)
        V = K
        d = self.hidden_size

        current_state = state_old
        attn = None
        for _ in range(self.write_iterations):
            Q = self.write_q(current_state)
            attn = torch.softmax(Q @ K.transpose(-2, -1) / (d ** 0.5), dim=-1)
            extracted = attn @ V
            state_new = self.write_out(extracted)
            gate = torch.sigmoid(self.gate_proj(
                torch.cat([current_state, state_new], dim=-1)
            ))
            current_state = gate * current_state + (1 - gate) * state_new

        return current_state, attn

    # ---- 诊断 ----

    def get_state_stats(self, state: torch.Tensor) -> dict:
        """状态统计 (read_scale / norm / effective_rank)"""
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
