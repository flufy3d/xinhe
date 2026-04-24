"""
TurnInterface (v6.2) — 自旋时序罗盘 W_turn（独立写侧 + 多相位共振搜索读侧）

与 FactInterface (W_fact) 并行，挂在 Qwen3 full_attention 层前做 read 注入。

v6.2 写侧升级（2026-04-24）：
  原设计让 W_turn 写侧复用 fact_interface 的 k_proj/v_proj —— 但 0a 训熟后这套投影专注
  entity token 单槽位，对 phrase 整段 tokens 的区分度不够，导致 bootstrap 0b 里不同
  phrase 投出来的 W_turn key 方向近似相同，读侧 q 完全无法找到特定 phrase。
  → 现在 TurnInterface 自带 k_proj_turn / v_proj_turn，和 fact 投影解耦。

核心设计：
  写侧独立可学参数：k_proj_turn / v_proj_turn（让 phrase-level content 有独立投影空间）
  γ 为 config 标量（默认 0.9），不训练
  旋转 inv_freq 为 buffer
  读侧可学参数：q_projs_turn / o_projs_turn / read_scale_turn

读侧：多相位共振搜索
  枚举相位 τ ∈ {0..phase_max} 并行内积 → softmax 共振挑选 → 加权融合。

旋转/时序语义：
  RoPE 内积性质：<R^a·q, R^b·k> = <q, R^(b-a)·k>
  每轮写入时对 W 做一次 d_k 维旋转（age +1），stored key 实际是 R^age·k_emitted。
  读取时对 q 做 P 组正旋转 q_τ = R^τ·q，并行内积 <R^τ·q, R^age·k> = <q, R^(age-τ)·k>
  在 age=τ 时等价于 <q, k>，共振最大 —— 由 softmax 自行选择最匹配的相位。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """split-half 约定（匹配 Qwen RoPE）：最后一维 [x1, x2] → [-x2, x1]。"""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotation_k(
    x: torch.Tensor,
    steps: torch.Tensor,
    inv_freq: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """对 x 最后一维 d_k 做 RoPE-style 旋转 steps 步。

    x: (..., d_k)
    steps: 可广播到 x.shape[:-1]；单位 = "轮"（turn），每轮 inv_freq 决定频率分布
    inv_freq: (d_k//2,) 固定 buffer
    inverse=True 时反向旋转（等价 steps → -steps）
    """
    sign = -1.0 if inverse else 1.0
    # angles: (..., d_k//2) = steps * inv_freq
    angles = sign * steps.unsqueeze(-1).to(x.dtype) * inv_freq.to(x.dtype)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # (..., d_k) split-half duplicate
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return x * cos + _rotate_half(x) * sin


def rotate_W_turn(
    W: torch.Tensor,       # (B,H,d_v,d_k)
    steps: torch.Tensor,   # 标量或 (B,)，把 d_k 维所有行旋转 steps 步
    inv_freq: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """W 的 d_k 维旋转（age 所有存储的 k）。"""
    B, H, d_v, d_k = W.shape
    if steps.dim() == 0:
        steps_b = steps.expand(B, H, d_v)            # 同步全 batch 所有 head/行
    else:
        steps_b = steps.view(B, 1, 1).expand(B, H, d_v)
    return apply_rotation_k(W, steps_b, inv_freq, inverse=inverse)


def rotate_query(
    q: torch.Tensor,       # (B,H,T,d_k)
    dtau: torch.Tensor,    # (B,T) per-token 预测的 Δτ
    inv_freq: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """对 query 的 d_k 维旋转 Δτ 步（查 Δτ 轮前写入的 key）。"""
    B, H, T, d_k = q.shape
    steps = dtau.view(B, 1, T).expand(B, H, T)
    return apply_rotation_k(q, steps, inv_freq, inverse=inverse)


class TurnInterface(nn.Module):
    """自旋时序罗盘 W_turn。与 FactInterface 同形状，写侧共享投影权重。"""

    def __init__(
        self,
        fact_interface: nn.Module,          # 引用 FactInterface，复用其 k_proj/v_proj
        hidden_size: int,
        n_heads: int = 16,
        head_dim: int = 64,
        n_layers: int = 24,
        turn_read_scale_init: float = -3.0,  # sigmoid(-3)≈0.047（与 fact read_scale 同档次，避开梯度死区）
        turn_gamma: float = 0.9,              # 固定衰减标量（非 Parameter）
        turn_rotation_base: float = 10000.0,
        turn_phase_max: int = 5,              # 多相位搜索窗口 τ ∈ {0..phase_max}
        turn_phase_temperature: float = 5.0,  # softmax 温度系数，越大越锐利
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim 必须是偶数（RoPE split-half 要求），得到 {head_dim}"
        assert turn_phase_max >= 0, f"turn_phase_max 必须 ≥ 0，得到 {turn_phase_max}"
        self.hidden_size = hidden_size
        self.H = n_heads
        self.d_k = head_dim
        self.d_v = head_dim
        self.turn_gamma = float(turn_gamma)    # 普通 float 而非 buffer/Parameter
        self.phase_max = int(turn_phase_max)   # 运行时可调，不影响权重 shape
        self.phase_temperature = float(turn_phase_temperature)  # softmax 温度，运行时可调

        # 对 FactInterface 保留非注册引用（不通过 nn.Module 登记，避免 state_dict 重复）
        # 写侧复用其 k_proj/v_proj → 0 可学参数
        self._fact_ref = [fact_interface]   # list 包一层防止 nn.Module 自动注册为子模块

        # ── 读侧：专属 q/o 投影（需要学"怎么查历史轮"）──
        self.q_projs_turn = nn.ModuleList([
            nn.Linear(hidden_size, n_heads * head_dim, bias=False)
            for _ in range(n_layers)
        ])
        self.o_projs_turn = nn.ModuleList([
            nn.Linear(n_heads * head_dim, hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_scale_turn = nn.Parameter(torch.tensor(turn_read_scale_init))

        # ── 写侧：独立 k/v 投影（v6.2 新增，不再复用 fact）──
        # 让 W_turn 对 phrase-level 多 token content 有独立可学的投影空间
        self.k_proj_turn = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.v_proj_turn = nn.Linear(hidden_size, n_heads * head_dim, bias=False)

        for lin in [*self.q_projs_turn, *self.o_projs_turn, self.k_proj_turn, self.v_proj_turn]:
            nn.init.xavier_uniform_(lin.weight)

        # ── 旋转频率：RoPE split-half inv_freq，buffer 不训练 ──
        inv_freq = 1.0 / (turn_rotation_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def fact_interface(self) -> nn.Module:
        """返回 FactInterface 引用（仅用于 write 时复用 k_proj/v_proj）。"""
        return self._fact_ref[0]

    # ---- 状态操作 ----

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """空白 W_turn: (B, H, d_v, d_k)，与 FactInterface.blank_state 同形状。"""
        if device is None:
            device = self.read_scale_turn.device
        dtype = self.read_scale_turn.dtype
        return torch.zeros(
            batch_size, self.H, self.d_v, self.d_k,
            device=device, dtype=dtype,
        )

    def read_layer(
        self,
        hidden_states: torch.Tensor,
        W_turn: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """多相位共振搜索 read：
           1. 投影 q，L2 归一
           2. 对 τ ∈ {0..phase_max} 分别正旋转 q_τ = R^τ · q
           3. 并行内积 r_τ = q_τ @ W_turn.T → (P,B,H,T,d_v)
           4. score_τ = ||r_τ||_2 / √d_v，softmax 归一得 α
           5. read = Σ_τ α_τ · r_τ → o_proj + sigmoid(scale) 残差注入

        hidden_states: (B,T,D)；W_turn: (B,H,d_v,d_k)；返回 (B,T,D)。
        """
        B, T, _ = hidden_states.shape
        dtype = hidden_states.dtype
        W_cast = W_turn.to(dtype=dtype, device=hidden_states.device)

        # ── 1. 投影 q 并 L2 归一（匹配 W_fact 写法）──
        q_w = self.q_projs_turn[layer_idx].weight.to(dtype=dtype)
        q = F.linear(hidden_states, q_w)                                       # (B,T,H*d_k)
        q = q.view(B, T, self.H, self.d_k).transpose(1, 2)                     # (B,H,T,d_k)
        q = F.normalize(q, dim=-1)

        # ── 2. 多相位 stack: 对每个 τ 生成 q_τ ──
        # angles: (P, d_k//2) = τ · inv_freq
        num_phases = self.phase_max + 1
        phase_idx = torch.arange(num_phases, device=q.device, dtype=dtype)     # (P,)
        inv_freq = self.inv_freq.to(dtype=dtype)
        angles = phase_idx.unsqueeze(-1) * inv_freq                             # (P, d_k//2)
        cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)         # (P, d_k) split-half duplicate
        sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)

        # 广播旋转: (1,B,H,T,d_k) * (P,1,1,1,d_k) → (P,B,H,T,d_k)
        q_un = q.unsqueeze(0)                                                   # (1,B,H,T,d_k)
        q_rot_half = _rotate_half(q_un)                                         # (1,B,H,T,d_k)
        cos_b = cos.view(num_phases, 1, 1, 1, self.d_k)
        sin_b = sin.view(num_phases, 1, 1, 1, self.d_k)
        q_all = q_un * cos_b + q_rot_half * sin_b                               # (P,B,H,T,d_k)

        # ── 3. 并行 read: r_τ = q_τ @ W_turn.T ──
        reads = torch.einsum("pbhtd,bhvd->pbhtv", q_all, W_cast)                # (P,B,H,T,d_v)

        # ── 4. 共振评分 + softmax（温度系数锐化，防早期梯度稀释）──
        scores = reads.norm(dim=-1) / (self.d_v ** 0.5)                         # (P,B,H,T)
        alpha = F.softmax(scores * self.phase_temperature, dim=0)               # (P,B,H,T)

        # ── 5. 加权融合 ──
        read = (alpha.unsqueeze(-1) * reads).sum(dim=0)                         # (B,H,T,d_v)

        # ── 6. 合头 + 输出投影 + 残差 ──
        merged = read.transpose(1, 2).reshape(B, T, self.H * self.d_v)
        o_w = self.o_projs_turn[layer_idx].weight.to(dtype=dtype)
        out = F.linear(merged, o_w)                                             # (B,T,D)
        scale = torch.sigmoid(self.read_scale_turn).to(dtype)

        # 诊断（供日志读取，detach 不传梯度）
        with torch.no_grad():
            self._last_alpha = alpha.detach()                                   # (P,B,H,T)

        return hidden_states + scale * out

    def write_from_content(
        self,
        W_turn_old: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        """程序性写入：W_turn^(t) = γ · R · W_turn^(t-1) + residual

        - R = 每轮 1 步的 RoPE 旋转（d_k 维），把旧 key 的"年龄"推进一格
        - residual = mean_t (v_t ⊗ k_t)，k/v 投影是 **独立** 的 k_proj_turn / v_proj_turn
          （v6.2 改，原来是复用 fact_interface.k_proj；诊断发现 fact 投影对 phrase 区分度不够）
        - γ = 固定 config 标量（默认 0.9），配合旋转共同提供"遗忘"

        W_turn_old: (B,H,d_v,d_k)；content: (B,T,D)；返回同形状 W_turn_new。
        """
        B, T, _ = content.shape
        dtype = W_turn_old.dtype
        c = content.to(dtype=dtype)

        # ── 写侧投影：独立 k_proj_turn / v_proj_turn（v6.2）──
        k_w = self.k_proj_turn.weight.to(dtype=dtype)
        v_w = self.v_proj_turn.weight.to(dtype=dtype)

        k = F.linear(c, k_w).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B,H,T,d_k)
        v = F.linear(c, v_w).view(B, T, self.H, self.d_v).transpose(1, 2)  # (B,H,T,d_v)
        k = F.normalize(k, dim=-1)

        # residual = mean outer product over T: (B,H,d_v,d_k)
        residual = torch.einsum("bhtv,bhtd->bhvd", v, k) / max(T, 1)

        # ── 旋转 W_old 一轮（age +1）──
        step_one = torch.ones((), device=W_turn_old.device, dtype=dtype)
        W_aged = rotate_W_turn(W_turn_old, step_one, self.inv_freq, inverse=False)

        # ── γ 衰减 + 叠加当前轮残影 ──
        return self.turn_gamma * W_aged + residual

    # ---- 诊断 ----

    def get_state_stats(self, W_turn: torch.Tensor) -> dict:
        """W_turn 范数 / 读强度。"""
        if W_turn.dim() == 4:
            W = W_turn[0]
        else:
            W = W_turn
        W_flat = W.reshape(W.shape[0], -1)
        w_norm = W_flat.norm(dim=-1).mean().item()
        return {
            "turn_read_scale": torch.sigmoid(self.read_scale_turn).item(),
            "W_turn_norm": w_norm,
            "turn_gamma": self.turn_gamma,
            "turn_phase_max": self.phase_max,
        }

    def get_phase_diagnostics(self) -> dict | None:
        """返回最近一次 read_layer 的相位分布统计。

        - turn_phase_entropy: softmax 熵均值，越小越锐利（训练中应从 log(P+1) 降到 <1.0）
        - turn_phase_argmax_mean: 所有位置 argmax 相位的均值，反映模型平均用到几轮前的信息
        """
        a = getattr(self, "_last_alpha", None)
        if a is None:
            return None
        # 熵: -Σ α log α, 均值 over (B,H,T)
        ent = -(a * (a.clamp_min(1e-12)).log()).sum(dim=0).mean().item()
        mode = a.argmax(dim=0).float().mean().item()
        return {
            "turn_phase_entropy": ent,
            "turn_phase_argmax_mean": mode,
            "turn_phase_max": self.phase_max,
        }
