"""
Hippocampus (v7) — 内容驱动的大一统自适应短期记忆

生物类比：单一 W: (B, H, d_v, d_k) 对应海马体短期工作记忆。不同 head 自发学出
不同时间尺度（快代谢 head 承接废话、慢代谢 head 承接事实）。未来 Phase 2 的
Neocortex 模块（MLP 长期固化）将从 Hippocampus 蒸馏。

相比 v5c/v6 FactInterface 的核心变化：
  + head_decay_logits: Parameter(H,) — per-head 寿命先验，初值 logit(linspace(0.8, 0.999, H))
  + time_shift: Linear(hidden, H) — 内容驱动的 γ 偏移，weight/bias 均零初始化
  × γ 加入 Delta Rule：W_t = γ_{h,t}·W_{t-1} + β_t·(v_t - W_{t-1}·k_t)⊗k_t^T
  × _delta_parallel 支持 per-head per-token γ（对数空间前缀积 + 三角求解）
  - 废除 W_fact/W_turn 分流，回到单一 W；时序不再由 RoPE 编码

参数量（H=16, d_k=d_v=64, hidden=1024, n_layers=6）≈ 14.7M + 16 (head_decay_logits) + 16.4k (time_shift)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_kernel import delta_rule_write


class Hippocampus(nn.Module):
    """大一统短期记忆。W: (B,H,d_v,d_k)，线性读 + 带 γ 的 Delta Rule 写。"""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 16,
        head_dim: int = 64,
        n_layers: int = 24,
        read_scale_init: float = -5.0,
        beta_bias_init: float = 0.0,
        gamma_head_init_low: float = 0.8,
        gamma_head_init_high: float = 0.999,
        delta_backend: str = "auto",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H = n_heads
        self.d_k = head_dim
        self.d_v = head_dim
        self._delta_backend = delta_backend

        # ── 读侧: 每层独立 q/o 投影（保留 per-layer hook 语义） ──
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, n_heads * head_dim, bias=False)
            for _ in range(n_layers)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(n_heads * head_dim, hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_scale = nn.Parameter(torch.tensor(read_scale_init))

        # ── 写侧: 全局共享 K/V/β 投影（作用于 final content_output） ──
        self.k_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, n_heads, bias=True)

        # ── v7 新增: per-head γ_h 先验 + 内容驱动的 Δγ ──
        # head_decay_logits: σ 后 linspace(low, high) 跨 H 个 head
        init_gamma = torch.linspace(gamma_head_init_low, gamma_head_init_high, n_heads)
        # σ^-1(y) = log(y/(1-y))
        init_logits = torch.log(init_gamma / (1.0 - init_gamma))
        self.head_decay_logits = nn.Parameter(init_logits)           # (H,)

        # time_shift: Linear(hidden, H)，weight+bias 均零初始化
        # 初期 γ = σ(θ_h + 0) = σ(θ_h)，完全依赖先验；训练中学出 "废话 → 减寿" 偏移
        self.time_shift = nn.Linear(hidden_size, n_heads, bias=True)
        nn.init.zeros_(self.time_shift.weight)
        nn.init.zeros_(self.time_shift.bias)

        for lin in [self.k_proj, self.v_proj, *self.q_projs, *self.o_projs]:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_bias_init)

        # 诊断用：最近一次 write_from_content 的 γ（无梯度）
        self._last_gamma: torch.Tensor | None = None

    # ---- 状态操作 ----

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """空白 W: (B, H, d_v, d_k)。零初始化保证空态读出正好为 0。"""
        if device is None:
            device = self.read_scale.device
        dtype = self.read_scale.dtype
        return torch.zeros(
            batch_size, self.H, self.d_v, self.d_k,
            device=device, dtype=dtype,
        )

    def read_layer(
        self,
        hidden_states: torch.Tensor,
        W: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """线性读：h_out = h + scale · merge_heads(q @ W)。
        hidden_states: (B,T,D)；W: (B,H,d_v,d_k)；返回 (B,T,D)。
        无 phase 搜索、无 softmax 竞争，依赖 k_proj 学出 content addressing。"""
        B, T, _ = hidden_states.shape
        dtype = hidden_states.dtype
        W_cast = W.to(dtype=dtype, device=hidden_states.device)
        q_w = self.q_projs[layer_idx].weight.to(dtype=dtype)
        o_w = self.o_projs[layer_idx].weight.to(dtype=dtype)

        q = F.linear(hidden_states, q_w)                           # (B,T,H*d_k)
        q = q.view(B, T, self.H, self.d_k).transpose(1, 2)         # (B,H,T,d_k)
        q = F.normalize(q, dim=-1)                                 # L2，匹配写侧
        read = torch.einsum("bhtd,bhvd->bhtv", q, W_cast)          # (B,H,T,d_v)
        merged = read.transpose(1, 2).reshape(B, T, self.H * self.d_v)
        out = F.linear(merged, o_w)                                # (B,T,D)
        scale = torch.sigmoid(self.read_scale).to(dtype)
        return hidden_states + scale * out

    def write_from_content(
        self,
        W_old: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        """Delta Rule 写（带 γ 衰减）：
          W_t = γ_{h,t}·W_{t-1} + β_t · (v_t - W_{t-1} k_t) ⊗ k_t^T

        γ_{h,t} = σ(head_decay_logits_h + time_shift(x_t)_h)  —— per-head × per-token
        W_old: (B,H,d_v,d_k)；content: (B,T,D)；返回 W_new 同 W_old 形状。

        写 kernel 后端由 self._delta_backend 决定（auto/fla/torch）。外层 per-segment
        gradient checkpoint 由 XinheModel.forward 按 config.per_segment_checkpoint 控制；
        本函数不再内层 checkpoint（FLA 后端有自带 backward；torch 后端由外层 ckpt 兜底）。"""
        B, T, _ = content.shape
        dtype = W_old.dtype
        c = content.to(dtype=dtype)
        k_w = self.k_proj.weight.to(dtype=dtype)
        v_w = self.v_proj.weight.to(dtype=dtype)
        b_w = self.beta_proj.weight.to(dtype=dtype)
        b_b = self.beta_proj.bias.to(dtype=dtype)
        ts_w = self.time_shift.weight.to(dtype=dtype)
        ts_b = self.time_shift.bias.to(dtype=dtype)
        hd_logits = self.head_decay_logits.to(dtype=dtype)

        k = F.linear(c, k_w).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B,H,T,d_k)
        v = F.linear(c, v_w).view(B, T, self.H, self.d_v).transpose(1, 2)  # (B,H,T,d_v)
        k = F.normalize(k, dim=-1)                                         # 抑制 W 范数爆炸
        beta = torch.sigmoid(F.linear(c, b_w, b_b)).transpose(1, 2)        # (B,H,T)

        # ── v7: 计算 per-head per-token γ ──
        # time_shift(x_t): (B, T, H)，加上 (H,) 先验后转置到 (B, H, T)
        ts = F.linear(c, ts_w, ts_b)                                       # (B,T,H)
        gamma = torch.sigmoid(hd_logits + ts).transpose(1, 2)              # (B,H,T)

        W = delta_rule_write(W_old, k, v, beta, gamma, backend=self._delta_backend)

        # 诊断快照（detached, no grad）
        self._last_gamma = gamma.detach()
        return W

    @staticmethod
    def _delta_parallel(W, k, v, beta, gamma) -> torch.Tensor:
        """Chunkwise 并行 Delta Rule (Yang et al. 2024) + per-head per-token γ 衰减。

        数学推导（γ 版本）：
          W_t = γ_t W_{t-1} + β_t (v_t - W_{t-1} k_t) k_t^T
          ⇒ W_T = G_T W_0 + Σ_{i=1}^T (G_T/G_i) β_i v'_i k_i^T
            其中 v'_i = v_i - G_{i-1} W_0 k_i - Σ_{l<i} β_l (G_{i-1}/G_l) (k_l·k_i) v'_l
                 G_i = γ_1·γ_2·...·γ_i（前缀积，G_0=1）

          写成三角系统 (I + A_tril) V' = V_rhs，一次 solve_triangular 解完。

        k: (B,H,T,d_k) 已 L2 归一；v: (B,H,T,d_v)；beta: (B,H,T)；
        gamma: (B,H,T) per-head per-token ∈ (0,1)；W: (B,H,d_v,d_k)。
        返回同形状 W_new。显存 O(T²) 但 T≤256 可控。

        数值稳定：log 空间 cumsum 避免前缀积下溢；solve 升 fp32 避 bf16 精度损失。
        """
        B, H, T, d_k = k.shape
        d_v = v.shape[-1]
        orig_dtype = W.dtype
        solve_dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype

        # 全量 fp32，最后降回 orig_dtype
        gamma_f = gamma.to(solve_dtype)
        W_f = W.to(solve_dtype)
        k_f = k.to(solve_dtype)
        v_f = v.to(solve_dtype)
        beta_f = beta.to(solve_dtype)

        # ── 前缀积（log 空间）──
        log_gamma = torch.log(gamma_f.clamp(min=1e-8))                     # (B,H,T)
        cum_log = torch.cumsum(log_gamma, dim=-1)                          # cum[i] = log(γ_1..γ_i)
        # cum_log_im1[i] = log(G_{i-1}) = log(γ_1..γ_{i-1})，i=0 时为 log(1)=0
        cum_log_im1 = F.pad(cum_log[..., :-1], (1, 0), value=0.0)          # (B,H,T)

        # ratio[i,l] = G_{i-1}/G_l = exp(cum_log_im1[i] - cum_log[l])，仅 l<i 有效
        # 上三角（l≥i）log_ratio 会是大正数，exp 溢出 inf，后续 0*inf=NaN 污染下三角。
        # 有效下三角区域 log_ratio ≤ 0（γ ≤ 1），clamp(max=0) 在下三角 no-op、
        # 把上三角安全钉为 exp(0)=1（反正会被 tril_mask 零掉）。
        log_ratio = (cum_log_im1.unsqueeze(-1) - cum_log.unsqueeze(-2)).clamp(max=0.0)
        ratio = torch.exp(log_ratio)                                       # (B,H,T,T) ∈ (0, 1]

        G_im1 = torch.exp(cum_log_im1)                                     # (B,H,T)
        G_T = torch.exp(cum_log[..., -1:])                                 # (B,H,1)

        # ── V_rhs[i] = v_i - G_{i-1} · W_0 k_i ──
        W0k = torch.einsum("bhvd,bhtd->bhtv", W_f, k_f)                    # (B,H,T,d_v)
        V_rhs = v_f - G_im1.unsqueeze(-1) * W0k

        # ── A[i,l] = β_l · ratio[i,l] · (k_i·k_l)，取严格下三角 ──
        KK = torch.einsum("bhid,bhjd->bhij", k_f, k_f)                     # (B,H,T,T)
        A = beta_f.unsqueeze(-2) * ratio * KK                              # (B,H,T,T)
        tril_mask = torch.tril(
            torch.ones(T, T, device=k.device, dtype=torch.bool), diagonal=-1,
        )
        A_tril = A * tril_mask.to(A.dtype)

        # L = I + A_tril, 单位下三角；solve L V' = V_rhs
        eye_T = torch.eye(T, device=k.device, dtype=solve_dtype)
        L = eye_T + A_tril
        L_flat = L.reshape(B * H, T, T)
        Vrhs_flat = V_rhs.reshape(B * H, T, d_v)
        Vp_flat = torch.linalg.solve_triangular(
            L_flat, Vrhs_flat, upper=False, unitriangular=True,
        )
        Vp = Vp_flat.reshape(B, H, T, d_v)

        # ── W_T = G_T · W_0 + Σ_i (G_T/G_i) · β_i · v'_i · k_i^T ──
        decay_to_T = torch.exp(cum_log[..., -1:] - cum_log)                # (B,H,T)
        weighted = (beta_f * decay_to_T).unsqueeze(-1) * Vp                # (B,H,T,d_v)
        W_new = G_T.unsqueeze(-1) * W_f + torch.einsum("bhtv,bhtd->bhvd", weighted, k_f)
        return W_new.to(orig_dtype)

    @staticmethod
    def _delta_loop(W, k, v, beta, gamma, T: int = None) -> torch.Tensor:
        """T 步顺序 Delta Rule（带 γ）。保留作为数学参考 / 单元测试对照用。

        gamma: (B,H,T) per-head per-token ∈ (0,1)
        """
        if T is None:
            T = k.shape[2]
        orig_dtype = W.dtype
        solve_dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype
        W = W.to(solve_dtype)
        k = k.to(solve_dtype)
        v = v.to(solve_dtype)
        beta = beta.to(solve_dtype)
        gamma = gamma.to(solve_dtype)

        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            g_t = gamma[:, :, t, None, None]
            b_t = beta[:, :, t, None, None]
            v_hat = torch.einsum("bhvd,bhd->bhv", W, k_t)
            error = v_t - v_hat
            W = g_t * W + b_t * torch.einsum("bhv,bhd->bhvd", error, k_t)
        return W.to(orig_dtype)

    # ---- 诊断 ----

    def get_state_stats(self, W: torch.Tensor) -> dict:
        """W 范数 / 有效秩（按头平均）+ γ 先验分布统计。"""
        if W.dim() == 4:
            W = W[0]                                                     # (H,d_v,d_k)
        W_flat = W.reshape(W.shape[0], -1)
        w_norm = W_flat.norm(dim=-1).mean().item()

        eranks = []
        for h in range(W.shape[0]):
            S = torch.linalg.svdvals(W[h].float())
            Sn = S / (S.sum() + 1e-10)
            Sn = Sn[Sn > 1e-10]
            if Sn.numel() == 0:
                eranks.append(0.0)
            else:
                eranks.append(torch.exp(-torch.sum(Sn * torch.log(Sn))).item())

        gamma_prior = torch.sigmoid(self.head_decay_logits)
        return {
            "read_scale": torch.sigmoid(self.read_scale).item(),
            "W_norm": w_norm,
            "W_effective_rank": sum(eranks) / len(eranks),
            "gamma_prior_min": gamma_prior.min().item(),
            "gamma_prior_max": gamma_prior.max().item(),
            "gamma_prior_mean": gamma_prior.mean().item(),
        }

    def get_gamma_diagnostics(self) -> dict | None:
        """最近一次 write_from_content 的 γ 分布（per-token × per-head 聚合）。

        返回 None 当尚未 forward 过；否则：
          gamma_token_mean / std / min / max —— batch 内所有 (B,H,T) γ 的统计
        """
        if self._last_gamma is None:
            return None
        g = self._last_gamma
        return {
            "gamma_token_mean": g.mean().item(),
            "gamma_token_std": g.std().item(),
            "gamma_token_min": g.min().item(),
            "gamma_token_max": g.max().item(),
        }
