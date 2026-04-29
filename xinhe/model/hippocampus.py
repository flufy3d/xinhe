"""
Hippocampus (v8) — 大一统短期记忆,纯 Delta Rule

生物类比:单一 W: (B, H, d_v, d_k) 对应海马体短期工作记忆。未来 Phase 2 的
Neocortex 模块(MLP 长期固化)将从 Hippocampus 蒸馏。

写入规则(无 γ 衰减):
  W_t = W_{t-1} + β_t · (v_t − W_{t-1} k_t) ⊗ k_t^T

后端策略(见 delta_kernel.py):
  - 训练 (model.training=True):强制 torch _delta_parallel(FLA Triton backward
    在 bf16 累加上有 5-25% 梯度幅度误差)
  - 推理 (model.eval()):按 self._delta_backend(默认 auto → Linux+CUDA 优先 FLA)
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

        for lin in [self.k_proj, self.v_proj, *self.q_projs, *self.o_projs]:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, beta_bias_init)

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
        """纯 Delta Rule(无 γ 衰减):
          W_t = W_{t-1} + β_t · (v_t - W_{t-1} k_t) ⊗ k_t^T

        W_old: (B,H,d_v,d_k)；content: (B,T,D)；返回 W_new 同 W_old 形状。
        写 kernel 后端由 self._delta_backend 决定。"""
        B, T, _ = content.shape
        dtype = W_old.dtype
        c = content.to(dtype=dtype)
        k_w = self.k_proj.weight.to(dtype=dtype)
        v_w = self.v_proj.weight.to(dtype=dtype)
        b_w = self.beta_proj.weight.to(dtype=dtype)
        b_b = self.beta_proj.bias.to(dtype=dtype)

        k = F.linear(c, k_w).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B,H,T,d_k)
        v = F.linear(c, v_w).view(B, T, self.H, self.d_v).transpose(1, 2)  # (B,H,T,d_v)
        k = F.normalize(k, dim=-1)                                         # 抑制 W 范数爆炸
        beta = torch.sigmoid(F.linear(c, b_w, b_b)).transpose(1, 2)        # (B,H,T)

        # 训练强制 torch:FLA Triton backward 在 bf16 累加上引入 5-25% 梯度幅度
        # 误差(read_scale 尤其严重),长序列累积导致优化器收敛到读不出 W 的最优。
        # 推理用 self._delta_backend(默认 auto → Linux+CUDA 优先 FLA),forward
        # 与 torch 差 < 0.5%,精度可忽略,可享受 FLA 加速。
        backend = "torch" if self.training else self._delta_backend
        return delta_rule_write(W_old, k, v, beta, backend=backend)

    @staticmethod
    def _delta_parallel(W, k, v, beta) -> torch.Tensor:
        """Chunkwise 并行 Delta Rule (Yang et al. 2024),无 γ 衰减:
          W_t = W_{t-1} + β_t (v_t - W_{t-1} k_t) k_t^T

        闭式:
          W_T = W_0 + Σ_i β_i v'_i k_i^T
          其中 v'_i = v_i - W_0 k_i - Σ_{l<i} β_l (k_l·k_i) v'_l
        三角系统 (I + A_tril) V' = V_rhs,一次 solve_triangular。

        k: (B,H,T,d_k) 已 L2 归一;v: (B,H,T,d_v);beta: (B,H,T);W: (B,H,d_v,d_k)。
        """
        B, H, T, d_k = k.shape
        d_v = v.shape[-1]
        orig_dtype = W.dtype
        solve_dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype

        W_f = W.to(solve_dtype)
        k_f = k.to(solve_dtype)
        v_f = v.to(solve_dtype)
        beta_f = beta.to(solve_dtype)

        # V_rhs[i] = v_i - W_0 k_i
        W0k = torch.einsum("bhvd,bhtd->bhtv", W_f, k_f)
        V_rhs = v_f - W0k

        # A[i,l] = β_l · (k_i·k_l),严格下三角
        KK = torch.einsum("bhid,bhjd->bhij", k_f, k_f)
        A = beta_f.unsqueeze(-2) * KK
        tril_mask = torch.tril(
            torch.ones(T, T, device=k.device, dtype=torch.bool), diagonal=-1,
        )
        A_tril = A * tril_mask.to(A.dtype)

        eye_T = torch.eye(T, device=k.device, dtype=solve_dtype)
        L = eye_T + A_tril
        L_flat = L.reshape(B * H, T, T)
        Vrhs_flat = V_rhs.reshape(B * H, T, d_v)
        Vp_flat = torch.linalg.solve_triangular(
            L_flat, Vrhs_flat, upper=False, unitriangular=True,
        )
        Vp = Vp_flat.reshape(B, H, T, d_v)

        # W_T = W_0 + Σ_i β_i v'_i k_i^T
        weighted = beta_f.unsqueeze(-1) * Vp
        W_new = W_f + torch.einsum("bhtv,bhtd->bhvd", weighted, k_f)
        return W_new.to(orig_dtype)

    # ---- 诊断 ----

    def get_state_stats(self, W: torch.Tensor) -> dict:
        """W 范数 / 有效秩(按头平均)。"""
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

        return {
            "read_scale": torch.sigmoid(self.read_scale).item(),
            "W_norm": w_norm,
            "W_effective_rank": sum(eranks) / len(eranks),
        }

    def get_gamma_diagnostics(self) -> dict | None:
        """γ 已移除,返回 None。"""
        return None
