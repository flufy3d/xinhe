"""
FactInterface (v6) — Delta Rule 联想记忆矩阵（W_fact，双流中的绝对语义空间）

v6 双流：与 TurnInterface (W_turn, 自旋时序罗盘) 并行注入 full_attention 层前。
FactInterface 继承自 v5c 设计，负责稳定的事实/语义记忆；TurnInterface 负责跨轮时序。

v5c 减法续集：
  删 slot 机制 (state_emb, write_q/out, gate_proj, value_head,
     read_k_projs, read_v_projs, generate_read_kv, write_iterations)
  只留 W: (B, H, d_v, d_k) 多头外积矩阵
  读线性：out = q @ W^T，无 softmax
  写 Delta Rule：W_{t} = W_{t-1} + β_t · (v_t - W_{t-1} k_t) ⊗ k_t^T
  误差项 (v - Wk) 天然完成"相似 key 消除干扰 + 同 key 覆写"

参数量（H=16, d_k=d_v=64, hidden=1024, n_layers=6）≈ 14.7M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class FactInterface(nn.Module):
    """Delta-Rule 联想记忆。W: (B,H,d_v,d_k)，线性读 + Delta Rule 写。"""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 16,
        head_dim: int = 64,
        n_layers: int = 24,
        read_scale_init: float = -5.0,
        beta_bias_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H = n_heads
        self.d_k = head_dim
        self.d_v = head_dim

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
        用 F.linear 动态对齐投影权重 dtype，兼容 autocast / 非 autocast（val 路径）。"""
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
        """Delta Rule 写：W_t = W_{t-1} + β_t · (v_t - W_{t-1} k_t) ⊗ k_t^T。
        W_old: (B,H,d_v,d_k)；content: (B,T,D)；返回 W_new 同 W_old 形状。
        训练时用 torch.utils.checkpoint 包住循环以省显存（Phase C ep_len=16 必须）。
        用 F.linear 动态对齐 dtype，保证 val 路径 (无 autocast) 也不会 mat1/mat2 mismatch。"""
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

        if self.training:
            W = grad_checkpoint(
                self._delta_parallel, W_old, k, v, beta,
                use_reentrant=False,
            )
        else:
            W = self._delta_parallel(W_old, k, v, beta)
        return W

    @staticmethod
    def _delta_parallel(W, k, v, beta) -> torch.Tensor:
        """Chunkwise 并行 Delta Rule (Yang et al. 2024)，单 chunk = full T。

        数学推导：Delta Rule `W_t = W_{t-1} + β_t (v_t - W_{t-1} k_t) k_t^T`
        展开得 `W_T = W_0 + Σ_i β_i v'_i k_i^T`，其中
          v'_i = (v_i - W_0 k_i) - Σ_{j<i} β_j (k_j · k_i) v'_j
        写成三角系统 `(I + A_tril) V' = V - W_0 K^T`，A_tril[i,j]=β_j(k_i·k_j)
        用 `torch.linalg.solve_triangular` 一次解完，替代 Python 级 for-loop。

        k: (B,H,T,d_k) 已 L2 归一；v: (B,H,T,d_v)；beta: (B,H,T)；W: (B,H,d_v,d_k)。
        返回同形状 W_new。GPU 利用率大幅提升，显存 O(T²) 但 T≤256 可控。
        """
        B, H, T, d_k = k.shape
        d_v = v.shape[-1]

        # RHS: V_rhs[i] = v_i - W_0 k_i   shape (B,H,T,d_v)
        V_rhs = v - torch.einsum("bhvd,bhtd->bhtv", W, k)

        # Key-key 点积 KK[i,j] = k_i · k_j    shape (B,H,T,T)
        KK = torch.einsum("bhid,bhjd->bhij", k, k)

        # A[i,j] = β_j · (k_i · k_j)，取严格下三角 (j<i)
        A = beta.unsqueeze(-2) * KK                                        # (B,H,T,T)
        tril_mask = torch.tril(
            torch.ones(T, T, device=k.device, dtype=torch.bool), diagonal=-1,
        )
        A_tril = A * tril_mask.to(A.dtype)

        # L = I + A_tril, 单位下三角；solve L V' = V_rhs
        # CUDA triangular_solve 不支持 bf16/fp16，低精度时临时 upcast 到 fp32
        orig_dtype = A.dtype
        solve_dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype
        eye_T = torch.eye(T, device=k.device, dtype=solve_dtype)
        L = eye_T + A_tril.to(solve_dtype)                                 # 对角为 1
        L_flat = L.reshape(B * H, T, T)
        Vrhs_flat = V_rhs.to(solve_dtype).reshape(B * H, T, d_v)
        Vp_flat = torch.linalg.solve_triangular(
            L_flat, Vrhs_flat, upper=False, unitriangular=True,
        )
        Vp = Vp_flat.reshape(B, H, T, d_v).to(orig_dtype)

        # W_new = W_0 + Σ_i β_i v'_i k_i^T
        weighted_V = beta.unsqueeze(-1) * Vp                               # (B,H,T,d_v)
        W_new = W + torch.einsum("bhtv,bhtd->bhvd", weighted_V, k)
        return W_new

    @staticmethod
    def _delta_loop(W, k, v, beta, T: int = None) -> torch.Tensor:
        """T 步顺序 Delta Rule。保留作为数学参考 / 单元测试对照用。"""
        if T is None:
            T = k.shape[2]
        for t in range(T):
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            v_hat = torch.einsum("bhvd,bhd->bhv", W, k_t)
            error = v_t - v_hat
            b_t = beta[:, :, t, None, None]
            W = W + b_t * torch.einsum("bhv,bhd->bhvd", error, k_t)
        return W

    # ---- 诊断 ----

    def get_state_stats(self, W: torch.Tensor) -> dict:
        """W 范数 / 有效秩（按头平均）。"""
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
