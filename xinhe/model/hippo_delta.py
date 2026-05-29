"""HippoDelta — Gated DeltaNet 快权重记忆(Phase 3,替换 TTT inner SGD)。

M ∈ R^{B,H,d_v,d_k} 线性快权重。
  write:  delta rule(含删旧关联 (I-βkkᵀ),对治 NIH distract 累加)
  retrieve: r = M·qᵀ(q 来自外部 QueryHead,Phase 2 接口对称)

训练纪律(见 [[feedback_delta_train_torch_infer_fla]]):
  - M 对 outer backprop **detach**(走 trainer TBPTT 边界 state.detach());
    W_k/W_v/W_β/τ_k 必须接梯度(经 r=M·qᵀ → M → write step,在 TBPTT 窗口内反传)。
  - k 归一 + learnable τ_k 严格保 β‖k‖²<1 → (g_t I-β k kᵀ) 收缩,M 不爆。
  - 训练强制 delta backend="torch";推理 Linux+CUDA 可 FLA。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_kernel import delta_rule_write
from .neural_memory_pair import AdaptiveRMSNorm


@dataclass
class HippoDeltaState:
    """单全局快权重 state。M fp32(数值稳定 + detach 纪律)。"""
    M: torch.Tensor | None = None          # (B, H, d_v, d_k)
    seq_index: int = 0

    def detach(self) -> "HippoDeltaState":
        return HippoDeltaState(self.M.detach() if self.M is not None else None, self.seq_index)

    def to(self, device) -> "HippoDeltaState":
        return HippoDeltaState(self.M.to(device) if self.M is not None else None, self.seq_index)


class HippoDelta(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_key: int = 256,
        d_value: int = 128,
        n_heads: int = 16,
        tau_k_init: float = 1.0,
        beta_bias_init: float = 0.0,
        gated: bool = False,
        delta_backend: str = "auto",
        spectral_norm_cap: float = 10.0,
    ):
        super().__init__()
        assert d_key % n_heads == 0 and d_value % n_heads == 0, \
            f"d_key({d_key})/d_value({d_value}) 必须能被 n_heads({n_heads}) 整除"
        self.d_model = d_model
        self.H = n_heads
        self.dk = d_key // n_heads
        self.dv = d_value // n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.gated = bool(gated)
        self._delta_backend = delta_backend
        self.spectral_norm_cap = float(spectral_norm_cap)

        self.pre_norm = AdaptiveRMSNorm(d_model)
        self.W_k = nn.Linear(d_model, d_key, bias=False)
        self.W_v = nn.Linear(d_model, d_value, bias=False)
        self.W_beta = nn.Linear(d_model, n_heads, bias=True)
        self.log_tau_k = nn.Parameter(torch.full((n_heads,), math.log(tau_k_init)))
        if self.gated:
            self.W_g = nn.Linear(d_model, n_heads, bias=True)

        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_beta.weight)
        nn.init.constant_(self.W_beta.bias, beta_bias_init)

        # 旁路诊断:detached scalar,trainer log 时才 .item()
        self.last_M_specnorm = torch.zeros(1)
        # probe 用:开 _spec_log_enabled 后,每次 write 把 specnorm 追加进 _spec_history
        # default off → 零热路径开销
        self._spec_history: list[float] = []
        self._spec_log_enabled: bool = False

    def blank_M(self, B: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(B, self.H, self.dv, self.dk, device=device, dtype=dtype)

    def _project(self, x: torch.Tensor):
        """content x(B,T,d_model) → k(归一/τ), v, beta(保 β‖k‖²<1), g(可选)。"""
        c = self.pre_norm(x)
        B, T, _ = c.shape
        k = self.W_k(c).view(B, T, self.H, self.dk).transpose(1, 2)   # (B,H,T,dk)
        v = self.W_v(c).view(B, T, self.H, self.dv).transpose(1, 2)   # (B,H,T,dv)
        k = F.normalize(k, dim=-1)                                    # ‖k‖=1
        tau = self.log_tau_k.exp().clamp_min(1e-3)                    # (H,)
        k = k / tau.view(1, self.H, 1, 1)                             # ‖k‖²=1/τ²
        beta = torch.sigmoid(self.W_beta(c)).transpose(1, 2)          # (B,H,T)∈(0,1)
        # 硬约束 β·‖k‖²<1 ⇔ β<τ²:防 (I-βkkᵀ) 不收缩 → M 爆炸
        beta = torch.minimum(beta, (tau ** 2 * (1.0 - 1e-3)).view(1, self.H, 1))
        g = torch.sigmoid(self.W_g(c)).transpose(1, 2) if self.gated else None
        return k, v, beta, g

    def write(self, x: torch.Tensor, state: HippoDeltaState | None) -> HippoDeltaState:
        """delta rule 写:M_t = M_{t-1}(g I - β k kᵀ) + β v kᵀ。返回新 state(M fp32)。"""
        if state is None or state.M is None:
            M_prev = self.blank_M(x.shape[0], x.device)
            seq = 0
        else:
            M_prev = state.M
            seq = state.seq_index
        k, v, beta, g = self._project(x)
        backend = "torch" if self.training else self._delta_backend
        M_new = delta_rule_write(
            M_prev.float(), k.float(), v.float(), beta.float(),
            g=(g.float() if g is not None else None), backend=backend,
        )
        with torch.no_grad():
            self.last_M_specnorm = torch.linalg.matrix_norm(
                M_new[0].float(), ord=2,
            ).max().detach()
            if self._spec_log_enabled:
                self._spec_history.append(float(self.last_M_specnorm.item()))
        return HippoDeltaState(M=M_new, seq_index=seq + x.shape[1])

    def retrieve(self, M: torch.Tensor | None, q: torch.Tensor) -> torch.Tensor:
        """r = M·qᵀ。q:(B,n_q,d_key) 外部 QueryHead;返回 (B,n_q,d_value)。
        M=None(空 state)→ 零(NM-zero sanity:首 turn / 空记忆读出正好 0)。"""
        B, n_q, _ = q.shape
        if M is None:
            return torch.zeros(B, n_q, self.d_value, device=q.device, dtype=q.dtype)
        qh = q.view(B, n_q, self.H, self.dk).transpose(1, 2)          # (B,H,n_q,dk)
        qh = F.normalize(qh, dim=-1)                                  # 与写侧 L2 对齐
        r = torch.einsum("bhnd,bhvd->bhnv", qh, M.to(qh.dtype))       # (B,H,n_q,dv)
        return r.transpose(1, 2).reshape(B, n_q, self.d_value)
