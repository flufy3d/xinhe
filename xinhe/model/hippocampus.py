"""Hippocampus — per-layer Delta Rule 短期记忆(v19,移植自 delta-w-end / v5c)。

与单全局 QueryHead+MAC/MAL(hippo_delta.py)的本质区别:
- read:在**每个** full-attention hook 层注入 `h += σ(read_scale)·o_proj(q @ W)`,
  q/o 是**每层独立全参**投影(不受 LoRA 低秩限制),read 落在 backbone 自己的表示空间,
  由完整 backbone + lm_head 解码 → 这是 delta-w-end 达 95%+ recall 的关键(本 session 白盒
  证明单全局压缩 decode 不泛化,见 project_v17_macdisabled_verdict)。
- write:segment 末用 content_output 一次 Delta Rule 更新 W。

数值安全:k L2 归一(‖k‖=1)+ β=sigmoid∈(0,1) → β·‖k‖²<1 天然成立,(I-βkkᵀ) 收缩,W 不爆
(无需 hippo_delta 的 τ_k/谱范数硬约束)。仅保留 last_M_specnorm 旁路诊断。

后端:训练强制 torch(FLA bf16 backward 5-25% 梯度误差);推理 auto→FLA(见 delta_kernel)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_kernel import delta_rule_write


class Hippocampus(nn.Module):
    """per-layer Delta read。W:(B,H,d_v,d_k);线性读 + Delta Rule 写。"""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 16,
        head_dim: int = 128,
        n_layers: int = 24,
        read_scale_init: float = -3.0,
        beta_bias_init: float = 0.0,
        delta_backend: str = "auto",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.H = n_heads
        self.d_k = head_dim
        self.d_v = head_dim
        self._delta_backend = delta_backend

        # 读侧:每层独立全参 q/o 投影(per-layer hook 语义)
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, n_heads * head_dim, bias=False)
            for _ in range(n_layers)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(n_heads * head_dim, hidden_size, bias=False)
            for _ in range(n_layers)
        ])
        self.read_scale = nn.Parameter(torch.tensor(float(read_scale_init)))

        # 写侧:全局共享 K/V/β 投影(作用于 final content_output)
        self.k_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, n_heads, bias=True)

        for lin in [self.k_proj, self.v_proj, *self.q_projs, *self.o_projs]:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, float(beta_bias_init))

        # 旁路诊断(detached)
        self.last_M_specnorm = torch.zeros(1)

    def blank_state(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.read_scale.device
        dtype = self.read_scale.dtype
        return torch.zeros(
            batch_size, self.H, self.d_v, self.d_k, device=device, dtype=dtype,
        )

    def read_layer(
        self,
        hidden_states: torch.Tensor,
        W: torch.Tensor,
        layer_idx: int,
        read_scale_mul: float = 1.0,
    ) -> torch.Tensor:
        """线性读:h_out = h + read_scale_mul·σ(read_scale)·o_proj(q @ W)。
        read_scale_mul=0 → 关 read(NM-zero ablation:供 shortcut gap + 三口径 eval)。"""
        if read_scale_mul == 0.0:
            return hidden_states
        B, T, _ = hidden_states.shape
        dtype = hidden_states.dtype
        W_cast = W.to(dtype=dtype, device=hidden_states.device)
        q_w = self.q_projs[layer_idx].weight.to(dtype=dtype)
        o_w = self.o_projs[layer_idx].weight.to(dtype=dtype)

        q = F.linear(hidden_states, q_w)                           # (B,T,H*d_k)
        q = q.view(B, T, self.H, self.d_k).transpose(1, 2)         # (B,H,T,d_k)
        q = F.normalize(q, dim=-1)                                 # L2,匹配写侧
        read = torch.einsum("bhtd,bhvd->bhtv", q, W_cast)          # (B,H,T,d_v)
        merged = read.transpose(1, 2).reshape(B, T, self.H * self.d_v)
        out = F.linear(merged, o_w)                                # (B,T,D)
        scale = torch.sigmoid(self.read_scale).to(dtype) * float(read_scale_mul)
        return hidden_states + scale * out

    def write_from_content(self, W_old: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """Delta Rule:W_t = W_{t-1} + β_t·(v_t - W_{t-1}k_t)⊗k_t^T。"""
        B, T, _ = content.shape
        dtype = W_old.dtype
        c = content.to(dtype=dtype)
        k_w = self.k_proj.weight.to(dtype=dtype)
        v_w = self.v_proj.weight.to(dtype=dtype)
        b_w = self.beta_proj.weight.to(dtype=dtype)
        b_b = self.beta_proj.bias.to(dtype=dtype)

        k = F.linear(c, k_w).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B,H,T,d_k)
        v = F.linear(c, v_w).view(B, T, self.H, self.d_v).transpose(1, 2)  # (B,H,T,d_v)
        k = F.normalize(k, dim=-1)                                         # ‖k‖=1 → β‖k‖²<1
        beta = torch.sigmoid(F.linear(c, b_w, b_b)).transpose(1, 2)        # (B,H,T)∈(0,1)

        backend = "torch" if self.training else self._delta_backend
        W_new = delta_rule_write(W_old.float(), k.float(), v.float(), beta.float(), backend=backend)
        W_new = W_new.to(dtype)
        with torch.no_grad():
            self.last_M_specnorm = torch.linalg.matrix_norm(
                W_new[0].float(), ord=2,
            ).max().detach()
        return W_new
