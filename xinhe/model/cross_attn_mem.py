"""Cross-Attention Memory(v16 D 路径)— softmax 检索替代 HippoDelta 的线性 `M·qᵀ`

设计思路:
  HippoDelta `r = M @ q^T` 是所有 stored (K,V) 的加权线性求和(无 softmax 选择)。
  多 distractor 时:r ≈ mean(V_i),target 信号被淹。read 0% 验证不 viable。

CrossAttnMem 改用 explicit (K, V) buffer + softmax attention:
  - write(turn_x): mean-pool turn 的 hidden,投到 K/V,append 进 buffer(circular,max_slots 满后丢最旧)
  - retrieve(q): softmax(q · K^T · scale) · V — soft-argmax 选最像的一个 V

  scale = sqrt(d_key) 让 softmax 对相似度差异敏感(normalized cosine 后峰锐)。

State 模型 = HippoDelta 接口对齐(blank_state / write / retrieve / detach / to),
xinhe_model 可通过 config 切 mem_type 在两者间消融。
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_memory_pair import AdaptiveRMSNorm


@dataclass
class CrossAttnMemState:
    """显式 (K, V) buffer。fp32 保数值;detach 走 trainer TBPTT 纪律。"""
    K_buf: torch.Tensor | None = None     # (B, N, d_key) — L2 normalized
    V_buf: torch.Tensor | None = None     # (B, N, d_value)
    n_written: int = 0                    # cumulative write count(不超 max_slots)

    def detach(self) -> "CrossAttnMemState":
        return CrossAttnMemState(
            K_buf=self.K_buf.detach() if self.K_buf is not None else None,
            V_buf=self.V_buf.detach() if self.V_buf is not None else None,
            n_written=self.n_written,
        )

    def to(self, device) -> "CrossAttnMemState":
        return CrossAttnMemState(
            K_buf=self.K_buf.to(device) if self.K_buf is not None else None,
            V_buf=self.V_buf.to(device) if self.V_buf is not None else None,
            n_written=self.n_written,
        )


class CrossAttnMem(nn.Module):
    """Cross-attention memory:soft retrieval。"""

    def __init__(
        self,
        d_model: int,
        d_key: int = 256,
        d_value: int = 128,
        max_slots: int = 32,
        pool: str = "mean",       # write turn 池化:"mean" / "last"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.max_slots = int(max_slots)
        self.pool = pool

        self.pre_norm = AdaptiveRMSNorm(d_model)
        self.W_k = nn.Linear(d_model, d_key, bias=False)
        self.W_v = nn.Linear(d_model, d_value, bias=False)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

        # 旁路诊断
        self.last_attn_max = torch.zeros(1)
        self.last_n_slots = 0

    # 接口与 HippoDelta 对齐:blank_M 改 blank_state(语义更通用)
    def blank_state(self, B: int, device, dtype=torch.float32) -> CrossAttnMemState:
        return CrossAttnMemState(
            K_buf=torch.zeros(B, 0, self.d_key, device=device, dtype=dtype),
            V_buf=torch.zeros(B, 0, self.d_value, device=device, dtype=dtype),
            n_written=0,
        )

    def write(
        self,
        x: torch.Tensor,
        state: CrossAttnMemState | None,
    ) -> CrossAttnMemState:
        """写入 1 个 turn 的 (K, V) entry。

        x: (B, T, d_model) — turn 的 hidden states(write 层之前的内容段)
        pool 到 (B, d_model) → K/V 投影 → 拼到 buffer。
        """
        B, T, _ = x.shape
        if self.pool == "last":
            x_pool = x[:, -1, :]                        # (B, d_model)
        else:                                            # "mean"
            x_pool = x.mean(dim=1)                       # (B, d_model)
        c = self.pre_norm(x_pool)                        # (B, d_model)
        k_new = self.W_k(c).unsqueeze(1)                 # (B, 1, d_key)
        v_new = self.W_v(c).unsqueeze(1)                 # (B, 1, d_value)
        # key L2-normalize → q·k 走 cosine 路线(数值稳定)
        k_new = F.normalize(k_new, dim=-1)

        if state is None or state.K_buf is None or state.n_written == 0:
            K_buf = k_new.to(torch.float32)
            V_buf = v_new.to(torch.float32)
            n = 1
        else:
            K_buf = torch.cat([state.K_buf, k_new.to(torch.float32)], dim=1)
            V_buf = torch.cat([state.V_buf, v_new.to(torch.float32)], dim=1)
            n = state.n_written + 1
            # 满后 circular:丢最旧
            if K_buf.shape[1] > self.max_slots:
                K_buf = K_buf[:, -self.max_slots:]
                V_buf = V_buf[:, -self.max_slots:]
                n = min(n, self.max_slots)
        self.last_n_slots = K_buf.shape[1]
        return CrossAttnMemState(K_buf=K_buf, V_buf=V_buf, n_written=n)

    def retrieve(
        self,
        state: CrossAttnMemState | None,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """softmax cross-attention 检索。

        q: (B, n_q, d_key)
        返回:(B, n_q, d_value)

        空 state → 0(与 HippoDelta 对齐:NM-zero / 空记忆读出 0)。
        """
        B, n_q, _ = q.shape
        if state is None or state.K_buf is None or state.n_written == 0:
            return torch.zeros(B, n_q, self.d_value, device=q.device, dtype=q.dtype)

        K_buf = state.K_buf.to(q.dtype)                  # (B, N, d_key)
        V_buf = state.V_buf.to(q.dtype)                  # (B, N, d_value)

        # 查询 L2 normalize(与 K 对齐 → q·K ∈ [-1, 1])
        q_norm = F.normalize(q, dim=-1)

        # 关键:用 sqrt(d_key) 放大对比(否则 cosine 差异微小,softmax 太平)
        scale = math.sqrt(float(self.d_key))
        scores = torch.bmm(q_norm, K_buf.transpose(1, 2)) * scale      # (B, n_q, N)
        attn = F.softmax(scores, dim=-1)
        with torch.no_grad():
            self.last_attn_max = attn.max().detach()

        r = torch.bmm(attn, V_buf)                                       # (B, n_q, d_value)
        return r
