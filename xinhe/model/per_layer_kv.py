"""per-layer K/V persistent memory(paper Titans MAC 严格形态)。

每个 full_attention 层独立的 K_pers / V_pers,不经过 q/k/v 投影,
直接拼接到该层 attention 的 K/V cache 前面(paper Eq 11-13)。

形状:(N_p, n_kv_heads, head_dim) — 与 GQA 对齐。
NoPE:K_pers 不应用 RoPE(persistent 是位置无关的),在 apply_rotary_pos_emb 之后才 cat。
"""
import torch
import torch.nn as nn


class PerLayerPersistentKV(nn.Module):
    """每层独立 K_pers / V_pers,nn.Parameter 容器。"""

    def __init__(
        self,
        n_persistent: int,
        n_kv_heads: int,
        head_dim: int,
        scale_init: float = 0.02,
    ):
        super().__init__()
        self.n_persistent = n_persistent
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # 小随机初始化(paper 没明说,常见 0.02 量级)
        self.K_pers = nn.Parameter(torch.randn(n_persistent, n_kv_heads, head_dim) * scale_init)
        self.V_pers = nn.Parameter(torch.randn(n_persistent, n_kv_heads, head_dim) * scale_init)

    def expand_for_attention(self, B: int, dtype, device):
        """返回 (B, n_kv_heads, N_p, head_dim) 的 K_pers, V_pers,
        匹配 attention 内 transpose 后的 K/V 形状 (B, n_kv, T, d)。"""
        K = self.K_pers.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)  # (B, n_kv, N_p, d)
        V = self.V_pers.unsqueeze(0).expand(B, -1, -1, -1).transpose(1, 2)
        return K.to(dtype=dtype, device=device), V.to(dtype=dtype, device=device)
