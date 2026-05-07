"""XinheQwen3FullAttention — 包装 Qwen3_5Attention 注入 per-layer K/V persistent memory。

paper Titans MAC Eq 11-13 严格形态:
  K = concat([K_pers (NoPE) ; K_real (RoPE applied)])
  V = concat([V_pers ; V_real])
  query 不变,attention 在扩展后的 K/V 上做。

实现策略:复刻 Qwen3_5Attention.forward 流程,在 RoPE 之后、attention_interface 之前
插入 K_pers/V_pers 拼接。复用所有原 module(q_proj/k_proj/v_proj/o_proj/q_norm/k_norm),
让 LoRA 注入对 q/k/v/o 仍然生效。

q_proj 注意:Qwen3_5 的 q_proj 输出 head_dim*2*n_heads,chunk 后前半是 query 后半是 gate
(attn_output_gate)。LoRA 注入到 q_proj 自动覆盖 gate 投影。
"""
import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .per_layer_kv import PerLayerPersistentKV


class XinheQwen3FullAttention(nn.Module):
    """包装原 Qwen3_5Attention,加 per-layer K/V persistent memory 拼接。"""

    def __init__(self, original: nn.Module, n_persistent: int):
        super().__init__()
        # 复用原 module(LoRA 已注入到这些 Linear 上)
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        # 复用原配置 / 元数据
        self.config = original.config
        self.head_dim = original.head_dim
        self.num_key_value_groups = original.num_key_value_groups
        self.scaling = original.scaling
        self.attention_dropout = original.attention_dropout
        self.layer_idx = original.layer_idx
        # KV head 数(GQA)— 从 config 取
        n_kv_heads = self.config.num_key_value_heads
        # per-layer K/V
        self.persistent = PerLayerPersistentKV(
            n_persistent=n_persistent,
            n_kv_heads=n_kv_heads,
            head_dim=self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]  # (B, T)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q_proj 的输出 chunk 出 query 和 gate(attn_output_gate)
        # q_proj 形状: (B, T, n_heads * head_dim * 2)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # === XINHE per-layer K/V 拼接(在 RoPE 之后,K_pers 是 NoPE)===
        N_p = self.persistent.n_persistent
        if N_p > 0:
            B = query_states.size(0)
            K_pers, V_pers = self.persistent.expand_for_attention(
                B, dtype=key_states.dtype, device=key_states.device,
            )  # (B, n_kv, N_p, d)
            key_states = torch.cat([K_pers, key_states], dim=-2)
            value_states = torch.cat([V_pers, value_states], dim=-2)

            # attention_mask 在 K 维(最后一维)前 prepend N_p 列 0(全可见 — 加性 mask 0 = no penalty)
            if attention_mask is not None:
                # attention_mask shape: (B, 1, T_q, T_k) 或 (B, n_h, T_q, T_k)
                pad_shape = list(attention_mask.shape)
                pad_shape[-1] = N_p
                pad = torch.zeros(
                    pad_shape, dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([pad, attention_mask], dim=-1)
        # === END XINHE ===

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward,
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
