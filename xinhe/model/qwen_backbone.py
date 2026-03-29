"""
Qwen Backbone 适配器

包装 HuggingFace transformers 的 Qwen 模型为统一 backbone 接口。
支持 Qwen3-0.6B 及同系列模型。
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .backbone import BackboneBase
from .config import XinheConfig


class QwenBackbone(nn.Module, BackboneBase):
    """
    Qwen backbone: 通过 transformers AutoModel 加载。

    相比 MiniMind，Qwen 语言能力更强，用于验证 state 机制的实际效果。
    """

    def __init__(self, config: XinheConfig):
        nn.Module.__init__(self)
        self.config = config

        # 加载 Qwen 模型，device_map="auto" 自动分配到多卡
        self.model = AutoModelForCausalLM.from_pretrained(
            config.backbone_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self._hidden_size = self.model.config.hidden_size

        # 冻结主干参数
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Qwen 使用 RoPE，需要构建 position embeddings
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        # 逐层跑 transformer blocks（多卡时跟随 layer 设备移动）
        for layer in self.model.model.layers:
            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_device)
            position_embeddings = (
                position_embeddings[0].to(layer_device),
                position_embeddings[1].to(layer_device),
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.model.norm(hidden_states)
        return hidden_states

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    def get_hidden_size(self) -> int:
        return self._hidden_size
