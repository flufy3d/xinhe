"""
MiniMind Backbone 适配器

包装 MiniMind 的 MiniMindForCausalLM 为统一 backbone 接口。
架构代码已内置在 model_minimind.py 中，无需外部依赖。
"""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .backbone import BackboneBase
from .config import XinheConfig
from .model_minimind import MiniMindForCausalLM, MiniMindConfig


class MiniMindBackbone(nn.Module, BackboneBase):
    """
    MiniMind backbone: 64M 参数，用于机制验证。
    """

    def __init__(self, config: XinheConfig):
        nn.Module.__init__(self)
        self.config = config
        self._hidden_size = config.hidden_size

        # 创建 MiniMind 模型
        mm_config = MiniMindConfig()
        self.model = MiniMindForCausalLM(mm_config)

        # 加载预训练权重
        weights_path = Path(config.backbone_weights_path)
        if config.backbone_weights_path:
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"MiniMind 权重文件不存在: {weights_path}. "
                    f"请检查 configs/minimind.yaml 中的 backbone.weights_path。"
                )
            if weights_path.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(str(weights_path))
            else:
                state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)

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
        seq_len = hidden_states.shape[1]
        device = hidden_states.device

        past_key_values = [None] * len(self.model.model.layers)

        for i, layer in enumerate(self.model.model.layers):
            position_embeddings = (
                self.model.model.freqs_cos[:seq_len],
                self.model.model.freqs_sin[:seq_len],
            )

            hidden_states, past_kv = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values[i],
                use_cache=False,
                attention_mask=attention_mask,
            )

        hidden_states = self.model.model.norm(hidden_states)
        return hidden_states

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    def get_hidden_size(self) -> int:
        return self._hidden_size
