"""
MiniMind Backbone 适配器

包装 MiniMind 的 MiniMindForCausalLM 为统一 backbone 接口。
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .backbone import BackboneBase
from .config import XinheConfig


class MiniMindBackbone(nn.Module, BackboneBase):
    """
    MiniMind backbone: 64M 参数，用于机制验证。

    通过 sys.path 动态导入 MiniMind 仓库代码，
    加载预训练权重后冻结主干参数。
    """

    def __init__(self, config: XinheConfig):
        nn.Module.__init__(self)
        self.config = config
        self._hidden_size = config.hidden_size

        # 动态导入 MiniMind 模块
        minimind_path = str(Path(config.backbone_model_path).resolve())
        if minimind_path not in sys.path:
            sys.path.insert(0, minimind_path)

        from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

        # 创建 MiniMind 模型
        mm_config = MiniMindConfig()
        self.model = MiniMindForCausalLM(mm_config)

        # 加载预训练权重
        weights_path = Path(config.backbone_weights_path)
        if weights_path.exists():
            if weights_path.is_dir():
                pth_files = list(weights_path.glob("*.pth"))
                if pth_files:
                    state_dict = torch.load(pth_files[0], map_location="cpu", weights_only=False)
                    self.model.load_state_dict(state_dict, strict=False)
            elif weights_path.suffix == ".pth":
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
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.model.model.layers):
            position_embeddings = self.model.model.pos_cis(hidden_states, seq_len=seq_len)

            hidden_states, past_kv, aux_loss = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values[i],
                use_cache=False,
                attention_mask=attention_mask,
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.model.model.norm(hidden_states)
        return hidden_states

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    def get_hidden_size(self) -> int:
        return self._hidden_size
