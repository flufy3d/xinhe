"""
Backbone 接口 + MiniMind 适配器

BackboneBase 定义统一接口，任何 transformer 只需实现:
  - embed(input_ids) -> (B, T, D)
  - forward_blocks(hidden_states, attention_mask) -> (B, T, D)
  - get_lm_head() -> nn.Linear

将来换 Qwen-7B 只需新建 QwenBackbone(BackboneBase)。
"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import XinheConfig


class BackboneBase(ABC):
    """Backbone 统一接口"""

    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token ids → embeddings, shape (B, T, D)"""
        ...

    @abstractmethod
    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """transformer blocks 前向传播, (B, T, D) → (B, T, D)"""
        ...

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """返回 language model head (用于 logits 计算)"""
        ...

    @abstractmethod
    def get_hidden_size(self) -> int:
        ...


class MiniMindBackbone(nn.Module, BackboneBase):
    """
    包装 MiniMind 的 MiniMindForCausalLM 为统一 backbone 接口。

    加载预训练权重后冻结主干参数，通过 LoRA 注入少量可训练参数。
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

        # 加载预训练权重（如果路径存在）
        weights_path = Path(config.backbone_weights_path)
        if weights_path.exists():
            # MiniMind 权重可能是单个文件或目录
            if weights_path.is_dir():
                # 查找 .pth 文件
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
        """token ids → embeddings"""
        return self.model.model.embed_tokens(input_ids)

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        直接跑 MiniMind 的 transformer blocks。

        hidden_states: (B, T, D) — 已经包含状态 token 的嵌入序列
        attention_mask: (B, 1, T, T) — 自定义 attention mask（状态双向 + 内容因果）
        """
        # MiniMind 内部会计算 RoPE position embeddings
        # 需要传入正确的 position_ids 或让它自动生成
        seq_len = hidden_states.shape[1]
        device = hidden_states.device

        # 计算 RoPE 的 cos/sin（MiniMind 用 precompute_pos_cis）
        # 直接调用 MiniMind 内部的层
        past_key_values = [None] * len(self.model.model.layers)
        total_aux_loss = torch.tensor(0.0, device=device)

        for i, layer in enumerate(self.model.model.layers):
            # MiniMind 的 block.forward 需要:
            #   hidden_states, position_embeddings, past_key_value, use_cache, attention_mask
            # position_embeddings 是 (cos, sin) tuple
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

        # 最终 layer norm
        hidden_states = self.model.model.norm(hidden_states)

        return hidden_states

    def get_lm_head(self) -> nn.Module:
        """返回 MiniMind 的 lm_head"""
        return self.model.lm_head

    def get_hidden_size(self) -> int:
        return self._hidden_size
