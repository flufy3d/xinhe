"""
Backbone 抽象接口

任何 transformer backbone 只需实现:
  - embed(input_ids) -> (B, T, D)
  - forward_blocks(hidden_states, attention_mask, position_ids, layer_hook) -> (B, T, D)
  - get_lm_head() -> nn.Module
  - get_hidden_size() -> int
  - get_num_layers() -> int
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BackboneBase(ABC):
    """Backbone 统一接口"""

    @abstractmethod
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token ids -> embeddings, shape (B, T, D)"""
        ...

    @abstractmethod
    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[callable] = None,
    ) -> torch.Tensor:
        """transformer blocks 前向传播, (B, T, D) -> (B, T, D)

        position_ids: (1, T) 自定义位置索引 (用于 RoPE)。None 时默认 0..T-1。
        layer_hook: 可选回调 (hidden_states, layer_idx) → hidden_states，每层之前调用。
        """
        ...

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """返回 language model head"""
        ...

    @abstractmethod
    def get_hidden_size(self) -> int:
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """返回 transformer 层数"""
        ...

    def get_hook_layer_indices(self) -> list[int]:
        """返回需要 layer_hook 的层索引。默认全部层。子类可覆盖。"""
        return list(range(self.get_num_layers()))
