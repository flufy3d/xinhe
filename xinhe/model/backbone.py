"""
Backbone 抽象接口

任何 transformer backbone 只需实现:
  - embed(input_ids) -> (B, T, D)
  - forward_blocks(hidden_states, attention_mask) -> (B, T, D)
  - get_lm_head() -> nn.Module
  - get_hidden_size() -> int
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
    ) -> torch.Tensor:
        """transformer blocks 前向传播, (B, T, D) -> (B, T, D)"""
        ...

    @abstractmethod
    def get_lm_head(self) -> nn.Module:
        """返回 language model head"""
        ...

    @abstractmethod
    def get_hidden_size(self) -> int:
        ...
