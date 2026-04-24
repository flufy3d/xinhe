"""
DualState — 心核 v6 双流记忆状态载体。

W_fact: Delta Rule 联想记忆（绝对语义空间），由 FactInterface 写入
W_turn: 自旋时序罗盘（相对时空），由 TurnInterface 写入（程序性，写侧无可学参数）

两个张量同形状 (B, H, d_v, d_k)，并行载体；hook 时并行注入 full_attention 层前。

对外提供 .detach() / .to() 让 trainer 的 `state = state.detach()` 不变形状。
"""
from typing import NamedTuple
import torch


class DualState(NamedTuple):
    """双流持久状态 (W_fact, W_turn)。"""
    W_fact: torch.Tensor    # (B, H, d_v, d_k)
    W_turn: torch.Tensor    # (B, H, d_v, d_k)

    def detach(self) -> "DualState":
        return DualState(self.W_fact.detach(), self.W_turn.detach())

    def to(self, device) -> "DualState":
        return DualState(self.W_fact.to(device), self.W_turn.to(device))

    def clone(self) -> "DualState":
        return DualState(self.W_fact.clone(), self.W_turn.clone())

    @property
    def shape(self):
        # 保持与 legacy 单 W 代码 state.shape 兼容（部分旧 eval 脚本会 peek shape）
        return self.W_fact.shape

    @property
    def device(self):
        return self.W_fact.device

    @property
    def dtype(self):
        return self.W_fact.dtype

    @property
    def requires_grad(self) -> bool:
        return bool(self.W_fact.requires_grad or self.W_turn.requires_grad)
