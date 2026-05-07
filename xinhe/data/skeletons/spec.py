"""骨架数据结构。

Skeleton.sequence 元素可以是：
  - 字符串：事件名（"A" / "B" / "C_prime" / ...）
  - DistractGroup 实例：表示一个 {En} 干扰段，按距离桶决定轮数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class DistractGroup:
    """{En} 干扰段。

    bucket_constraint: 限制只能采指定桶（"near"/"mid"/"far"/"very_far" 或 None 表示按默认分布）
    max_turns: 上限（如 S10 的 DG_short = max_turns=2）
    expansion: "short"(默认)= 现有短句模式;"paragraph" = 拼接 Congliu R1 长输出做长段
    paragraph_token_target: 仅 expansion="paragraph" 生效,目标 token 数(粗估,字符数 ≈ 1.5×)
    """
    bucket_constraint: Optional[str] = None
    max_turns: Optional[int] = None
    label: str = "DG"
    expansion: str = "short"
    paragraph_token_target: int = 300


SeqItem = Union[str, DistractGroup]


@dataclass
class Skeleton:
    id: str
    sequence: list[SeqItem]
    weight: float = 1.0
    description: str = ""

    def event_count(self) -> int:
        """非 distract 事件数。"""
        return sum(1 for s in self.sequence if isinstance(s, str))
