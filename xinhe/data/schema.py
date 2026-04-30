"""
统一数据 schema。

每条样本最终落 JSONL 时形如：

    {
      "sample_id": "uuid",
      "stage": "0" | "1",
      "skeleton_id": "S1" | None,
      "meta": {...},
      "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...",
         "train_loss": "true" | "lm_only" | "false",
         "value": [...] | None,
         "value_span": [[s, e], ...]   # 相对 content 的 char span
         "value_tier": "hard" | "soft" | None,
         "weight_per_span": float}
      ]
    }

四元一致性强制：value None ⇔ value_span [] ⇔ value_tier None ⇔ weight_per_span 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ValueTier(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class TrainLossMode(str, Enum):
    TRUE = "true"        # lm_weight=1.0, value tokens=weight_per_span
    LM_ONLY = "lm_only"  # lm_weight=0.3, value tokens 同步降权
    FALSE = "false"      # 整段不算 loss


def normalize_train_loss(v: Any) -> str:
    """兼容历史 bool / 新字符串：True→'true'，False→'false'，str 校验。"""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        if v in ("true", "lm_only", "false"):
            return v
        raise ValueError(f"非法 train_loss 字符串: {v!r}")
    raise TypeError(f"train_loss 类型非法: {type(v).__name__}")


@dataclass
class AssistantTurn:
    role: str  # 必须是 "assistant"
    content: str
    train_loss: str = "true"
    value: Optional[list[str]] = None
    value_span: list[list[int]] = field(default_factory=list)  # [[start,end],...]
    value_tier: Optional[str] = None  # "hard" | "soft" | None
    weight_per_span: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "role": "assistant",
            "content": self.content,
            "train_loss": self.train_loss,
            "value": self.value,
            "value_span": self.value_span,
            "value_tier": self.value_tier,
            "weight_per_span": self.weight_per_span,
        }
        return d


@dataclass
class UserTurn:
    content: str

    def to_dict(self) -> dict:
        return {"role": "user", "content": self.content}


@dataclass
class Sample:
    sample_id: str
    stage: str  # "0" or "1"
    skeleton_id: Optional[str]
    meta: dict
    conversations: list[dict]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "stage": self.stage,
            "skeleton_id": self.skeleton_id,
            "meta": self.meta,
            "conversations": self.conversations,
        }


# ── 校验 ──

class SchemaError(ValueError):
    """schema 校验失败（四元一致性 / 严格交替 / span 越界等）。"""


def validate_assistant_turn(turn: dict, *, content_len: int) -> None:
    """对单个 assistant turn 做四元一致性 + span 越界 + train_loss 合法性校验。"""
    if turn.get("role") != "assistant":
        raise SchemaError(f"role 不是 assistant: {turn.get('role')!r}")

    content = turn.get("content", "")
    if not isinstance(content, str):
        raise SchemaError(f"content 必须是 str，得到 {type(content).__name__}")

    # train_loss
    try:
        train_loss = normalize_train_loss(turn.get("train_loss", "true"))
    except (ValueError, TypeError) as e:
        raise SchemaError(str(e)) from e

    value = turn.get("value")
    spans = turn.get("value_span", []) or []
    tier = turn.get("value_tier")
    weight = float(turn.get("weight_per_span", 0.0) or 0.0)

    has_value = value is not None and len(value) > 0
    has_span = len(spans) > 0
    has_tier = tier is not None
    has_weight = weight > 0.0

    # 四元一致性：要么全有，要么全无
    flags = (has_value, has_span, has_tier, has_weight)
    if any(flags) and not all(flags):
        raise SchemaError(
            f"四元不一致 value={has_value} span={has_span} tier={has_tier} weight={has_weight}"
        )

    if has_value:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise SchemaError("value 必须是 list[str]")
        if tier not in ("hard", "soft"):
            raise SchemaError(f"value_tier 必须是 hard/soft，得到 {tier!r}")
        if not isinstance(spans, list) or not all(
            isinstance(s, (list, tuple)) and len(s) == 2 for s in spans
        ):
            raise SchemaError("value_span 必须是 list[[start,end]]")
        # span 越界 + 顺序检查
        for s, e in spans:
            if not (0 <= s < e <= content_len):
                raise SchemaError(f"span [{s},{e}] 越界（content_len={content_len}）")
        # train_loss=false 时 value 应为空（false 段不参与梯度）
        if train_loss == "false":
            raise SchemaError("train_loss=false 不应携带 value（无梯度的回合不需要打 value 权重）")


def validate_sample(sample: dict) -> None:
    """整条样本校验：

    - conversations 严格交替 user/assistant
    - assistant turn 四元一致 + span 不越界
    - meta.n_turns 与 conversations 长度一致
    """
    convs = sample.get("conversations", [])
    if not convs:
        raise SchemaError("conversations 为空")

    # 严格交替
    for i, turn in enumerate(convs):
        expected = "user" if i % 2 == 0 else "assistant"
        if turn.get("role") != expected:
            raise SchemaError(
                f"第 {i} 轮 role={turn.get('role')!r}，期望 {expected!r}（user/assistant 必须严格交替）"
            )

    # user/assistant 必须配对
    if len(convs) % 2 != 0:
        raise SchemaError(f"对话总数必须是偶数（user+assistant 配对），得到 {len(convs)}")

    # n_turns 一致性
    expected_turns = len(convs) // 2
    meta = sample.get("meta", {})
    if "n_turns" in meta and meta["n_turns"] != expected_turns:
        raise SchemaError(
            f"meta.n_turns={meta['n_turns']} 与实际轮数 {expected_turns} 不一致"
        )

    # 逐 assistant 校验
    for turn in convs:
        if turn.get("role") == "assistant":
            validate_assistant_turn(turn, content_len=len(turn.get("content", "")))
