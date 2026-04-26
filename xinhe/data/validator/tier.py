"""Hard / Soft / Reject 分级。

输入：value surface（事件声明的）+ assistant content + 可选 relation。
输出：tier ∈ {"hard", "soft", "reject"} + 在 content 中实际出现的 char span（若 hard/soft）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from xinhe.data.validator.aliases import (
    get_aliases,
    soft_eligible_for_value,
)
from xinhe.data.validator.normalize import fold, fold_keep_id


class TierVerdict(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    REJECT = "reject"


@dataclass
class TierResult:
    verdict: TierVerdict
    span: Optional[tuple[int, int]] = None       # 在 content 中的 char span（hard/soft 才有）
    matched_surface: Optional[str] = None         # 实际命中的 surface（soft 时是 alias）
    reason: str = ""


def _find_substring(content: str, surface: str) -> Optional[tuple[int, int]]:
    if not surface:
        return None
    idx = content.find(surface)
    if idx < 0:
        return None
    return (idx, idx + len(surface))


def _find_substring_normalized(content: str, surface: str) -> Optional[tuple[int, int]]:
    """折叠后查找 —— 找到则映射回原 content 中的 char span。

    简化策略：在原 content 上滑动窗口（长度 = len(surface) ± 2），任一窗口的 fold 与 surface 的 fold 相等即命中。
    """
    s_norm = fold(surface)
    if not s_norm:
        return None
    L = len(surface)
    for delta in (0, -1, 1, -2, 2):
        win_len = L + delta
        if win_len <= 0:
            continue
        for i in range(len(content) - win_len + 1):
            window = content[i:i + win_len]
            if fold(window) == s_norm:
                return (i, i + win_len)
    return None


def classify_tier(
    value: str,
    content: str,
    *,
    relation: Optional[str] = None,
    relation_soft_eligible: bool = True,
) -> TierResult:
    """对单个 value 在 assistant content 中分级。

    流程：
      1. 直接子串 → Hard
      2. 折叠（去标点/全半角/简繁/大小写）后子串 → Hard（rewrite span 到原 content 中匹配位置）
      3. alias 表中找到的形式恰为 content 子串 + value 允许 soft → Soft
      4. 其他 → Reject
    """
    # 1. 直接子串
    span = _find_substring(content, value)
    if span:
        return TierResult(TierVerdict.HARD, span=span, matched_surface=value)

    # 2. ID 类保留折叠
    span2 = _find_substring(content, fold_keep_id(value))
    if span2:
        return TierResult(TierVerdict.HARD, span=span2, matched_surface=content[span2[0]:span2[1]])

    # 3. 完整无损折叠
    span3 = _find_substring_normalized(content, value)
    if span3:
        return TierResult(TierVerdict.HARD, span=span3, matched_surface=content[span3[0]:span3[1]])

    # 4. alias 表
    if not (relation and not relation_soft_eligible):
        if soft_eligible_for_value(value, relation=relation):
            for alias in get_aliases(value):
                a_span = _find_substring(content, alias)
                if a_span:
                    return TierResult(TierVerdict.SOFT, span=a_span,
                                      matched_surface=alias,
                                      reason=f"alias({alias})")
                a_span2 = _find_substring_normalized(content, alias)
                if a_span2:
                    return TierResult(TierVerdict.SOFT, span=a_span2,
                                      matched_surface=content[a_span2[0]:a_span2[1]],
                                      reason=f"alias_fold({alias})")

    return TierResult(TierVerdict.REJECT, reason="no hard/soft match")
