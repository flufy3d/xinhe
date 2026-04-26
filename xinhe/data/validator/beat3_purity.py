"""Beat 3 纯洁性：Beat 3 user+assistant 文本不得含 Beat 1 canonical values 或 aliases。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from xinhe.data.validator.entity_extract import any_term_in_text
from xinhe.data.validator.normalize import fold


@dataclass
class PurityResult:
    ok: bool
    leaked_terms: list[str]
    reason: str = ""


def check_beat3_purity(
    beat3_texts: Iterable[str],
    banned_terms: list[str],
    *,
    fold_normalize: bool = True,
) -> PurityResult:
    """检查 beat3_texts 中是否泄漏 banned_terms。

    banned_terms 通常 = Beat 1 canonical values ∪ alias 表 ∪ jieba 实体词碎片
    """
    leaked: set[str] = set()
    for text in beat3_texts:
        # 原文检查
        for hit in any_term_in_text(text, banned_terms):
            leaked.add(hit)
        # 折叠检查（防止"苹 果"加空格、繁简变体规避）
        if fold_normalize:
            folded_text = fold(text)
            for term in banned_terms:
                folded_term = fold(term)
                if folded_term and folded_term in folded_text:
                    leaked.add(term)

    if leaked:
        return PurityResult(ok=False, leaked_terms=sorted(leaked),
                            reason=f"Beat3 含 {len(leaked)} 条 Beat1 canonical/alias 泄漏")
    return PurityResult(ok=True, leaked_terms=[])
