"""实体抽取 —— 用 jieba 分词做 Beat 3 纯洁性检测。

依赖：可选 jieba（pip install jieba）。如果未装，fallback 到 char-ngram 切分（粗略但够用）。

只用于 Beat 3 检查 banned_terms 是否出现 —— 不是 NER，无须高准确率。
"""
from __future__ import annotations

import re
from functools import lru_cache


_HAS_JIEBA: bool | None = None


def _init_jieba() -> bool:
    global _HAS_JIEBA
    if _HAS_JIEBA is not None:
        return _HAS_JIEBA
    try:
        import jieba  # noqa: F401
        _HAS_JIEBA = True
    except ImportError:
        _HAS_JIEBA = False
    return _HAS_JIEBA


@lru_cache(maxsize=1024)
def _ngrams(text: str, n: int) -> tuple[str, ...]:
    return tuple(text[i:i + n] for i in range(len(text) - n + 1))


def tokenize(text: str) -> list[str]:
    """中文分词。jieba 优先，fallback 用 1/2/3-gram。"""
    if _init_jieba():
        import jieba
        return [t for t in jieba.lcut(text, cut_all=False) if t.strip()]
    # fallback: 多粒度 ngram 联合
    out: set[str] = set()
    for n in (1, 2, 3, 4):
        out.update(g for g in _ngrams(text, n) if g.strip())
    return list(out)


def find_term(text: str, term: str) -> bool:
    """单纯子串检查（已经是核心 truth），加 token 级辅助检查。"""
    if not term:
        return False
    if term in text:
        return True
    # token 级（jieba 切完后是否包含等价 token）
    tokens = tokenize(text)
    return term in tokens


def any_term_in_text(text: str, terms: list[str]) -> list[str]:
    """返回 text 中命中的所有 term。"""
    return [t for t in terms if find_term(text, t)]
