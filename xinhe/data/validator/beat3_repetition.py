"""Beat 3 重复检测：捕捉 LLM degeneracy(同一短句 / 短语高频循环)。

Bug 现象(实测 oss-20b 偶发):
  Beat 3 assistant 内部反复出现 "你说它的叫声像是想让你知道它在干什么..." 15+ 次,
  validator 字数 / 纯洁性 / scope 全过,但训练进去会让 LM 学到"重复是 OK 的"。

检测策略:
  - 取每个 Beat 3 assistant turn 的 content
  - 滑窗 N-char(默认 12) 统计频次
  - 任一 N-gram 出现次数 ≥ MAX_REPEAT(默认 4) → reject
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RepetitionResult:
    ok: bool
    reason: str = ""
    repeated_ngram: str = ""
    count: int = 0
    turn_index: int = -1


def _count_ngrams(text: str, n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    if len(text) < n:
        return counts
    for i in range(len(text) - n + 1):
        gram = text[i:i + n]
        # 跳过仅空白 / 标点的窗口(降误判)
        if all(not c.isalnum() and not ('一' <= c <= '鿿') for c in gram):
            continue
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def check_beat3_repetition(
    conversations: list[dict],
    beat3_indices: Iterable[int],
    *,
    n: int = 12,
    max_repeat: int = 4,
) -> RepetitionResult:
    """对每个 Beat 3 assistant turn 检 12-char 重复。
    >=4 次同样 12-char 串视为 degeneracy。
    """
    for idx in beat3_indices:
        if idx < 0 or idx >= len(conversations):
            continue
        turn = conversations[idx]
        if turn.get("role") != "assistant":
            continue
        text = turn.get("content", "")
        if not text:
            continue
        counts = _count_ngrams(text, n)
        if not counts:
            continue
        worst = max(counts.items(), key=lambda kv: kv[1])
        gram, c = worst
        if c >= max_repeat:
            return RepetitionResult(
                ok=False,
                reason=f"Beat 3 turn[{idx}] 含 {c}× 重复短语 {gram!r}",
                repeated_ngram=gram,
                count=c,
                turn_index=idx,
            )
    return RepetitionResult(ok=True)
