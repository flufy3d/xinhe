"""Beat 3 结构校验：长度 + 轮数硬下限 + Beat 4 存在性。

ling / 小模型常见退化：把 Beat 3（强干扰段）压缩到 1 个 user/assistant pair 就草草了事，
让 W lookup 学不到"长干扰下记忆保持"的信号。yaml 设的 beat3_min_turns=4 / beat3_min_chars=1500
原本只塞在 prompt 里，LLM 不一定遵守，validator 必须复核。

判定（任一不满足整条 reject）：
  - Beat 3 必须存在（plan.beat3_turn_indices 非空）
  - Beat 3 user/assistant pair 数 ≥ plan.beat3_min_turns
  - Beat 3 assistant content 累计中文字符 ≥ plan.beat3_min_chars
  - Beat 4 必须存在（plan.beat4_turn_indices 非空）

只在 1A 5-Beat 流上执行；1B world_qa 因 plan=None 自动跳过。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Beat3StructureResult:
    ok: bool
    n_pairs: int
    n_zh_chars: int
    has_beat4: bool
    reason: str = ""


def _zh_count(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")


def check_beat3_structure(
    conversations: list[dict],
    beat3_turn_indices: list[int],
    beat4_turn_indices: list[int],
    *,
    min_pairs: int,
    min_chars: int,
) -> Beat3StructureResult:
    if not beat3_turn_indices:
        return Beat3StructureResult(
            ok=False, n_pairs=0, n_zh_chars=0, has_beat4=bool(beat4_turn_indices),
            reason="Beat 3 turn indices 为空（LLM 未生成 Beat 3）",
        )

    # 由 parser 写入的 indices 同时含 user 和 assistant；按 assistant 计数 pair（lm_only 标记）
    asst_idxs = [i for i in beat3_turn_indices
                 if 0 <= i < len(conversations)
                 and conversations[i].get("role") == "assistant"]
    n_pairs = len(asst_idxs)
    n_zh = sum(_zh_count(conversations[i].get("content", "")) for i in asst_idxs)
    has_beat4 = bool(beat4_turn_indices) and any(
        0 <= i < len(conversations) and conversations[i].get("role") == "assistant"
        and conversations[i].get("value")
        for i in beat4_turn_indices
    )

    reasons = []
    if n_pairs < min_pairs:
        reasons.append(f"Beat 3 pairs={n_pairs} < min={min_pairs}")
    if n_zh < min_chars:
        reasons.append(f"Beat 3 zh_chars={n_zh} < min={min_chars}")
    if not has_beat4:
        reasons.append("Beat 4 缺失或 assistant.value 为空")

    return Beat3StructureResult(
        ok=len(reasons) == 0,
        n_pairs=n_pairs,
        n_zh_chars=n_zh,
        has_beat4=has_beat4,
        reason="; ".join(reasons) if reasons else "ok",
    )
