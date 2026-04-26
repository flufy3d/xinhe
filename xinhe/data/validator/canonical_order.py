"""Canonical 出现顺序检测：assistant 不得抢在 user 前面输出 canonical。

Bug 现象(实测 oss-20b 偶发):
  facts = [..., ('干蒸烧卖', self, fav_food)]
  user[8]:  "你觉得西凤酒配什么菜好？"
  asst[9]:  "西凤酒和辣子鸡、干蒸烧卖都很搭。"      ← assistant 主动提"干蒸烧卖"
  user[10]: "对了,我还想试试干蒸烧卖。"             ← user 被动附和

干蒸烧卖在 user[10] 出现 → user_injection 字面通过,但 LLM 实际是
"无中生有"地把 fact 抛给 user。训练时 assistant 学到"凭空编造 user 属性是 OK 的"。

判定规则:
  对每个 canonical_value v:
    first_user_idx = 首次包含 v 的 user turn idx
    first_asst_idx = 首次包含 v 的 assistant turn idx
    若 first_asst_idx 存在 且 < first_user_idx (或 first_user_idx 不存在) → reject
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanonicalOrderResult:
    ok: bool
    reason: str = ""
    bad_value: str = ""
    first_user_idx: int = -1
    first_asst_idx: int = -1


def check_canonical_order(
    conversations: list[dict],
    canonical_values: list[str],
) -> CanonicalOrderResult:
    """对每个 canonical_value,查 user / assistant 各自首次出现位置。
    要求 first_user_idx 存在 且 < first_asst_idx。
    """
    for v in canonical_values:
        if not v:
            continue
        first_user = -1
        first_asst = -1
        for i, c in enumerate(conversations):
            content = c.get("content", "") or ""
            if v not in content:
                continue
            role = c.get("role")
            if role == "user" and first_user == -1:
                first_user = i
            elif role == "assistant" and first_asst == -1:
                first_asst = i
            if first_user >= 0 and first_asst >= 0:
                break
        if first_asst < 0:
            # assistant 从未提到 v(理论上 Beat 1/Beat 4 都该提)→ 不在本检测器范围
            continue
        if first_user < 0 or first_asst < first_user:
            return CanonicalOrderResult(
                ok=False,
                reason=f"canonical {v!r} 出现顺序错乱: first_user={first_user}, first_asst={first_asst}(assistant 抢先输出)",
                bad_value=v,
                first_user_idx=first_user,
                first_asst_idx=first_asst,
            )
    return CanonicalOrderResult(ok=True)
