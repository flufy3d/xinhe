"""Beat 4 scope 错配检测：召回的 value 归属(self/third_party) 必须与 user 提问句的人称对齐。

Bug 现象 1(scope 错配):
  facts = [('浅咖', third_party, subject=谢洋洋), ('新闻记者', self)]
  user[Beat 4]: "你还记得我说过我喜欢什么颜色吗？"   ← 全用"我"自指
  asst[Beat 4]: "你说你喜欢浅咖。"                  ← 错把 third_party 的颜色当 self 召回

Bug 现象 2(user 自报答案,非真正召回提问):
  fact = ('无极', self, home_city)
  user[Beat 4]: "也得去无极看看,空气好。"  ← 没"我"自指 + 自己说出 canonical,根本不是召回 trigger
  asst[Beat 4]: "你老家是无极。"          ← 顶多算 echo,W lookup 学不到任何东西

判定规则:
  对 Beat 4 assistant 召回的每个 value v(在 plan canonical 中):
    - 若 v.scope == "third_party":
        user 句必须提到 v.subject(人名) 或第三方泛指词(朋友/同事/他/她/...)
        否则 → reject(user 全自指却召回 third_party value,语义错配)
    - 若 v.scope == "self":
        user 句必须含"我"自指
        否则 → reject(user 没自指 → 要么人称错配,要么 user 自报答案不是召回提问)

注意 multi_fact 召回:user 句往往同时含"我"+"朋友 X",每个 value 独立判定,不会互相误伤。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_THIRD_PARTY_PRONOUNS = (
    "朋友", "同事", "邻居", "老同学", "前同事", "老板", "客户", "房东",
    "他", "她", "他们", "她们", "对方",
)


@dataclass
class ScopeResult:
    ok: bool
    reason: str = ""
    inverted_turn: int = -1
    bad_value: str = ""


def check_beat4_scope(
    conversations: list[dict],
    beat4_indices: Iterable[int],
    facts_meta: list[dict],
) -> ScopeResult:
    """facts_meta: [{canonical_value, scope, subject, ...}]
    若 plan 没有 third_party fact 也没 self fact,直接 ok。
    """
    if not facts_meta:
        return ScopeResult(ok=True)

    fact_by_value = {f.get("canonical_value"): f for f in facts_meta}

    for b4_idx in sorted(set(int(i) for i in beat4_indices)):
        if b4_idx < 1 or b4_idx >= len(conversations):
            continue
        a_turn = conversations[b4_idx]
        if a_turn.get("role") != "assistant":
            continue
        u_turn = conversations[b4_idx - 1]
        if u_turn.get("role") != "user":
            continue
        u_text = u_turn.get("content", "")
        values = a_turn.get("value") or []
        if not values:
            continue

        for v in values:
            fact = fact_by_value.get(v)
            if not fact:
                continue
            scope = fact.get("scope", "self")
            subject = (fact.get("subject") or "").strip()

            if scope == "third_party":
                # user 句必须提到 subject 或第三方泛指词
                has_subject = bool(subject) and subject in u_text
                has_pronoun = any(p in u_text for p in _THIRD_PARTY_PRONOUNS)
                if not (has_subject or has_pronoun):
                    return ScopeResult(
                        ok=False,
                        reason=f"third_party value {v!r}(subject={subject!r}) 召回但 user 全自指: {u_text[:80]!r}",
                        inverted_turn=b4_idx - 1,
                        bad_value=v,
                    )
            elif scope == "self":
                # self value: user 句必须含"我"自指
                # （没"我"通常意味着两类问题之一:人称错配 OR user 自报答案非真正召回提问）
                if "我" not in u_text:
                    return ScopeResult(
                        ok=False,
                        reason=(
                            f"self value {v!r} 召回但 user 句无'我'自指 "
                            f"(可能 user 没真正发起召回提问,而是自己讲出了答案): "
                            f"{u_text[:80]!r}"
                        ),
                        inverted_turn=b4_idx - 1,
                        bad_value=v,
                    )

    return ScopeResult(ok=True)
