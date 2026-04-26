"""Beat 4 人称反转检测：self-scope fact 召回时,user 提问不准用"你"指 assistant。

Bug 现象（实测 ~4% self-scope 样本）:
  Beat 1: user "我老家是怀化"
  Beat 4 user: "你老家是哪里来着?" ← 把 user 的 fact 当成 assistant 的属性问

正确写法:
  Beat 4 user: "我老家是哪里来着?" / "我是不是说我老家是怀化"  ← user 自指
  四个 recall_form 例子里 user 都用"我"自指,LLM 偶发把第三段干扰期的"你"语境延续。

检测策略:
  - 仅当 plan 里有 self-scope canonical fact 时启用
  - 对每个 Beat 4 之前的 user turn (beat4_indices[i]-1),
    检测 content 是否含以下"你+(占有/属性/动作)+fact 关键词"模式
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# self-scope fact 召回时禁止出现的"你"指 assistant 询问句式
# 关键词来自 RELATIONS 里 self-scope 的 label：
#   fav_food / fav_color / fav_brand / fav_hobby / home_city / hometown / job / pet_kind / pet_name
# 模式按经验观察反转样本归纳，宁可严不可漏（误判由 LLM retry 修复）
_INVERSION_PATTERNS: list[re.Pattern] = [
    # 城市 / 老家
    re.compile(r"你的?老家"),
    re.compile(r"你住在哪"),
    re.compile(r"你住的(城市|地方|那个)"),
    re.compile(r"你的(城市|家乡)"),
    re.compile(r"你家在哪"),
    re.compile(r"你来自哪"),
    re.compile(r"你是哪里人"),
    # 食物
    re.compile(r"你(喜欢|爱)吃"),
    re.compile(r"你的(食物|爱吃的|口味)"),
    re.compile(r"你平时吃"),
    # 品牌 / 饮料
    re.compile(r"你(常用|常喝|爱喝)"),
    re.compile(r"你的(品牌|牌子)"),
    re.compile(r"你平时.{0,4}(喝|用)"),
    # 颜色
    re.compile(r"你(喜欢|爱)什么颜色"),
    re.compile(r"你的颜色"),
    # 爱好 / 兴趣
    re.compile(r"你的(爱好|兴趣)"),
    re.compile(r"你平时(喜欢|爱).{0,3}(做|玩)"),
    re.compile(r"你业余时间"),
    # 职业 / 工作
    re.compile(r"你做什么工作"),
    re.compile(r"你的(工作|职业)"),
    re.compile(r"你是做什么的"),
    # 宠物
    re.compile(r"你养(的|了|什么)"),
    re.compile(r"你的(宠物|猫|狗|鸟|鱼)"),
]


@dataclass
class PronounResult:
    ok: bool
    reason: str = ""
    inverted_turn: int = -1
    matched_pattern: str = ""


def check_beat4_pronoun(
    conversations: list[dict],
    beat4_indices: Iterable[int],
    facts_scope: dict[str, str],
) -> PronounResult:
    """检测 Beat 4 之前的 user turn 是否对 self-scope fact 用了"你"指 assistant。

    facts_scope: {canonical_value: "self"|"third_party"|"object"}
    若 plan 中没有 self-scope fact,直接 ok。
    """
    # 仅当存在 self-scope fact 时检测
    has_self = any(s == "self" for s in facts_scope.values())
    if not has_self:
        return PronounResult(ok=True)

    beat4_list = sorted(set(int(i) for i in beat4_indices))
    for b4_idx in beat4_list:
        # Beat 4 是 assistant turn,前一个 (b4_idx-1) 是触发 recall 的 user turn
        u_idx = b4_idx - 1
        if u_idx < 0 or u_idx >= len(conversations):
            continue
        user_turn = conversations[u_idx]
        if user_turn.get("role") != "user":
            continue
        text = user_turn.get("content", "")
        if not text:
            continue
        for pat in _INVERSION_PATTERNS:
            m = pat.search(text)
            if m:
                return PronounResult(
                    ok=False,
                    reason=f"Beat 4 user[{u_idx}] 用'你'指 assistant 问 self-scope fact: {m.group(0)!r}",
                    inverted_turn=u_idx,
                    matched_pattern=pat.pattern,
                )
    return PronounResult(ok=True)
