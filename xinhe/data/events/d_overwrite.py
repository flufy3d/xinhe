"""D (Overwrite)：覆盖纠错事件 — 同 key 旧值失效，新值成为 active。"""
from __future__ import annotations

import random

from xinhe.data.events._helpers import (
    pick_template,
    relation_by_name,
)
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState, OP_OVERWRITE
from xinhe.data.templates.d_overwrite import POOL


@register_event("D")
class OverwriteEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 必须有一个已 active 的 self-scope key 才能覆盖
        active_self = [
            k for k in state.all_active_keys()
            if k[2] == "self" and k in ctx.canonical_pool
        ]
        if not active_self:
            return []

        key = rng.choice(active_self)
        rel = relation_by_name(key[1])
        rec = state.query(key)
        if rec is None or not rec.values:
            return []
        old = rec.values[0]
        # 选个跟 old 不同的新值
        bank = ctx.bank(rel.bank)
        new = old
        for _ in range(20):
            cand = bank.sample_one(rng)
            if cand != old:
                new = cand
                break
        if new == old:
            return []  # 同值无意义

        tmpl = pick_template(POOL, rng, ctx, relation=rel.name)
        user_text = tmpl.user_text.format(old=old, new=new)
        asst_text = tmpl.asst_text.format(old=old, new=new)

        state.apply(OP_OVERWRITE, key, [new], turn_index=turn_idx, mode=rel.mode)
        ctx.canonical_pool[key] = new

        u = make_user_turn(user_text)
        # asst 只把 {new} 当 value（旧值不打权重）
        tier = "hard"
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=[new],
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]


@register_event("D_partial")
class PartialOverwriteEvent(AtomicEvent):
    """D_partial：用于 S7 — 多 key 中只覆盖一个。

    与 D 等价（同样 OP_OVERWRITE），区别在于 skeleton 上下文：S7 调用前会先跑 F
    写多个 key，D_partial 在其中挑一个覆盖。
    """
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        return OverwriteEvent().run(rng, state, ctx, turn_idx)
