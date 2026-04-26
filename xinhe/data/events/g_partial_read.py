"""G (Partial-Read)：多写后单读 — 从 ctx.canonical_pool 中挑一个 key 单独召回。"""
from __future__ import annotations

import random

from xinhe.data.events._helpers import pick_template, relation_by_name
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState
from xinhe.data.templates.g_partial_read import POOL


@register_event("G")
class PartialReadEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # object scope 只走 A/B；G/H/L 不挑 object key
        readable = [k for k in state.all_active_keys() if k in ctx.canonical_pool and k[2] != "object"]
        if not readable:
            return []

        key = rng.choice(readable)
        rec = state.query(key)
        if rec is None or not rec.values:
            return []
        rel = relation_by_name(key[1])
        value = rec.values[0]

        tmpl = pick_template(POOL, rng, ctx)
        user_text = tmpl.user_text.format(relation_word=rel.label, value=value)
        asst_text = tmpl.asst_text.format(relation_word=rel.label, value=value)

        u = make_user_turn(user_text)
        tier = "hard"
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=[value],
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]
