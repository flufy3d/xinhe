"""A (Write)：单 key 写入。"""
from __future__ import annotations

import random

from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.events._helpers import (
    pick_template,
    relation_by_name,
    make_key,
)
from xinhe.data.memory_state import MemoryState, OP_WRITE
from xinhe.data.templates.a_write import POOL


@register_event("A")
class WriteEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        tmpl = pick_template(POOL, rng, ctx)
        rel = relation_by_name(tmpl.meta["relation"])
        value = ctx.bank(rel.bank).sample_one(rng)
        key = make_key(rel, ctx, rng)

        user_text = tmpl.user_text.format(subject=key[0], value=value)
        asst_text = tmpl.asst_text.format(subject=key[0], value=value)

        state.apply(OP_WRITE, key, [value], turn_index=turn_idx, mode=rel.mode)
        ctx.canonical_pool[key] = value
        if key not in ctx.used_keys:
            ctx.used_keys.append(key)

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
