"""F (Multi-Write)：单轮多 key 并发写入（user 一句话抛 N 条事实，asst 一句话确认）。"""
from __future__ import annotations

import random

from xinhe.data.events._helpers import (
    pick_template,
    relation_by_name,
    make_key,
)
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState, OP_MULTI_WRITE
from xinhe.data.templates.f_multi_write import POOL


@register_event("F")
class MultiWriteEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        tmpl = pick_template(POOL, rng, ctx)
        relations = tmpl.meta["relations"]
        # 采样每个 relation 的 value
        slots: dict[str, str] = {}
        values_in_order: list[str] = []
        for i, rname in enumerate(relations):
            rel = relation_by_name(rname)
            v = ctx.bank(rel.bank).sample_one(rng)
            slot_key = f"v{i + 1}"
            slots[slot_key] = v
            values_in_order.append(v)
            key = make_key(rel, ctx, rng)
            state.apply(OP_MULTI_WRITE, key, [v], turn_index=turn_idx, mode=rel.mode)
            ctx.canonical_pool[key] = v
            if key not in ctx.used_keys:
                ctx.used_keys.append(key)

        user_text = tmpl.user_text.format(**slots)
        asst_text = tmpl.asst_text.format(**slots)

        u = make_user_turn(user_text)
        tier = "hard"
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=values_in_order,
            tier=tier,
            base_weight=ctx.value_weight(tier),  # 每 span 自动除以 len
        )
        return [(u, a)]
