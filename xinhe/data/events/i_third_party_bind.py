"""I (Third-Party-Bind)：引入第三方实体 + 绑定 1 条属性。

写入 (subject_name, relation, third_party) key。
后续 G/H/L_partial 通过 ctx.canonical_pool["__third_party_subject"] 复用同一 subject。
"""
from __future__ import annotations

import random

from xinhe.data.events._helpers import (
    pick_template,
    relation_by_name,
    get_or_seed_third_party,
)
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState, OP_BIND_THIRD, OP_WRITE
from xinhe.data.templates.i_third_party_bind import POOL


@register_event("I")
class ThirdPartyBindEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        tmpl = pick_template(POOL, rng, ctx)
        rel = relation_by_name(tmpl.meta["relation"])
        subject = get_or_seed_third_party(ctx, rng)
        value = ctx.bank(rel.bank).sample_one(rng)
        key = (subject, rel.name, "third_party")

        # 先 BIND 声明（history 标记），再 WRITE 真正写值
        state.apply(OP_BIND_THIRD, key, [], turn_index=turn_idx, mode=rel.mode)
        state.apply(OP_WRITE, key, [value], turn_index=turn_idx, mode=rel.mode)
        ctx.canonical_pool[key] = value
        if key not in ctx.used_keys:
            ctx.used_keys.append(key)

        user_text = tmpl.user_text.format(subject=subject, value=value)
        asst_text = tmpl.asst_text.format(subject=subject, value=value)

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
