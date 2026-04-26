"""J (Augment)：set 模式追加 — 旧值保留，新值并存。

J 不能与 D 共享模板池（D 是覆盖，J 是并列）。
J 触发：必须存在已 active 的 set-mode 或 scalar key（J 会把 scalar 升级为 set）。
"""
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
from xinhe.data.memory_state import MemoryState, OP_AUGMENT, OP_WRITE
from xinhe.data.templates.j_augment import POOL


@register_event("J")
class AugmentEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 先看 J 模板支持哪些 relation
        supported_relations = {t.meta.get("relation") for t in POOL.templates}

        # 找已 active 的 self-scope key 且 relation 在 J 模板里
        active_self = [
            k for k in state.all_active_keys()
            if k[2] == "self" and k[1] in supported_relations and k in ctx.canonical_pool
        ]

        # 没有合适的 active key → 自己先写一个再 augment（方便骨架不必强 A 在前）
        if not active_self:
            tmpl = rng.choice(POOL.templates)
            rel = relation_by_name(tmpl.meta["relation"])
            old = ctx.bank(rel.bank).sample_one(rng)
            key = ("user", rel.name, "self")
            state.apply(OP_WRITE, key, [old], turn_index=turn_idx, mode="set")
            ctx.canonical_pool[key] = old
            if key not in ctx.used_keys:
                ctx.used_keys.append(key)
        else:
            key = rng.choice(active_self)
            rel = relation_by_name(key[1])
            old = state.query(key).values[0]
            cands = [t for t in POOL.templates if t.meta.get("relation") == rel.name]
            tmpl = rng.choice(cands) if cands else rng.choice(POOL.templates)
            if tmpl.meta.get("relation") != rel.name:
                rel = relation_by_name(tmpl.meta["relation"])
                key = ("user", rel.name, "self")
                old = state.query(key).values[0] if state.query(key) else ctx.bank(rel.bank).sample_one(rng)

        # 选一个不同的 new value
        bank = ctx.bank(rel.bank)
        new = old
        for _ in range(20):
            cand = bank.sample_one(rng)
            if cand != old:
                new = cand
                break
        if new == old:
            return []

        state.apply(OP_AUGMENT, key, [new], turn_index=turn_idx, mode="set")
        # canonical_pool 标记最新 surface（多值时取最近写入）
        ctx.canonical_pool[key] = new

        user_text = tmpl.user_text.format(old=old, new=new)
        asst_text = tmpl.asst_text.format(old=old, new=new)

        u = make_user_turn(user_text)
        tier = "hard"
        # 主要训练目标是 {new} 召回；{old} 在 asst 里复读，不重复打权重
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=[new],
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]
