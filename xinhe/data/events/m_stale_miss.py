"""M (Stale-Miss)：覆盖后用户问旧值，asst 说明旧值已不再有效 + 给新值。"""
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
from xinhe.data.memory_state import MemoryState, OP_OVERWRITE
from xinhe.data.templates.m_stale_miss import POOL


@register_event("M")
class StaleMissEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 找经历过 OVERWRITE 且仍 active 的 key
        candidates = []
        for k in state.all_active_keys():
            if any(op.op == OP_OVERWRITE for op in state.history.get(k, [])):
                candidates.append(k)
        if not candidates:
            return []
        key = rng.choice(candidates)
        rel = relation_by_name(key[1])
        rec = state.query(key)
        new = rec.values[0]
        old = state.previous_value(key)
        if not old:
            return []
        old_val = old[0]

        cands = [t for t in POOL.templates if t.meta.get("relation") == rel.name]
        tmpl = rng.choice(cands) if cands else rng.choice(POOL.templates)
        user_text = tmpl.user_text.format(old=old_val, new=new)
        asst_text = tmpl.asst_text.format(old=old_val, new=new)

        u = make_user_turn(user_text)
        tier = "hard"
        # 训练目标：当前 active 值 {new}（旧值已 stale）
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=[new],
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]
