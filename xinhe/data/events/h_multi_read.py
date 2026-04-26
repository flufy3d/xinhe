"""H (Multi-Read)：并发召回多个 active key（按 ctx.used_keys 顺序）。"""
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
from xinhe.data.memory_state import MemoryState
from xinhe.data.templates.h_multi_read import POOL


@register_event("H")
class MultiReadEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 收集当前 active 且在 ctx.canonical_pool 内的 key（排除 object scope）
        readable_keys = [k for k in ctx.used_keys if state.query(k) is not None and k[2] != "object"]
        if len(readable_keys) < 2:
            return []

        # 按模板支持的 n_values 选模板
        max_n = min(3, len(readable_keys))
        candidates = [t for t in POOL.templates if t.meta.get("n_values", 0) <= max_n]
        if not candidates:
            return []
        tmpl = rng.choice(candidates)
        n = tmpl.meta["n_values"]

        # 取最近 n 个 key（按写入顺序）
        keys_to_read = readable_keys[-n:]
        values = []
        for k in keys_to_read:
            rec = state.query(k)
            if rec is None or not rec.values:
                return []
            values.append(rec.values[0])

        slots = {f"v{i+1}": v for i, v in enumerate(values)}
        user_text = tmpl.user_text.format(**slots)
        asst_text = tmpl.asst_text.format(**slots)

        u = make_user_turn(user_text)
        tier = "hard"
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=values,
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]
