"""B (Read)：召回当前 active value。

依赖：state 中存在对应 (subject, relation, scope) 的 active 记录。
通常由 skeleton runner 在 A/F 之后插入 B；若无可读 key，骨架会插入新的 A 兜底。
"""
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
)
from xinhe.data.memory_state import MemoryState
from xinhe.data.templates.b_read import POOL


@register_event("B")
class ReadEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 优先读 ctx.canonical_pool 中已写过的 key（同 sample 内 A/F 的产物）
        readable = [k for k in state.all_active_keys() if k in ctx.canonical_pool]
        if not readable:
            # fallback: 没有可读 key，跳过事件（runner 会处理）
            return []

        # 选一个 active key 来读
        key = rng.choice(readable)
        rel_name = key[1]
        rel = relation_by_name(rel_name)
        rec = state.query(key)
        if rec is None or not rec.values:
            return []
        value = rec.values[0]  # scalar 默认取第一个；set 模式由 H 处理

        tmpl = pick_template(POOL, rng, ctx, relation=rel_name)
        user_text = tmpl.user_text.format(subject=key[0], value=value)
        asst_text = tmpl.asst_text.format(subject=key[0], value=value)

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
