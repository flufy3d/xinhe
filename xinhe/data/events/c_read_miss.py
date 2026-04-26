"""C (Read-Miss)：查询从未写入的 key → 拒答。"""
from __future__ import annotations

import random

from xinhe.data.events._helpers import pick_template, relation_by_name
from xinhe.data.events._relations import sample_relation
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState
from xinhe.data.templates.c_read_miss import POOL


@register_event("C")
class ReadMissEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 选一个目前 state 中不存在 active 记录的 self-scope relation
        active_relations = {k[1] for k in state.all_active_keys()}
        for _ in range(20):
            rel = sample_relation(rng, scope="self")
            if rel.name not in active_relations:
                break
        else:
            return []  # 所有 relation 都已被写过，跳过

        tmpl = pick_template(POOL, rng, ctx)
        # 没 {value}，但 {relation_word} 来自 RelationSpec.label
        user_text = tmpl.user_text.format(relation_word=rel.label)
        asst_text = tmpl.asst_text.format(relation_word=rel.label)

        u = make_user_turn(user_text)
        # train_loss=true（拒答梯度有用）；value=None（无 hard span）
        a = make_assistant_turn(asst_text, train_loss="true")
        return [(u, a)]
