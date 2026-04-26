"""K (Stale-Read)：覆盖后旧值复读 → asst 纠正为新值。

依赖：必须有一个 key 经历过 OP_OVERWRITE（is_stale(key) 为 True 且 active）。
通常 skeleton S5 = [A, D, {En}, K] 在 D 后插 K。
"""
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
from xinhe.data.templates.k_stale_read import POOL


@register_event("K")
class StaleReadEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        # 找到经历过 OVERWRITE 的 active key
        candidate_keys = []
        for k in state.all_active_keys():
            ops = state.history.get(k, [])
            if not ops:
                continue
            # 必须有 OVERWRITE 历史 + 仍 active
            if any(op.op == OP_OVERWRITE for op in ops):
                candidate_keys.append(k)
        if not candidate_keys:
            return []

        key = rng.choice(candidate_keys)
        rel = relation_by_name(key[1])
        rec = state.query(key)
        new = rec.values[0]
        old = state.previous_value(key)
        if not old:
            return []
        old_val = old[0]

        # 选 K 模板（优先 relation 匹配）
        cands = [t for t in POOL.templates if t.meta.get("relation") == rel.name]
        tmpl = rng.choice(cands) if cands else rng.choice(POOL.templates)
        user_text = tmpl.user_text.format(old=old_val, new=new)
        asst_text = tmpl.asst_text.format(old=old_val, new=new)

        u = make_user_turn(user_text)
        tier = "hard"
        # 训练目标：纠正后的新值 {new}（不再是旧值）
        a = make_assistant_turn(
            asst_text,
            train_loss="true",
            values=[new],
            tier=tier,
            base_weight=ctx.value_weight(tier),
        )
        return [(u, a)]
