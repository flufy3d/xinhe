"""L (Reverse-Erase) + L_partial + C_prime（撤销后查询）。

L: 撤销单 key → tombstone。
L_partial: 多 key 中撤销 1 个，其余保留。
C_prime: L 后用户主动询问被撤销的 key，asst 必须拒答（不能给出旧值）。
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
from xinhe.data.memory_state import MemoryState, OP_ERASE
from xinhe.data.templates.l_reverse_erase import (
    POOL_L,
    POOL_L_PARTIAL,
    POOL_C_PRIME,
)


@register_event("L")
class ReverseEraseEvent(AtomicEvent):
    """L：撤销一个已 active 的 key → tombstone。"""

    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        active = [k for k in state.all_active_keys() if k in ctx.canonical_pool and k[2] != "object"]
        if not active:
            return []
        key = rng.choice(active)
        rel = relation_by_name(key[1])

        state.apply(OP_ERASE, key, [], turn_index=turn_idx, mode=rel.mode)
        ctx.erased_keys.append(key)

        tmpl = pick_template(POOL_L, rng, ctx)
        user_text = tmpl.user_text.format(relation_word=rel.label)
        asst_text = tmpl.asst_text.format(relation_word=rel.label)

        u = make_user_turn(user_text)
        # train_loss=true，但 value=None（拒答 / 撤销没有 value 标记）
        a = make_assistant_turn(asst_text, train_loss="true")
        return [(u, a)]


@register_event("L_partial")
class PartialEraseEvent(AtomicEvent):
    """L_partial：多键中撤销 1 个 key，其余保留。"""

    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        active = [k for k in state.all_active_keys() if k in ctx.canonical_pool and k[2] != "object"]
        if len(active) < 2:
            return []

        # 选要删的 + 至少一个要保留的
        erase_key = rng.choice(active)
        keep_keys = [k for k in active if k != erase_key]
        if not keep_keys:
            return []
        keep_key = rng.choice(keep_keys)

        erase_rel = relation_by_name(erase_key[1])
        keep_rel = relation_by_name(keep_key[1])

        state.apply(OP_ERASE, erase_key, [], turn_index=turn_idx, mode=erase_rel.mode)
        ctx.erased_keys.append(erase_key)

        tmpl = pick_template(POOL_L_PARTIAL, rng, ctx)
        user_text = tmpl.user_text.format(
            erase_word=erase_rel.label, keep_word=keep_rel.label
        )
        asst_text = tmpl.asst_text.format(
            erase_word=erase_rel.label, keep_word=keep_rel.label
        )

        u = make_user_turn(user_text)
        a = make_assistant_turn(asst_text, train_loss="true")
        return [(u, a)]


@register_event("C_prime")
class CPrimeEvent(AtomicEvent):
    """C_prime：L 之后用户问被擦的 key，asst 必须拒答（不能给出旧值）。

    训练目标：让模型学会"被撤销的 key 当前没有记录"，而不是去 W 里挖旧痕迹。
    """

    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        if not ctx.erased_keys:
            return []
        key = ctx.erased_keys[-1]
        rel = relation_by_name(key[1])

        tmpl = pick_template(POOL_C_PRIME, rng, ctx)
        user_text = tmpl.user_text.format(relation_word=rel.label)
        asst_text = tmpl.asst_text.format(relation_word=rel.label)

        u = make_user_turn(user_text)
        a = make_assistant_turn(asst_text, train_loss="true")
        return [(u, a)]
