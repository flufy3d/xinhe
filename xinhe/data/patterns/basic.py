"""
patterns/basic.py —— persona 槽基础 turn kinds

5 个 kind：reveal_single / recall / refusal / overwrite / reveal_multi
（reveal_multi 涉及多槽，依赖 multi_fact_templates，保留 chat.py 里以免触发循环）
"""
from __future__ import annotations
from typing import Optional
import random

from xinhe.data.persona import Persona, _sample_slot
from xinhe.data.refusal_templates import sample_refusal
from xinhe.data.templates import (
    FACT_TEMPLATES, RECALL_TEMPLATES, OVERWRITE_TEMPLATES,
)
from xinhe.data.registry import register_turn_kind


@register_turn_kind("reveal_single")
def reveal_single(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """披露下一个（按 reveal_order 顺序的）未披露 slot。"""
    unrev = persona.unrevealed_slots()
    if not unrev:
        return None
    slot = unrev[0]
    value = persona.slot_value(slot)
    user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
    persona.revealed.add(slot)
    return {
        "user": user_tmpl.format(v=value),
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }


@register_turn_kind("recall")
def recall(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """问已披露的 slot。"""
    if not persona.revealed:
        return None
    slot = rng.choice(list(persona.revealed))
    value = persona.slot_value(slot)
    user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
    return {
        "user": user_tmpl,
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }


@register_turn_kind("refusal")
def refusal(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """问未披露的 slot → 拒答（无 VALUE 权重）。"""
    candidates = persona.refusal_candidates()
    if not candidates:
        return None
    slot = rng.choice(candidates)
    user, asst = sample_refusal(rng, slot)
    return {
        "user": user,
        "assistant": asst,
        "train_loss": True,
        # 不给 value！避免学死具体拒答措辞
    }


@register_turn_kind("overwrite")
def overwrite(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """纠正已披露的 slot，替换 persona 的值。"""
    if not persona.revealed:
        return None
    slot = rng.choice(list(persona.revealed))
    if slot not in OVERWRITE_TEMPLATES:
        return None

    # 采一个新值（不同于旧值）
    old = persona.slot_value(slot)
    new_val = old
    for _ in range(5):
        candidate = _sample_slot(rng, slot)
        if candidate != old:
            new_val = candidate
            break
    if new_val == old:
        return None

    setattr(persona, slot, new_val)

    user_tmpl, asst_tmpl = rng.choice(OVERWRITE_TEMPLATES[slot])
    return {
        "user": user_tmpl.format(v=new_val),
        "assistant": asst_tmpl.format(v=new_val),
        "train_loss": True,
        "value": new_val,
    }
