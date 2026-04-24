"""
patterns/chat.py —— chat-type turn kinds（含 reveal_multi、compositional、third_party）

5 个 kind：general_chat / world_qa / reveal_multi / compositional / third_party
"""
from __future__ import annotations
from typing import Optional
import random

from xinhe.data.persona import Persona
from xinhe.data.multi_fact_templates import sample_multi_reveal
from xinhe.data.templates import (
    COMPOSITIONAL_TEMPLATES,
    ENTITY_FACT_TEMPLATES, ENTITY_RECALL_TEMPLATES,
)
from xinhe.data.registry import register_turn_kind


@register_turn_kind("general_chat")
def general_chat(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """从 cache 读自然闲聊，train_loss=False（防 LoRA 漂）。"""
    if cache is None:
        return None
    t = cache.pop_chat(rng)
    if t is None:
        return None
    return {
        "user": t["user"],
        "assistant": t["assistant"],
        "train_loss": False,
    }


@register_turn_kind("world_qa")
def world_qa(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """从 cache 读事实 QA，train_loss=True，无 VALUE 加权。"""
    if cache is None:
        return None
    t = cache.pop_qa(rng)
    if t is None:
        return None
    return {
        "user": t["user"],
        "assistant": t["assistant"],
        "train_loss": True,
    }


@register_turn_kind("reveal_multi")
def reveal_multi(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """一句话多 fact 披露，value 为 list[str]。"""
    result = sample_multi_reveal(rng, persona)
    if result is None:
        return None
    for s in result["slots"]:
        persona.revealed.add(s)
    return {
        "user": result["user"],
        "assistant": result["assistant"],
        "train_loss": True,
        "value": result["values"],
    }


@register_turn_kind("compositional")
def compositional(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """跨槽组合问答（"我这个年纪的人适合做什么"等）。"""
    if len(persona.revealed) < 1:
        return None
    viable = [
        t for t in COMPOSITIONAL_TEMPLATES
        if all(s in persona.revealed for s in t[0])
    ]
    if not viable:
        return None
    slots, user_tmpl, asst_tmpl = rng.choice(viable)
    fill = {s: persona.slot_value(s) for s in slots}
    values = list(fill.values())
    return {
        "user": user_tmpl.format(**fill),
        "assistant": asst_tmpl.format(**fill),
        "train_loss": True,
        "value": values,
    }


@register_turn_kind("third_party")
def third_party(rng: random.Random, persona: Persona, cache=None) -> Optional[dict]:
    """第三方 entity fact 或 recall。50/50 分布。"""
    if not persona.third_party:
        return None
    tp_key = rng.choice(list(persona.third_party.keys()))
    tp_data = persona.third_party[tp_key]
    if not tp_data:
        return None
    slot = rng.choice(list(tp_data.keys()))
    value = tp_data[slot]

    if slot not in ENTITY_FACT_TEMPLATES:
        return None

    if rng.random() < 0.5 and slot in ENTITY_FACT_TEMPLATES:
        user_tmpl, asst_tmpl = rng.choice(ENTITY_FACT_TEMPLATES[slot])
    else:
        if slot not in ENTITY_RECALL_TEMPLATES:
            return None
        user_tmpl, asst_tmpl = rng.choice(ENTITY_RECALL_TEMPLATES[slot])

    return {
        "user": user_tmpl.format(e=tp_key, ea=tp_key, v=value),
        "assistant": asst_tmpl.format(e=tp_key, ea=tp_key, v=value),
        "train_loss": True,
        "value": value,
    }
