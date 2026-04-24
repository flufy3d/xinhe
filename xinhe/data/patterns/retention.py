"""
patterns/retention.py —— 结构化 retention/overwrite 类 episode

4 个 pattern：
  stress_retention     —— reveal → chat × N → recall（单槽 retention through distractor）
  multi_slot_retention —— reveal A, B, [C] → chat × N → recall 全部（多槽同时 retention）
  fact_vs_transient    —— anchor (VALUE 5x) + distractor chatter + 召回 anchor
  rapid_overwrite      —— 同 slot 连续覆写 N 次 + 末轮 recall 最新值

val 生成器同名 `val_*` 对应每个 pattern。
"""
from __future__ import annotations
import random
from typing import Optional

from xinhe.data.persona import Persona, sample_persona, SLOT_NAMES, _sample_slot
from xinhe.data.templates import (
    FACT_TEMPLATES, RECALL_TEMPLATES, OVERWRITE_TEMPLATES,
    SIMILAR_ENTITY_PAIRS, ANCHOR_REVEAL_TEMPLATES, ANCHOR_RECALL_TEMPLATES,
    DISTRACTOR_CHATTER_TEMPLATES,
)
from xinhe.data.registry import register_pattern, register_val, get_turn_kind


def _distractor_chat_or_qa(rng: random.Random, cache, persona: Optional[Persona] = None) -> dict:
    """通用 distractor：70% chat / 30% world_qa / 兜底 refusal / 最终兜底 fake 一句。
    强制 train_loss=False，不传梯度。"""
    general_chat = get_turn_kind("general_chat")
    world_qa = get_turn_kind("world_qa")
    refusal = get_turn_kind("refusal")

    if rng.random() < 0.7:
        t = general_chat(rng, persona, cache)
    else:
        t = world_qa(rng, persona, cache)
    if t is None and persona is not None:
        t = refusal(rng, persona, cache)
    if t is None:
        t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
    else:
        t = {**t, "train_loss": False}
    return t


@register_pattern("stress_retention")
def generate_stress_retention_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_chat_between: int = 2,
    max_chat_between: int = 5,
) -> list[dict]:
    """单槽 retention：reveal(s) → chat × N → recall(s) 逐槽重复 + 结尾 cross-recall。"""
    if persona is None:
        persona = sample_persona(rng, num_reveal=rng.randint(2, 4))
    turns = []

    for slot in persona.reveal_order:
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl.format(v=value),
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
        persona.revealed.add(slot)

        n_chat = rng.randint(min_chat_between, max_chat_between)
        for _ in range(n_chat):
            turns.append(_distractor_chat_or_qa(rng, cache, persona))

        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })

    # 结尾 cross-recall
    if len(persona.revealed) >= 2:
        random_slot = rng.choice(list(persona.revealed))
        value = persona.slot_value(random_slot)
        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[random_slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
    return turns


@register_pattern("multi_slot_retention")
def generate_multi_slot_retention_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_chat: int = 2,
    max_chat: int = 5,
) -> list[dict]:
    """多槽同时 retention：连续 reveal 2-3 槽 → chat × N → 随机顺序 recall 全部。"""
    n_reveal = rng.randint(2, 3)
    if persona is None:
        persona = sample_persona(rng, num_reveal=n_reveal)
    turns = []

    for slot in persona.reveal_order:
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl.format(v=value),
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
        persona.revealed.add(slot)

    n_chat = rng.randint(min_chat, max_chat)
    for _ in range(n_chat):
        turns.append(_distractor_chat_or_qa(rng, cache, persona))

    recall_order = list(persona.revealed)
    rng.shuffle(recall_order)
    for slot in recall_order:
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
    return turns


@register_pattern("fact_vs_transient")
def generate_fact_vs_transient_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    num_distractors: int = 3,
) -> list[dict]:
    """长期 anchor (VALUE 5x) + 短期 distractor chatter + 召回 anchor。

    靠 VALUE 权重非对称逼模型把 anchor 放进稳态 head、distractor 放进短代谢 head。
    """
    domain, anchor, distractor = rng.choice(SIMILAR_ENTITY_PAIRS)
    if (domain not in ANCHOR_REVEAL_TEMPLATES
            or domain not in ANCHOR_RECALL_TEMPLATES
            or domain not in DISTRACTOR_CHATTER_TEMPLATES):
        return []

    reveal_u, reveal_a = rng.choice(ANCHOR_REVEAL_TEMPLATES[domain])
    turns = [{
        "user": reveal_u.format(anchor=anchor),
        "assistant": reveal_a.format(anchor=anchor),
        "train_loss": True,
        "value": anchor,
    }]

    distractor_pairs = DISTRACTOR_CHATTER_TEMPLATES[domain]
    if len(distractor_pairs) >= num_distractors:
        chosen = rng.sample(distractor_pairs, num_distractors)
    else:
        chosen = [rng.choice(distractor_pairs) for _ in range(num_distractors)]
    for du, da in chosen:
        turns.append({
            "user": du.format(distractor=distractor),
            "assistant": da.format(distractor=distractor),
            "train_loss": True,
        })

    recall_u, recall_a = rng.choice(ANCHOR_RECALL_TEMPLATES[domain])
    turns.append({
        "user": recall_u,
        "assistant": recall_a.format(anchor=anchor),
        "train_loss": True,
        "value": anchor,
    })
    return turns


@register_pattern("rapid_overwrite")
def generate_rapid_overwrite_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    n_overwrites: int = 3,
    inter_chat_prob: float = 0.3,
) -> list[dict]:
    """同 slot 连续覆写 N 次 + 末轮 recall 最新值。验证 Delta Rule (v - W·k) 擦旧能力。"""
    candidate_slots = [s for s in SLOT_NAMES if s in OVERWRITE_TEMPLATES and s in FACT_TEMPLATES]
    if not candidate_slots:
        return []
    slot = rng.choice(candidate_slots)

    values: list[str] = []
    for _ in range(n_overwrites + 1):
        for __ in range(10):
            v = _sample_slot(rng, slot)
            if all(v[:2] != e[:2] for e in values):
                values.append(v)
                break
        else:
            values.append(_sample_slot(rng, slot))

    fallback = persona or sample_persona(rng, num_reveal=1)
    turns = []

    user_t, asst_t = rng.choice(FACT_TEMPLATES[slot])
    turns.append({
        "user": user_t.format(v=values[0]),
        "assistant": asst_t.format(v=values[0]),
        "train_loss": True,
        "value": values[0],
    })
    for i in range(1, len(values)):
        if rng.random() < inter_chat_prob:
            turns.append(_distractor_chat_or_qa(rng, cache, fallback))
        ow_u, ow_a = rng.choice(OVERWRITE_TEMPLATES[slot])
        turns.append({
            "user": ow_u.format(v=values[i]),
            "assistant": ow_a.format(v=values[i]),
            "train_loss": True,
            "value": values[i],
        })

    final = values[-1]
    rec_u, rec_a = rng.choice(RECALL_TEMPLATES[slot])
    turns.append({
        "user": rec_u,
        "assistant": rec_a.format(v=final),
        "train_loss": True,
        "value": final,
    })
    return turns


# ── Val 生成器（eval_fn 由 persona_joint.py 注入，避免循环 import）──

def _build_last_turn_val(pattern_fn):
    """构造 val 生成器：调用 pattern 直到得到 last-turn-value-bearing 的 episode。"""
    def _gen_val(rng: random.Random, cache, n_samples: int) -> list[list[dict]]:
        episodes = []
        attempts = 0
        while len(episodes) < n_samples and attempts < n_samples * 4:
            attempts += 1
            ep = pattern_fn(rng, None, cache)
            if ep and ep[-1].get("train_loss") and ep[-1].get("value"):
                episodes.append(ep)
        return episodes
    return _gen_val


# 仅注册 val 生成器（eval_fn=None 占位，persona_joint.py 会 patch 上 eval）
register_val("multi_slot_retention")(_build_last_turn_val(generate_multi_slot_retention_episode))
register_val("rapid_overwrite")(_build_last_turn_val(generate_rapid_overwrite_episode))
register_val("irrelevant_forget")(_build_last_turn_val(generate_stress_retention_episode))
