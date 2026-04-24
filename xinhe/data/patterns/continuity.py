"""
patterns/continuity.py —— v7.1 新增：上下文连续性类别

4 个 pattern：
  reference_back      —— 用户引用自己之前披露的 slot，要求 recall（吸收原 meta_recall）
  context_followup    —— assistant 自然引用已披露 persona 回答开放问题
  topic_continuation  —— 围绕同一 persona 主题 3-5 轮深入 + 末轮 recall 某 sub-fact
  entity_tracking     —— 引入第三方 entity → distractor → 代词消解 recall

所有 val 仅注册生成器，eval_fn 由 persona_joint.py patch。
"""
from __future__ import annotations
import random
from typing import Optional

from xinhe.data.persona import Persona, sample_persona
from xinhe.data.samplers import random_name
from xinhe.data.templates import (
    FACT_TEMPLATES, RECALL_TEMPLATES,
    QUOTE_BACK_TEMPLATES, META_RECALL_USER_TEMPLATES,
    CONTEXT_FOLLOWUP_TEMPLATES, TOPIC_CHAIN_SEEDS,
    ENTITY_FACT_TEMPLATES, REFERENCE_BACK_ALL,
)
from xinhe.data.registry import register_pattern, register_val, get_turn_kind


def _distractor(rng: random.Random, cache, persona: Optional[Persona]) -> dict:
    general_chat = get_turn_kind("general_chat")
    world_qa = get_turn_kind("world_qa")
    if rng.random() < 0.7:
        t = general_chat(rng, persona, cache)
    else:
        t = world_qa(rng, persona, cache)
    if t is None:
        t = {"user": "嗯。", "assistant": "好的。", "train_loss": False}
    else:
        t = {**t, "train_loss": False}
    return t


# ═══════════════════════════════════════════════════════════════════
# 1. reference_back: user 引用自己之前披露的 slot，要求 recall
# ═══════════════════════════════════════════════════════════════════

@register_pattern("reference_back")
def generate_reference_back_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_chat: int = 1,
    max_chat: int = 2,
    meta_prob: float = 0.3,
) -> list[dict]:
    """user reveal → 1-2 chat → user quote-back recall。
    meta_prob 概率使用 META_RECALL_USER_TEMPLATES 的"我刚才说了什么"通用 query（吸收原 meta_recall）。

    明确区别于 refusal（"你没告诉过"）：这里是"我告诉过你请 recall"。
    """
    persona = persona or sample_persona(rng, num_reveal=2)
    # 挑一个能从 QUOTE_BACK_TEMPLATES 覆盖的 slot
    viable_slots = [s for s in persona.reveal_order if s in QUOTE_BACK_TEMPLATES]
    if not viable_slots:
        return []
    slot = viable_slots[0]
    value = persona.slot_value(slot)

    # reveal
    user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
    turns = [{
        "user": user_tmpl.format(v=value),
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }]
    persona.revealed.add(slot)

    # 1-2 轮 chat
    n_chat = rng.randint(min_chat, max_chat)
    for _ in range(n_chat):
        turns.append(_distractor(rng, cache, persona))

    # recall：meta 风格（"我刚才说了什么"）或 quote-back
    if rng.random() < meta_prob:
        # 元认知自指 —— 回答是最近披露的 persona 事实
        recall_u = rng.choice(META_RECALL_USER_TEMPLATES)
        # assistant 用 quote-back 的 recall 模板回答
        _, asst_tmpl = rng.choice(QUOTE_BACK_TEMPLATES[slot])
        recall_a = asst_tmpl.format(v=value)
    else:
        user_q_tmpl, asst_tmpl = rng.choice(QUOTE_BACK_TEMPLATES[slot])
        recall_u = user_q_tmpl
        recall_a = asst_tmpl.format(v=value)

    turns.append({
        "user": recall_u,
        "assistant": recall_a,
        "train_loss": True,
        "value": value,
    })
    return turns


# ═══════════════════════════════════════════════════════════════════
# 2. context_followup: assistant 引用 persona 回答开放问题
# ═══════════════════════════════════════════════════════════════════

@register_pattern("context_followup")
def generate_context_followup_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_chat: int = 1,
    max_chat: int = 2,
) -> list[dict]:
    """reveal persona → 1-2 chat → user 开放问题 → assistant 引用 persona 回答。"""
    persona = persona or sample_persona(rng, num_reveal=2)
    viable_slots = [s for s in persona.reveal_order if s in CONTEXT_FOLLOWUP_TEMPLATES]
    if not viable_slots:
        return []
    slot = viable_slots[0]
    value = persona.slot_value(slot)

    # reveal
    user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
    turns = [{
        "user": user_tmpl.format(v=value),
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }]
    persona.revealed.add(slot)

    # 1-2 轮 chat
    n_chat = rng.randint(min_chat, max_chat)
    for _ in range(n_chat):
        turns.append(_distractor(rng, cache, persona))

    # user 开放问题 → assistant 引用 persona
    user_q, asst_tmpl = rng.choice(CONTEXT_FOLLOWUP_TEMPLATES[slot])
    turns.append({
        "user": user_q,
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,   # value 在 assistant 回答里
    })
    return turns


# ═══════════════════════════════════════════════════════════════════
# 3. topic_continuation: 围绕 persona 主题 3-5 轮深入 + 末轮 recall sub-fact
# ═══════════════════════════════════════════════════════════════════

@register_pattern("topic_continuation")
def generate_topic_continuation_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
) -> list[dict]:
    """主题链多轮深入 + 末轮 recall。

    结构：
      t=0 reveal(slot=v)
      t=1..N chat_beats（围绕主题，部分轮披露 sub_fact）
      t=N+1 recall 最近披露的 sub_fact
    """
    persona = persona or sample_persona(rng, num_reveal=2)
    viable_slots = [s for s in persona.reveal_order if s in TOPIC_CHAIN_SEEDS]
    if not viable_slots:
        return []
    slot = viable_slots[0]
    value = persona.slot_value(slot)

    seed_chains = TOPIC_CHAIN_SEEDS[slot]
    chain = rng.choice(seed_chains)

    # t=0 reveal
    user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
    turns = [{
        "user": user_tmpl.format(v=value),
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }]
    persona.revealed.add(slot)

    # chain beats
    last_sub_fact = None
    for user_q, asst_tmpl, sub_fact in chain:
        try:
            asst_a = asst_tmpl.format(v=value, anchor=sub_fact or "")
        except KeyError:
            asst_a = asst_tmpl.format(v=value)
        turn = {
            "user": user_q.format(v=value),
            "assistant": asst_a,
            "train_loss": True,
        }
        if sub_fact:
            turn["value"] = sub_fact
            last_sub_fact = sub_fact
        turns.append(turn)

    # 确保最后一个 turn 是 value-bearing（便于 val 判定）
    # 如果 chain 最后 turn 没 value，补一个 explicit recall
    if not turns[-1].get("value") and last_sub_fact:
        turns.append({
            "user": f"刚说的那个细节是什么来着？",
            "assistant": last_sub_fact,
            "train_loss": True,
            "value": last_sub_fact,
        })
    return turns


# ═══════════════════════════════════════════════════════════════════
# 4. entity_tracking: 第三方 entity + distractor + 代词消解 recall
# ═══════════════════════════════════════════════════════════════════

_THIRD_PARTY_RELATIONS = ["我朋友", "我同事", "我表弟", "我室友", "我老板", "我邻居"]


@register_pattern("entity_tracking")
def generate_entity_tracking_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_distance: int = 1,
    max_distance: int = 4,
) -> list[dict]:
    """setup(entity attribute) → distractor × Δτ → user 用代词指代 recall。

    关键区别于 third_party turn kind：third_party 是一次性 fact/recall；
    entity_tracking 是"引入他 → 话题跑偏 distractor → 用代词 recall"。
    """
    dtau = rng.randint(min_distance, max_distance)

    # 构造第三方 entity
    relation = rng.choice(_THIRD_PARTY_RELATIONS)
    entity_name = random_name(rng)
    tp_key = f"{relation}{entity_name}"

    # 挑一个 attribute slot（job/age/city/hobby）
    attr_slot = rng.choice(["job", "age", "city", "hobby"])
    if attr_slot not in ENTITY_FACT_TEMPLATES:
        return []

    # 采 attribute 值
    from xinhe.data.persona import _sample_slot
    attr_value = _sample_slot(rng, attr_slot)

    # setup
    setup_u_tpl, setup_a_tpl = rng.choice(ENTITY_FACT_TEMPLATES[attr_slot])
    turns = [{
        "user": setup_u_tpl.format(e=tp_key, ea=tp_key, v=attr_value),
        "assistant": setup_a_tpl.format(e=tp_key, ea=tp_key, v=attr_value),
        "train_loss": True,
        "value": attr_value,
    }]

    # distractor × dtau
    fallback = persona or sample_persona(rng, num_reveal=1)
    for _ in range(dtau):
        turns.append(_distractor(rng, cache, fallback))

    # recall 用代词："他" / "她" / "他最近..."
    pronouns = ["他", "她"] if attr_slot != "pet" else ["那只"]
    pronoun = rng.choice(pronouns)
    recall_templates = {
        "job": [
            (f"{pronoun}是干啥的来着？", f"{pronoun}是{attr_value}。"),
            (f"{pronoun}的工作是什么？", f"{pronoun}是{attr_value}。"),
            (f"{pronoun}最近在做什么？", f"{pronoun}是{attr_value}。"),
        ],
        "age": [
            (f"{pronoun}多大来着？", f"{pronoun}{attr_value}岁。"),
            (f"{pronoun}几岁？", f"{pronoun}{attr_value}岁。"),
        ],
        "city": [
            (f"{pronoun}在哪来着？", f"{pronoun}在{attr_value}。"),
            (f"{pronoun}住哪？", f"{pronoun}住在{attr_value}。"),
        ],
        "hobby": [
            (f"{pronoun}平时喜欢干啥？", f"{pronoun}喜欢{attr_value}。"),
            (f"{pronoun}爱好是什么？", f"{pronoun}的爱好是{attr_value}。"),
        ],
    }
    pool = recall_templates.get(attr_slot)
    if not pool:
        return []
    recall_u, recall_a = rng.choice(pool)
    turns.append({
        "user": recall_u,
        "assistant": recall_a,
        "train_loss": True,
        "value": attr_value,
    })
    return turns


# ═══════════════════════════════════════════════════════════════════
# Val 生成器（eval_fn 由 persona_joint.py 注入）
# ═══════════════════════════════════════════════════════════════════

def _build_val_gen(pattern_fn):
    def _gen(rng: random.Random, cache, n_samples: int) -> list[list[dict]]:
        episodes = []
        attempts = 0
        while len(episodes) < n_samples and attempts < n_samples * 4:
            attempts += 1
            ep = pattern_fn(rng, None, cache)
            if ep and ep[-1].get("train_loss") and ep[-1].get("value"):
                episodes.append(ep)
        return episodes
    return _gen


register_val("reference_back")(_build_val_gen(generate_reference_back_episode))
register_val("context_followup")(_build_val_gen(generate_context_followup_episode))
register_val("topic_continuation")(_build_val_gen(generate_topic_continuation_episode))
register_val("entity_tracking")(_build_val_gen(generate_entity_tracking_episode))
