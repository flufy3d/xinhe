"""
patterns/phrase.py —— 随机短语类 episode（非 persona 槽的整段 retention）

2 个 pattern：
  verbatim_recall       —— 记住随机字母数字 phrase → distractor → 原样复述
  adversarial_temporal  —— 3 个随机 phrase + 按时序选一（0b/Stage 1 stress）

核心：phrase 是运行时生成的随机 alnum 序列，backbone LM 无先验可猜，
必须从 W 真实检索 → 强制 Delta Rule 读写路径跑通。
"""
from __future__ import annotations
import random
from typing import Optional

from xinhe.data.persona import Persona, sample_persona
from xinhe.data.templates import (
    VERBATIM_PHRASES, VERBATIM_SETUP_TEMPLATES, VERBATIM_SETUP_ACKS,
    VERBATIM_RECALL_USER_TEMPLATES, VERBATIM_RECALL_ASST_TEMPLATES,
    ADVERSARIAL_SETUP_TEMPLATES,
    ADVERSARIAL_ORDINAL_QUERY_EARLIEST,
    ADVERSARIAL_ORDINAL_QUERY_MIDDLE,
    ADVERSARIAL_ORDINAL_QUERY_LATEST,
    ADVERSARIAL_DISTANCE_QUERY_TEMPLATES,
)
from xinhe.data.registry import register_pattern, register_val, get_turn_kind


_ALNUM_POOL = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _pick_phrase(rng: random.Random) -> str:
    """固定 8 字符的随机字母+数字序列，空格分隔（防 BPE 合并）。
    配合 setup asst echo，模型学会存取任意随机序列。
    """
    return _random_alphanumeric_phrase(rng, min_len=8, max_len=8)


def _random_alphanumeric_phrase(rng: random.Random, min_len: int = 8, max_len: int = 12) -> str:
    """[adversarial_temporal 专用] 随机字母+数字字符串，W 硬验证用。"""
    length = rng.randint(min_len, max_len)
    chars = [rng.choice(_ALNUM_POOL) for _ in range(length)]
    return " ".join(chars)


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


@register_pattern("verbatim_recall")
def generate_verbatim_recall_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    min_distance: int = 0,
    max_distance: int = 0,
) -> list[dict]:
    """2-turn 纯净版: setup (告知 phrase) + recall (询问 phrase)。

    默认无 distractor（max_distance=0）→ episode 2 turn 全 train_loss=True，
    每 turn 梯度有效，不浪费 GPU；每 episode 一半告知+一半询问。

    max_distance>0 时在 setup 和 recall 之间插入 dtau 个 distractor turn（pressure 版），
    用于 Stage 1 retention 压力测试。

    关键设计：
    - setup asst 是 **无 phrase echo 的 ack**（避免模型学 copy-user 捷径）
    - recall asst 用多样模板（含 {phrase}），eval 只检查 phrase 字符 span
    """
    dtau = rng.randint(min_distance, max_distance) if max_distance > 0 else 0
    phrase = _pick_phrase(rng)

    setup_u = rng.choice(VERBATIM_SETUP_TEMPLATES).format(phrase=phrase)
    setup_a = rng.choice(VERBATIM_SETUP_ACKS)   # 不含 phrase
    recall_u = rng.choice(VERBATIM_RECALL_USER_TEMPLATES)
    recall_a = rng.choice(VERBATIM_RECALL_ASST_TEMPLATES).format(phrase=phrase)

    fallback = persona or sample_persona(rng, num_reveal=1)
    turns: list[dict] = [
        {"user": setup_u, "assistant": setup_a, "train_loss": True},
    ]
    for _ in range(dtau):
        turns.append(_distractor(rng, cache, fallback))
    turns.append({
        "user": recall_u,
        "assistant": recall_a,
        "train_loss": True,
        "value": phrase,
    })
    return turns


@register_pattern("adversarial_temporal")
def generate_adversarial_temporal_episode(
    rng: random.Random,
    persona: Optional[Persona] = None,
    cache=None,
    n_entries: int = 3,
    phase_max: int = 5,
) -> list[dict]:
    """setup₀ - filler - setup₁ - filler - setup₂ - [trailing] - recall。ordinal 60% / distance 40%。"""
    if n_entries != 3:
        raise ValueError(f"仅支持 n_entries=3")

    phrases = [_random_alphanumeric_phrase(rng) for _ in range(n_entries)]
    setup_tpls = rng.sample(ADVERSARIAL_SETUP_TEMPLATES, n_entries)

    fallback = persona or sample_persona(rng, num_reveal=1)
    turns: list[dict] = []
    setup_turn_idx: list[int] = []

    for i in range(n_entries):
        setup_u_tpl, setup_a = setup_tpls[i]
        turns.append({
            "user": setup_u_tpl.format(phrase=phrases[i]),
            "assistant": setup_a,
            "train_loss": False,
        })
        setup_turn_idx.append(len(turns) - 1)
        if i < n_entries - 1:
            turns.append(_distractor(rng, cache, fallback))

    if rng.random() < 0.5:
        turns.append(_distractor(rng, cache, fallback))

    recall_turn_idx = len(turns)
    target_i = rng.randint(0, n_entries - 1)
    target_phrase = phrases[target_i]
    dtau = recall_turn_idx - setup_turn_idx[target_i] - 1

    if dtau > phase_max:
        return []

    use_distance = (rng.random() < 0.4) and (dtau >= 1)
    if use_distance:
        recall_u = rng.choice(ADVERSARIAL_DISTANCE_QUERY_TEMPLATES).format(dtau=dtau)
    else:
        if target_i == 0:
            pool = ADVERSARIAL_ORDINAL_QUERY_EARLIEST
        elif target_i == n_entries - 1:
            pool = ADVERSARIAL_ORDINAL_QUERY_LATEST
        else:
            pool = ADVERSARIAL_ORDINAL_QUERY_MIDDLE
        recall_u = rng.choice(pool)

    turns.append({
        "user": recall_u,
        "assistant": target_phrase,
        "train_loss": True,
        "value": target_phrase,
    })
    return turns


# ── Val 生成器（eval_fn 由 persona_joint.py 注入）──

def _verbatim_val_gen(rng: random.Random, cache, n_samples: int) -> list[list[dict]]:
    episodes = []
    attempts = 0
    while len(episodes) < n_samples and attempts < n_samples * 4:
        attempts += 1
        ep = generate_verbatim_recall_episode(rng, None, cache)
        if ep and ep[-1].get("value"):
            episodes.append(ep)
    return episodes


register_val("verbatim")(_verbatim_val_gen)
