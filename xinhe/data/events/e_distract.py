"""E (Distract):闲聊 / 推理段干扰段。

train_loss="lm_only":lm_weight=0.3,让 backbone 保持流畅但抑制 W 写入门 β。
不调用 state.apply,不污染 MemoryState。

数据源(短句模式 ctx.distract_expansion=="short"):
  优先从 xinhe/data/dicts/files/distract_chat.jsonl 抽(按 ctx.dict_split 切分)。
  若该语料缺失或为空,fallback 到 templates/e_distract.py 的内置 POOL。

数据源(段落模式 ctx.distract_expansion=="paragraph"):
  从 xinhe/data/dicts/files/congliu_distract.jsonl 抽 1-3 个 R1 推理样本,
  把它们的 assistant 文本用 \\n\\n 拼成单个长 assistant turn(长 reasoning 段),
  user 端用 topic-prompt 包装。目标 char 长度 ≈ paragraph_token_target × 1.5。
  若 congliu_distract 缺失,fallback 到短句模式。
"""
from __future__ import annotations

import random

from xinhe.data.dicts.bank import load_pairs
from xinhe.data.events.base import (
    AtomicEvent,
    ConvPair,
    EventContext,
    make_assistant_turn,
    make_user_turn,
    register_event,
)
from xinhe.data.memory_state import MemoryState
from xinhe.data.templates.e_distract import POOL


_PAIR_BANK_CACHE: dict[tuple, object] = {}


def _try_load_corpus(name: str, split: str):
    """缓存式加载某 pair 语料;不存在或为空则返 None。"""
    key = (name, split)
    if key in _PAIR_BANK_CACHE:
        return _PAIR_BANK_CACHE[key]
    try:
        pb = load_pairs(name, split=split)
    except (FileNotFoundError, ValueError):
        pb = None
    _PAIR_BANK_CACHE[key] = pb
    return pb


_PARAGRAPH_USER_PROMPTS = [
    "我刚遇到一个有意思的问题,你帮我想想看。",
    "我在思考一个问题,可以听听你的想法吗?",
    "我看到一道题,想知道你怎么分析。",
    "我有个疑问想跟你讨论一下。",
    "可以帮我想想这道题怎么解吗?",
    "我对一个问题很好奇,想听你的看法。",
]


def _run_paragraph(
    rng: random.Random,
    ctx: EventContext,
    target_tokens: int,
) -> list[ConvPair]:
    """段落模式:从 congliu_distract 拼 1-3 个 R1 推理段成长 assistant turn。

    fallback 顺序:congliu_distract → distract_chat 的多个短句拼接 → POOL 短句单条。
    """
    target_chars = int(target_tokens * 1.5)  # 中文字符 ≈ 1.5×token

    pb = _try_load_corpus("congliu_distract", ctx.dict_split)
    if pb is not None and len(pb) > 0:
        chunks = []
        total = 0
        # 最多拼 3 个,达到 target_chars 即停
        for _ in range(3):
            pair = pb.sample_one(rng)
            asst_text = pair["assistant"]
            chunks.append(asst_text)
            total += len(asst_text)
            if total >= target_chars:
                break
        user_text = rng.choice(_PARAGRAPH_USER_PROMPTS)
        asst_text = "\n\n".join(chunks)
        u = make_user_turn(user_text)
        a = make_assistant_turn(asst_text, train_loss="lm_only")
        return [(u, a)]

    # fallback A:用 distract_chat 多条短句拼一段(增长比单条长)
    pb = _try_load_corpus("distract_chat", ctx.dict_split)
    if pb is not None and len(pb) > 0:
        chunks = []
        total = 0
        for _ in range(8):
            pair = pb.sample_one(rng)
            chunks.append(pair["assistant"])
            total += len(pair["assistant"])
            if total >= target_chars:
                break
        user_text = rng.choice(_PARAGRAPH_USER_PROMPTS)
        asst_text = "\n\n".join(chunks)
        u = make_user_turn(user_text)
        a = make_assistant_turn(asst_text, train_loss="lm_only")
        return [(u, a)]

    # fallback B:POOL 单条
    tmpl = rng.choice(POOL.templates)
    u = make_user_turn(tmpl.user_text)
    a = make_assistant_turn(tmpl.asst_text, train_loss="lm_only")
    return [(u, a)]


def _run_short(
    rng: random.Random,
    ctx: EventContext,
) -> list[ConvPair]:
    """短句模式:原行为 — distract_chat 单条 pair 或 POOL fallback。"""
    pb = _try_load_corpus("distract_chat", ctx.dict_split)
    if pb is not None and len(pb) > 0:
        pair = pb.sample_one(rng)
        user_text = pair["user"]
        asst_text = pair["assistant"]
    else:
        tmpl = rng.choice(POOL.templates)
        user_text = tmpl.user_text
        asst_text = tmpl.asst_text

    u = make_user_turn(user_text)
    a = make_assistant_turn(asst_text, train_loss="lm_only")
    return [(u, a)]


@register_event("E")
class DistractEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        if getattr(ctx, "distract_expansion", "short") == "paragraph":
            target = int(getattr(ctx, "distract_paragraph_token_target", 300))
            return _run_paragraph(rng, ctx, target)
        return _run_short(rng, ctx)
