"""E (Distract)：闲聊干扰段。

train_loss="lm_only"：lm_weight=0.3，让 backbone 保持流畅但抑制 W 写入门 β。
不调用 state.apply，不污染 MemoryState。

数据源：
  优先从 xinhe/data/dicts/files/distract_chat.jsonl 抽（按 ctx.dict_split 切分）。
  若该语料缺失或为空，fallback 到 templates/e_distract.py 的内置 POOL。
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


_PAIR_BANK_CACHE: dict[str, object] = {}


def _try_load_corpus(split: str):
    """缓存式加载 distract_chat 语料；不存在或为空则返 None。"""
    if split in _PAIR_BANK_CACHE:
        return _PAIR_BANK_CACHE[split]
    try:
        pb = load_pairs("distract_chat", split=split)
    except (FileNotFoundError, ValueError):
        pb = None
    _PAIR_BANK_CACHE[split] = pb
    return pb


@register_event("E")
class DistractEvent(AtomicEvent):
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        pb = _try_load_corpus(ctx.dict_split)
        if pb is not None and len(pb) > 0:
            pair = pb.sample_one(rng)
            user_text = pair["user"]
            asst_text = pair["assistant"]
        else:
            tmpl = rng.choice(POOL.templates)
            user_text = tmpl.user_text
            asst_text = tmpl.asst_text

        u = make_user_turn(user_text)
        # train_loss="lm_only":保流畅但不让 W 抢资源
        a = make_assistant_turn(asst_text, train_loss="lm_only")
        return [(u, a)]
