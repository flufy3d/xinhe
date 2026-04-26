"""Stage 1 world_qa 工具：

- wrap_world_qa_episodes: 老 1B 独立流（已废弃但保留供 caller fallback）
- sample_world_qa_pairs: v7+ 新接口，抽 K 个 (user, assistant) pair 给 1A driver 当
  warmup 闲聊插入到 Beat 1 之前，用于打散 fact 位置偏置 + 利用 episode_length。

按 dict_split 切分（与字符串字典共享 SHA1 切分逻辑）。
"""
from __future__ import annotations

import random
import uuid
from typing import Iterator

from xinhe.data.dicts.bank import dict_version, load_pairs
from xinhe.data.schema import Sample


def sample_world_qa_pairs(
    *, k: int, rng: random.Random, dict_split: str = "train",
) -> list[dict]:
    """抽 k 个 world_qa pair,返回 list[{"user": ..., "assistant": ...}]。
    缺失或为空返回 []。"""
    if k <= 0:
        return []
    try:
        pb = load_pairs("world_qa", split=dict_split)
    except (FileNotFoundError, ValueError):
        return []
    pairs = list(pb.pairs)
    if not pairs:
        return []
    rng.shuffle(pairs)
    return pairs[:k] if len(pairs) >= k else (pairs * (1 + k // max(1, len(pairs))))[:k]


def wrap_world_qa_episodes(
    *,
    n_samples: int,
    rng: random.Random,
    dict_split: str = "train",
) -> Iterator[Sample]:
    """从 world_qa 语料抽样 n_samples 条，每条包成 1-turn episode。

    若语料缺失或为空（PairBank 抛异常），生成 0 条（caller 自己 fallback）。
    """
    try:
        pb = load_pairs("world_qa", split=dict_split)
    except (FileNotFoundError, ValueError):
        return
    pairs = list(pb.pairs)
    if not pairs:
        return
    rng.shuffle(pairs)
    if len(pairs) >= n_samples:
        take = pairs[:n_samples]
    else:
        take = pairs * (1 + n_samples // max(1, len(pairs)))
        take = take[:n_samples]
    for p in take:
        conversations = [
            {"role": "user", "content": p["user"]},
            {
                "role": "assistant",
                "content": p["assistant"],
                "train_loss": "true",
                "value": None,
                "value_span": [],
                "value_tier": None,
                "weight_per_span": 0.0,
            },
        ]
        yield Sample(
            sample_id=uuid.uuid4().hex[:12],
            stage="1",
            skeleton_id=None,
            meta={
                "n_turns": 1,
                "substream": "1B",
                "source": "world_qa",
                "dict_version": dict_version(),
                "dict_split": dict_split,
            },
            conversations=conversations,
        )
