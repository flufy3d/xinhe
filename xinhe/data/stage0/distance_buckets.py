"""距离桶采样：决定 {En} 段插入多少轮 distract。"""
from __future__ import annotations

import random
from typing import Optional


# 文档 §7
BUCKETS = {
    "near":     {"turns": (1, 1), "p": 0.20},
    "mid":      {"turns": (2, 3), "p": 0.35},
    "far":      {"turns": (4, 6), "p": 0.30},
    "very_far": {"turns": (7, 9), "p": 0.15},
}


def sample_distance_bucket(
    rng: random.Random,
    *,
    bucket_constraint: Optional[str] = None,
    distribution: Optional[dict[str, float]] = None,
) -> tuple[str, int]:
    """返回 (bucket_name, n_turns)。

    bucket_constraint 强制指定桶；否则按 distribution（默认 BUCKETS 概率）。
    n_turns 在桶的 turns 区间均匀采样。
    """
    if bucket_constraint:
        if bucket_constraint not in BUCKETS:
            raise ValueError(f"未知桶: {bucket_constraint}")
        name = bucket_constraint
    else:
        if distribution is None:
            distribution = {k: v["p"] for k, v in BUCKETS.items()}
        names = list(distribution.keys())
        weights = [distribution[n] for n in names]
        name = rng.choices(names, weights=weights, k=1)[0]

    lo, hi = BUCKETS[name]["turns"]
    n = rng.randint(lo, hi)
    return name, n
