"""轮数采样:截断高斯 μ=8 σ=2,clamp [4, 12]。"""
from __future__ import annotations

import random


def sample_target_turns(
    rng: random.Random,
    *,
    mu: float = 8.0,
    sigma: float = 2.0,
    lo: int = 4,
    hi: int = 12,
) -> int:
    for _ in range(50):
        x = rng.gauss(mu, sigma)
        n = int(round(x))
        if lo <= n <= hi:
            return n
    return max(lo, min(hi, int(round(mu))))
