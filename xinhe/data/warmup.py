"""通用 world_qa warmup 注入。

stage1 driver 与 stage0 runner 共享：在 sample.conversations 前面塞 K 个 world_qa
(user, assistant) pair，标 lm_only / value None。

K 由 caller 提供的 k_sampler 闭包决定（不同 stage 用不同分布）：
  - stage1: rng.choices([0,1,2], [0.4,0.3,0.3])  —— 打散 fact 位置偏置
  - stage0: rng.randint(1, max_turns - n_turns)  —— 利用 padding 余量

train_loss 默认 "lm_only"（与 stage1 driver 现行一致），意图：
  - 让 W 状态跑过 warmup 段产生干扰，但不让 LM 学习信号被通用 QA 主导
  - 保留 0.3× 弱 LM loss，顺带保留通用语言能力
"""
from __future__ import annotations

import random
from typing import Callable

from xinhe.data.schema import Sample
from xinhe.data.shared.world_qa import sample_world_qa_pairs


def inject_world_qa_warmup(
    sample: Sample,
    rng: random.Random,
    *,
    dict_split: str,
    k_sampler: Callable[[random.Random, Sample], int],
    train_loss: str = "lm_only",
) -> None:
    """前置 K 个 world_qa pair 到 sample.conversations，原地修改 sample。

    无副作用 fallback：
      - k_sampler 返回 0 → 不动 sample，meta["warmup_pairs"] = 0
      - world_qa 语料缺失或返回空 → 不动 sample，meta["warmup_pairs"] = 0

    更新 meta:
      - warmup_pairs: 实际注入的 pair 数
      - n_turns:      原 n_turns + warmup_pairs
    """
    k = int(k_sampler(rng, sample))
    if k <= 0:
        sample.meta["warmup_pairs"] = 0
        return

    pairs = sample_world_qa_pairs(k=k, rng=rng, dict_split=dict_split)
    if not pairs:
        sample.meta["warmup_pairs"] = 0
        return

    warmup: list[dict] = []
    for p in pairs:
        warmup.append({"role": "user", "content": p["user"]})
        warmup.append({
            "role": "assistant",
            "content": p["assistant"],
            "train_loss": train_loss,
            "value": None,
            "value_span": [],
            "value_tier": None,
            "weight_per_span": 0.0,
        })

    sample.conversations = warmup + list(sample.conversations)
    sample.meta["n_turns"] = sample.meta.get("n_turns", 0) + len(pairs)
    sample.meta["warmup_pairs"] = len(pairs)
