"""Stage 0 数据集生成入口(v9: 无 warmup 注入)。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from xinhe.data.io import write_jsonl
from xinhe.data.schema import Sample
from xinhe.data.skeletons.library import sample_skeleton, SKELETONS
from xinhe.data.skeletons.runner import SkeletonRunner


def generate_stage0_episode(
    rng: random.Random,
    *,
    skeleton_id: Optional[str] = None,
    skeleton_weights: Optional[dict[str, float]] = None,
    dict_split: str = "train",
    distance_distribution: Optional[dict[str, float]] = None,
    weight_table: Optional[dict] = None,
    max_turns: int = 12,
    force_relation: Optional[str] = None,
) -> Sample:
    """生成一条 Stage 0 样本。"""
    if skeleton_id:
        skeleton = SKELETONS[skeleton_id]
    else:
        skeleton = sample_skeleton(rng, weights=skeleton_weights)
    runner = SkeletonRunner(
        dict_split=dict_split,
        stage="0",
        distance_distribution=distance_distribution,
        weight_table=weight_table,
        max_turns=max_turns,
        force_relation=force_relation,
    )
    return runner.run(skeleton, rng)


def generate_stage0_dataset(
    out_path: str | Path,
    *,
    n_samples: int,
    seed: int,
    skeleton_weights: Optional[dict[str, float]] = None,
    dict_split: str = "train",
    distance_distribution: Optional[dict[str, float]] = None,
    rejected_path: str | Path | None = None,
    progress_every: int = 1000,
    max_turns: int = 12,
    force_relation: Optional[str] = None,
) -> tuple[int, int]:
    """批量生成 Stage 0 数据集到 jsonl。"""
    rng = random.Random(seed)

    def _gen():
        for i in range(n_samples):
            try:
                sample = generate_stage0_episode(
                    rng,
                    skeleton_weights=skeleton_weights,
                    dict_split=dict_split,
                    distance_distribution=distance_distribution,
                    max_turns=max_turns,
                    force_relation=force_relation,
                )
            except Exception:
                continue
            if not sample.conversations:
                continue
            yield sample
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [skeleton] {i + 1}/{n_samples}")

    return write_jsonl(_gen(), out_path, rejected_path=rejected_path)
