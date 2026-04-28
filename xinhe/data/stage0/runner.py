"""Stage 0 数据集生成入口。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from xinhe.data.io import write_jsonl
from xinhe.data.schema import Sample
from xinhe.data.skeletons.library import sample_skeleton, SKELETONS
from xinhe.data.skeletons.runner import SkeletonRunner
from xinhe.data.warmup import inject_world_qa_warmup


def _stage0_warmup_k(rng: random.Random, sample: Sample, *, max_turns: int) -> int:
    """Stage 0 warmup K 分布：rng.randint(1, 余量)，余量 ≤ 0 时返回 0。

    余量 = max_turns - n_turns(原生 episode 已用)，把 padding 槽用 world_qa 填上。
    """
    margin = max_turns - sample.meta.get("n_turns", 0)
    if margin <= 0:
        return 0
    return rng.randint(1, margin)


def generate_stage0_episode(
    rng: random.Random,
    *,
    skeleton_id: Optional[str] = None,
    skeleton_weights: Optional[dict[str, float]] = None,
    dict_split: str = "train",
    distance_distribution: Optional[dict[str, float]] = None,
    weight_table: Optional[dict] = None,
    max_turns: int = 12,
    inject_warmup: bool = True,
) -> Sample:
    """生成一条 Stage 0 样本。

    Args:
        max_turns: 单 episode 最大 user-asst pair 数；同时是 SkeletonRunner 的 budget 上限
                   和 warmup K 的余量上限。必须 = dataloader max_turns_per_episode（1:1 对齐）。
        inject_warmup: 是否在生成完 skeleton 后前置 world_qa warmup pair（默认 True）。
                       关闭用于 unit test 检验原始 skeleton 输出。
    """
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
    )
    sample = runner.run(skeleton, rng)

    if inject_warmup:
        inject_world_qa_warmup(
            sample,
            rng,
            dict_split=dict_split,
            k_sampler=lambda r, s: _stage0_warmup_k(r, s, max_turns=max_turns),
            train_loss="lm_only",
        )

    return sample


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
    inject_warmup: bool = True,
) -> tuple[int, int]:
    """批量生成 Stage 0 数据集到 jsonl。

    Returns:
        (n_ok, n_rejected)
    """
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
                    inject_warmup=inject_warmup,
                )
            except Exception:
                continue  # 罕见情况：池空或事件 run 抛错；跳过
            if not sample.conversations:
                continue
            yield sample
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  [stage0] {i + 1}/{n_samples}")

    return write_jsonl(_gen(), out_path, rejected_path=rejected_path)
