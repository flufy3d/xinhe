"""SkeletonGenerator:本地骨架驱动 (原 stage0),无 LLM 调用。"""
from __future__ import annotations

from typing import Optional

from xinhe.data.generators.base import Generator, GenerateRequest
from xinhe.data.generators.skeleton.runner import generate_stage0_dataset


class SkeletonGenerator(Generator):
    name = "skeleton"

    def __init__(
        self,
        *,
        max_turns: int,
        skeleton_weights: Optional[dict[str, float]] = None,
        distance_bucket: Optional[dict[str, float]] = None,
        force_relation: Optional[str] = None,
    ):
        self.max_turns = max_turns
        self.skeleton_weights = skeleton_weights
        self.distance_bucket = distance_bucket
        self.force_relation = force_relation

    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]:
        return generate_stage0_dataset(
            req.out_path,
            n_samples=req.n_samples,
            seed=req.seed,
            skeleton_weights=self.skeleton_weights,
            force_relation=self.force_relation,
            dict_split=req.split,
            distance_distribution=self.distance_bucket,
            rejected_path=req.rejected_path,
            progress_every=max(1, req.n_samples // 10),
            max_turns=req.max_turns,
        )
