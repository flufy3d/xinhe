"""DialogGenerator:LLM 5-Beat 多轮对话 (原 stage1)。

底层 generate_stage1_dataset 已自带 _single_instance_lock 锁守护,
所以 Generator._maybe_lock 不需要 override。
"""
from __future__ import annotations

from typing import Optional

from xinhe.data.generators.base import Generator, GenerateRequest
from xinhe.data.generators.dialog.driver_core import generate_stage1_dataset


class DialogGenerator(Generator):
    name = "dialog"

    def __init__(
        self,
        *,
        max_turns: int,
        model: str = "deepseek-v4-flash",
        workers: int = 4,
        n_canonical_range: tuple[int, int] | list[int] = (1, 3),
        n_turns_range: tuple[int, int] | list[int] = (10, 14),
        beat3_min_turns: int = 1,
        beat3_min_chars: int = 500,
        beat3_chars_tolerance: float = 0.8,
        mix: Optional[dict[str, float]] = None,
        weight_table: Optional[dict] = None,
    ):
        self.max_turns = max_turns          # dialog 实际不用,但 dispatcher 统一传入
        self.model = model
        self.workers = workers
        self.n_canonical_range = tuple(n_canonical_range)
        self.n_turns_range = tuple(n_turns_range)
        self.beat3_min_turns = beat3_min_turns
        self.beat3_min_chars = beat3_min_chars
        self.beat3_chars_tolerance = beat3_chars_tolerance
        self.mix = mix
        self.weight_table = weight_table

    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]:
        return generate_stage1_dataset(
            req.out_path,
            n_samples=req.n_samples,
            seed=req.seed,
            mix=self.mix,
            dict_split=req.split,
            n_canonical_range=self.n_canonical_range,
            n_turns_range=self.n_turns_range,
            beat3_min_turns=self.beat3_min_turns,
            beat3_min_chars=self.beat3_min_chars,
            beat3_chars_tolerance=self.beat3_chars_tolerance,
            workers=self.workers,
            model=self.model,
            rejected_path=req.rejected_path,
            weight_table=self.weight_table,
            resume=req.resume,
        )
