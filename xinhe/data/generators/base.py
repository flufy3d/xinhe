"""Generator 基类:提供缓存检测 + 锁守护套壳,子类只实现 _generate_impl。

加新生成器(如基于小说提取):
  1. 在 generators/<kind>/generator.py 里写一个 Generator 子类
  2. 在 generators/__init__.py 里注册到 GENERATORS
不需要改 dispatcher、validator、CLI。
"""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class GenerateRequest:
    """单次 generate 调用的所有运行时参数。"""
    out_path: Path
    n_samples: int
    seed: int
    split: str                  # "train" | "val"
    rejected_path: Path | None
    max_turns: int              # 从 stage_cfg.training.max_turns_per_episode 派生
    force: bool = False
    resume: bool = True


class Generator(ABC):
    """造一种数据的"种类":skeleton (本地骨架) / dialog (LLM) / 未来 novel ...

    yaml data 块里除 kind/out_dir/num_*/seed 之外的字段全部 kwargs 透传到 __init__,
    子类自己接住自己关心的(skeleton 的 skeleton_weights、dialog 的 model 等)。
    拼写错会在 __init__ 抛 TypeError,启动期暴露。
    """
    name: ClassVar[str]

    def __init__(self, **cfg):
        # 基类不消化任何 cfg;子类 override __init__ 接住自己的字段
        if cfg:
            unexpected = ", ".join(sorted(cfg))
            raise TypeError(f"{type(self).__name__}: unexpected config keys: {unexpected}")

    def generate(self, req: GenerateRequest) -> tuple[int, int]:
        """套壳:缓存命中 → 跳过;否则进锁 → 调子类实现。返回 (kept, rejected)。"""
        if not req.force and self._is_cached(req):
            existing = self._count_existing(req.out_path)
            print(f"  [缓存] 跳过 {req.out_path} ({existing} 条)")
            return existing, 0
        with self._maybe_lock(req.out_path):
            return self._generate_impl(req)

    @abstractmethod
    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]: ...

    def _maybe_lock(self, out_path: Path):
        """默认无锁;DialogGenerator 覆写为 _single_instance_lock 防多进程双写。"""
        return contextlib.nullcontext()

    def _is_cached(self, req: GenerateRequest) -> bool:
        """已存在且条数 ≥ 95% 目标 → 视为缓存命中。"""
        if not req.out_path.exists():
            return False
        if req.n_samples <= 0:
            return False
        return self._count_existing(req.out_path) >= req.n_samples * 0.95

    @staticmethod
    def _count_existing(path: Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
