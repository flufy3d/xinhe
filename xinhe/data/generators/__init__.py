"""数据生成器注册表。

加新生成器 = 在这个 dict 里加一行(import 子类) + 在 generators/<kind>/ 写实现。
dispatcher (scripts/generate_data.py) 不需要改。
"""
from __future__ import annotations

from xinhe.data.generators.base import Generator, GenerateRequest
from xinhe.data.generators.skeleton.generator import SkeletonGenerator
from xinhe.data.generators.dialog.generator import DialogGenerator

GENERATORS: dict[str, type[Generator]] = {
    "skeleton": SkeletonGenerator,
    "dialog": DialogGenerator,
}

__all__ = ["GENERATORS", "Generator", "GenerateRequest"]
