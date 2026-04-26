"""S1..S11 骨架定义。

参考文档 §6 骨架池：
  S1  [A, {En}, B]                            — 基础读写 + 长程保持
  S2  [A, C, {En}, B]                         — 拒答不污染后续读
  S3  [F, G, H]                               — 并发写入与局部/全量召回
  S4  [A, D, B]                               — 基础擦写
  S5  [A, D, {En}, K]                         — 擦写后 Stale-Read 对抗
  S6  [A, D, {En}, M]                         — 覆盖后旧值查询的"已不再是"信号
  S7  [F, D_partial, {En}, H]                 — 多实体中只覆盖一项,长距后多读全部
  S8  [I, C, G]                               — 第三方实体绑定与拒答
  S9  [A, J, {En}, H]                         — Augment 与 Multi-Read 联用,防止 J 学成 D
  S10 [A, {En_long}, L, {En_short}, C_prime]  — Reverse-Erase 后查询应得 Miss
  S11 [F, L_partial, {En}, H, C_prime]        — 局部遗忘,防止学成整体拒答
"""
from __future__ import annotations

import random
from typing import Optional

from xinhe.data.skeletons.spec import Skeleton, DistractGroup


_DG = DistractGroup()
_DG_LONG = DistractGroup(bucket_constraint=None, label="DG_long")           # 默认全分布
_DG_SHORT = DistractGroup(bucket_constraint="near", max_turns=2, label="DG_short")  # 0-2 轮


SKELETONS: dict[str, Skeleton] = {
    "S1":  Skeleton("S1",  ["A", _DG, "B"], weight=1.0,
                    description="基础读写 + 长程保持"),
    "S2":  Skeleton("S2",  ["A", "C", _DG, "B"], weight=0.6,
                    description="拒答不污染后续读"),
    "S3":  Skeleton("S3",  ["F", "G", "H"], weight=0.8,
                    description="并发写入与局部/全量召回"),
    "S4":  Skeleton("S4",  ["A", "D", "B"], weight=0.8,
                    description="基础擦写"),
    "S5":  Skeleton("S5",  ["A", "D", _DG, "K"], weight=1.0,
                    description="擦写后 Stale-Read 对抗"),
    "S6":  Skeleton("S6",  ["A", "D", _DG, "M"], weight=0.6,
                    description="覆盖后旧值查询"),
    "S7":  Skeleton("S7",  ["F", "D_partial", _DG, "H"], weight=1.0,
                    description="多实体中只覆盖一项,长距后多读"),
    "S8":  Skeleton("S8",  ["I", "C", "G"], weight=0.5,
                    description="第三方实体绑定与拒答"),
    "S9":  Skeleton("S9",  ["A", "J", _DG, "H"], weight=0.6,
                    description="Augment 与 Multi-Read 联用"),
    "S10": Skeleton("S10", ["A", _DG_LONG, "L", _DG_SHORT, "C_prime"], weight=0.8,
                    description="Reverse-Erase 后查询应得 Miss"),
    "S11": Skeleton("S11", ["F", "L_partial", _DG, "H", "C_prime"], weight=0.8,
                    description="局部遗忘,防止学成整体拒答"),
}


def get_skeleton(skid: str) -> Skeleton:
    if skid not in SKELETONS:
        raise KeyError(f"未知 skeleton: {skid!r}")
    return SKELETONS[skid]


def sample_skeleton(
    rng: random.Random,
    *,
    weights: Optional[dict[str, float]] = None,
) -> Skeleton:
    """加权采样。weights=None 用 SKELETONS 中默认权重。"""
    if weights is None:
        weights = {sid: sk.weight for sid, sk in SKELETONS.items()}
    sids = list(weights.keys())
    ws = [max(0.0, weights[s]) for s in sids]
    if sum(ws) <= 0:
        raise ValueError("骨架权重全为 0")
    chosen = rng.choices(sids, weights=ws, k=1)[0]
    return SKELETONS[chosen]
