"""S1..S11 骨架定义(注:S6 已合并入 S5,实际有效骨架为 10 个 + S_simple)。

参考文档 §6 骨架池:
  S1  [A, {En}, B]                            — 基础读写 + 长程保持
  S2  [A, C, {En}, B]                         — 拒答不污染后续读
  S3  [F, G, H]                               — 并发写入与局部/全量召回
  S4  [A, D, B]                               — 基础擦写
  S5  [A, D, {En}, K]                         — 擦写后 Stale-Query(陈述/疑问两种句式都在 K 池)
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


# v9.5:含 {En} 的 skeleton 切到 paragraph DistractGroup,episode 真正长起来
# (paper Titans MAC 在长 context 下 NM 通路才有用武之地;短 distract 让 attention 直接看到全文)
_DG_PARA = DistractGroup(expansion="paragraph", paragraph_token_target=300, label="DG")
_DG_LONG_PARA = DistractGroup(expansion="paragraph", paragraph_token_target=400, label="DG_long")
_DG_SHORT = DistractGroup(bucket_constraint="near", max_turns=2, label="DG_short",
                          expansion="short")  # S10 短段保留原行为(对照,paper-style 短跨度)


SKELETONS: dict[str, Skeleton] = {
    "S1":  Skeleton("S1",  ["A", _DG_PARA, "B"], weight=1.0,
                    description="基础读写 + 长程保持"),
    "S2":  Skeleton("S2",  ["A", "C", _DG_PARA, "B"], weight=0.6,
                    description="拒答不污染后续读"),
    "S3":  Skeleton("S3",  ["F", "G", "H"], weight=0.8,
                    description="并发写入与局部/全量召回"),
    "S4":  Skeleton("S4",  ["A", "D", "B"], weight=0.8,
                    description="基础擦写"),
    "S5":  Skeleton("S5",  ["A", "D", _DG_PARA, "K"], weight=1.6,
                    description="擦写后 Stale-Query(原 S5+S6 合并;K 池含陈述/疑问两种句式)"),
    "S7":  Skeleton("S7",  ["F", "D_partial", _DG_PARA, "H"], weight=1.0,
                    description="多实体中只覆盖一项,长距后多读"),
    "S8":  Skeleton("S8",  ["I", "C", "G"], weight=0.5,
                    description="第三方实体绑定与拒答"),
    "S9":  Skeleton("S9",  ["A", "J", _DG_PARA, "H"], weight=0.6,
                    description="Augment 与 Multi-Read 联用"),
    "S10": Skeleton("S10", ["A", _DG_LONG_PARA, "L", _DG_SHORT, "C_prime"], weight=0.8,
                    description="Reverse-Erase 后查询应得 Miss"),
    "S11": Skeleton("S11", ["F", "L_partial", _DG_PARA, "H", "C_prime"], weight=0.8,
                    description="局部遗忘,防止学成整体拒答"),

    # warmup 专用:纯 reveal+recall,无 distract。课程起点,迫使 W 学到 token 级 verbatim
    "S_simple": Skeleton("S_simple", ["A", "B"], weight=0.0,
                         description="warmup-only:纯 2-turn 写读,验证 W 通路基础"),
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
