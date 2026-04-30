"""SkeletonRunner：执行一个骨架，输出完整 Sample dict。

执行流程:
  1. 创建 MemoryState + EventContext。
  2. 按 skeleton.sequence 顺序执行：
     - 字符串 → get_event(...).run(...) 追加 conv pair
     - DistractGroup → 按桶采 N 轮，每轮跑一次 E
  3. 总轮数受 (mu=8, sigma=2, lo=4, hi=max_turns) 软约束：实际轮数会因事件 emit 数变化，
     SkeletonRunner 不强行截断（事件返回的 ConvPair 列表是原子的）；
     若某些 skeleton 的最低事件需求 > target_turns，仍执行（保完整性）。
  4. 把 distract 桶记到 meta["distance_bucket"]。

  max_turns 必须 = dataloader max_turns_per_episode（1 turn → 1 tensor，1:1 对齐），
  否则 dataloader 会静默截断。caller 由 stage 配置 (max_turns_per_episode) 显式传入；
  validate_stage_config 会拦截不一致。
"""
from __future__ import annotations

import random
import uuid
from typing import Optional

from xinhe.data.dicts.bank import dict_version
from xinhe.data.events import get_event
from xinhe.data.events.base import EventContext
from xinhe.data.memory_state import MemoryState
from xinhe.data.schema import Sample
from xinhe.data.skeletons.spec import Skeleton, DistractGroup
from xinhe.data.shared.distance_buckets import sample_distance_bucket
from xinhe.data.shared.turn_count import sample_target_turns


class SkeletonRunner:
    def __init__(
        self,
        *,
        dict_split: str = "train",
        stage: str = "0",
        distance_distribution: Optional[dict[str, float]] = None,
        weight_table: Optional[dict] = None,
        max_turns: int = 12,
        force_relation: Optional[str] = None,
    ) -> None:
        self.dict_split = dict_split
        self.stage = stage
        self.distance_distribution = distance_distribution
        self.weight_table = weight_table
        self.max_turns = max_turns
        self.force_relation = force_relation

    def run(
        self,
        skeleton: Skeleton,
        rng: random.Random,
        *,
        sample_id: Optional[str] = None,
    ) -> Sample:
        ctx = EventContext(
            dict_split=self.dict_split,
            stage=self.stage,
            weight_table=self.weight_table or {
                ("0", "hard"): 5.0,
                ("1", "hard"): 3.0,
                ("1", "soft"): 1.5,
            },
        )
        if self.force_relation:
            ctx.canonical_pool["__force_relation"] = self.force_relation
        state = MemoryState()

        target_turns = sample_target_turns(rng, hi=self.max_turns)
        conversations: list[dict] = []
        bucket_record: dict = {}
        bucket_main: Optional[str] = None

        for i, slot in enumerate(skeleton.sequence):
            if isinstance(slot, DistractGroup):
                # 距离桶 → N 轮 E
                bucket, n_distract = sample_distance_bucket(
                    rng,
                    bucket_constraint=slot.bucket_constraint,
                    distribution=self.distance_distribution,
                )
                if slot.max_turns is not None:
                    n_distract = min(n_distract, slot.max_turns)
                # 软约束：若已有事件 + n_distract 会让总轮数超 max_turns，缩减
                events_remaining = sum(
                    1 for s in skeleton.sequence[i + 1:] if isinstance(s, str)
                )
                budget = max(0, self.max_turns - len(conversations) // 2 - events_remaining)
                if n_distract > budget:
                    n_distract = budget
                if n_distract < 0:
                    n_distract = 0

                E = get_event("E")
                for k in range(n_distract):
                    pairs = E.run(rng, state, ctx, turn_idx=len(conversations) // 2)
                    for u, a in pairs:
                        conversations.append(u)
                        conversations.append(a)
                # 记录桶位
                if bucket_main is None:
                    bucket_main = bucket
                bucket_record[slot.label] = {"bucket": bucket, "turns": n_distract}
            else:
                evt = get_event(slot)
                pairs = evt.run(rng, state, ctx, turn_idx=len(conversations) // 2)
                for u, a in pairs:
                    conversations.append(u)
                    conversations.append(a)

        n_turns = len(conversations) // 2
        meta = {
            "n_turns": n_turns,
            "target_turns": target_turns,
            "distance_bucket": bucket_main,
            "distance_buckets_detail": bucket_record,
            "seed": getattr(rng, "_seed", None),
            "dict_version": dict_version(),
            "dict_split": self.dict_split,
            "memory_snapshot": state.snapshot(),
        }
        return Sample(
            sample_id=sample_id or uuid.uuid4().hex[:12],
            stage=self.stage,
            skeleton_id=skeleton.id,
            meta=meta,
            conversations=conversations,
        )
