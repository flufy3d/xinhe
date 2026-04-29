"""
原子事件基类与注册表。

每个事件模块定义一个 AtomicEvent 子类，并通过 @register_event 装饰自动注册。

事件输出 ConvPair 列表（user + assistant 一对一对），其中 assistant 字典已带：
  - content
  - train_loss ("true" / "lm_only" / "false")
  - value (list[str] 或 None)
  - value_span (list[[start, end]] 在 content 字符坐标系下)
  - value_tier ("hard" / "soft" / None)
  - weight_per_span (float, 0 if no value)

事件内部职责：
  1. 从 ctx.bank 选 entity / value
  2. 调 state.apply(...) 写 MemoryState
  3. 渲染 user / assistant 文本
  4. 计算 char span（用 content.index(value) 兜底）
  5. 决定 value_tier + weight_per_span（hard 默认 5.0 / Stage 1 hard 3.0；
     由 ctx.weight_table 给出，per-sample value 数量平均守恒）

事件不主动跑 validator —— validator 在样本完成后整体校验。
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from xinhe.data.memory_state import MemoryState, Key
from xinhe.data.dicts.bank import EntityBank, load_bank


ConvPair = tuple[dict, dict]   # (user_msg, assistant_msg)


# ── 事件运行上下文 ──

@dataclass
class EventContext:
    """事件运行时所需的外部依赖。

    单条样本生成期间共享同一个 EventContext，事件之间通过其中的 bank / canonicals
    协调实体复用（例如 K 用 D 写过的旧值，F 后的 H 复用 F 写过的多 key）。

    Fields:
        dict_split:    "train" / "val" / "test"
        stage:         "0" / "1"
        weight_table:  按 (stage, tier) → 基础 VALUE_WEIGHT
        canonical_pool:  当前样本共享的 (key, surface) 池 —— skeleton 内事件复用
        used_keys:     当前样本已被 active 写过的 key 集合（H/G 从中选）
    """
    dict_split: str = "train"
    stage: str = "0"
    weight_table: dict = field(default_factory=lambda: {
        ("0", "hard"): 5.0,
        ("1", "hard"): 3.0,
        ("1", "soft"): 1.5,
    })
    canonical_pool: dict = field(default_factory=dict)  # key -> surface
    used_keys: list[Key] = field(default_factory=list)
    erased_keys: list[Key] = field(default_factory=list)

    # 内部 lazy bank 缓存
    _banks: dict[str, EntityBank] = field(default_factory=dict)

    def bank(self, category: str):
        """返回 EntityBank 或 duck-typed 等价物(如 SyntheticNameBank)。"""
        if category not in self._banks:
            if category == "synthetic_full_name":
                from xinhe.data.events._relations import SyntheticNameBank
                self._banks[category] = SyntheticNameBank(self.dict_split)
            else:
                self._banks[category] = load_bank(category, self.dict_split)
        return self._banks[category]

    def value_weight(self, tier: str = "hard") -> float:
        return self.weight_table.get((self.stage, tier), 1.0)


# ── 事件基类 ──

class AtomicEvent(ABC):
    name: str = "?"

    @abstractmethod
    def run(
        self,
        rng: random.Random,
        state: MemoryState,
        ctx: EventContext,
        turn_idx: int,
    ) -> list[ConvPair]:
        """执行事件，返回若干 user/assistant pair。

        turn_idx 是事件首轮的轮次序号（0-based）；事件可能产出多轮（F/H 通常 1 轮，
        L_partial 也是 1 轮）。返回值长度即事件吃掉的轮数。
        """


# ── 注册表 ──

EVENT_REGISTRY: dict[str, AtomicEvent] = {}


def register_event(name: str) -> Callable[[type[AtomicEvent]], type[AtomicEvent]]:
    def deco(cls: type[AtomicEvent]) -> type[AtomicEvent]:
        inst = cls()
        inst.name = name
        EVENT_REGISTRY[name] = inst
        return cls
    return deco


def get_event(name: str) -> AtomicEvent:
    if name not in EVENT_REGISTRY:
        raise KeyError(f"未注册的事件: {name}（已注册: {list(EVENT_REGISTRY)}）")
    return EVENT_REGISTRY[name]


# ── 工具：value_span 抽取 + 权重分配 ──

def find_value_spans(content: str, values: list[str]) -> list[list[int]]:
    """对 values 中每个 surface，在 content 中找首次出现位置 → char span。

    多个 value 时按出现顺序排列；找不到的 value 抛 ValueError（生成器有 bug）。
    """
    spans = []
    for v in values:
        if not v:
            raise ValueError("value surface 为空")
        idx = content.find(v)
        if idx < 0:
            raise ValueError(f"value {v!r} 不在 content 中: {content!r}")
        spans.append([idx, idx + len(v)])
    return spans


def make_assistant_turn(
    content: str,
    *,
    train_loss: str = "true",
    values: Optional[list[str]] = None,
    tier: Optional[str] = None,
    base_weight: float = 0.0,
) -> dict:
    """构造一个 assistant turn dict（已带 schema 字段）。

    若 values 非空，会:
      - 找 char span
      - weight_per_span = base_weight / len(values)（per-sample 守恒）
    """
    if values:
        spans = find_value_spans(content, values)
        if not tier:
            tier = "hard"
        weight = base_weight / max(1, len(values))
        return {
            "role": "assistant",
            "content": content,
            "train_loss": train_loss,
            "value": list(values),
            "value_span": spans,
            "value_tier": tier,
            "weight_per_span": weight,
        }
    return {
        "role": "assistant",
        "content": content,
        "train_loss": train_loss,
        "value": None,
        "value_span": [],
        "value_tier": None,
        "weight_per_span": 0.0,
    }


def make_user_turn(content: str) -> dict:
    return {"role": "user", "content": content}
