"""
Persona — 统一对话数据生成的"人物"数据结构。

每个 episode = 一个 Persona × 12-20 turn 自然对话，
对话推进过程中按"reveal_order"披露 4-6 个槽，其余槽永远不披露，
天然制造"问了但未披露 → 应该拒答"的机会。

设计哲学：persona 是生成时一次性采全（ground truth），
"披露"是对话层面的渐进过程，`revealed` 集合在 turn 状态机里增长。
"""
from dataclasses import dataclass, field
from typing import Optional
import random

from xinhe.data.generate_memory_data import (
    random_name, random_city, random_food, random_job,
    random_hobby, random_age, random_pet, random_number,
)


# 所有槽的名字（8 槽）
SLOT_NAMES = ["name", "age", "city", "food", "job", "hobby", "pet", "number"]


@dataclass
class Persona:
    """单个 persona 的全部槽位 ground truth + 披露状态。

    生成时：所有槽一次性采满值（即使当前 episode 不披露）。
    对话时：按 reveal_order 顺序披露 4-6 个槽，其余槽仅用作 refusal 命中源。
    """
    name: str = ""
    age: str = ""                # 以字符串存储（和 random_age 口径一致）
    city: str = ""
    food: str = ""
    job: str = ""
    hobby: str = ""
    pet: str = ""
    number: str = ""

    # 披露控制
    reveal_order: list = field(default_factory=list)   # 这个 episode 决定披露哪几个槽，按什么顺序
    revealed: set = field(default_factory=set)         # 已经在对话中说出的槽（运行时增长）

    # 第三方人物（可选，用于 third_party turn kind）
    third_party: dict = field(default_factory=dict)    # {"我朋友小明": {"job": "...", "age": "..."}}

    def slot_value(self, slot: str) -> str:
        return getattr(self, slot)

    def unrevealed_slots(self) -> list:
        """reveal_order 里还没披露的槽（refusal 候选来自全部 SLOT_NAMES 减 revealed）"""
        return [s for s in self.reveal_order if s not in self.revealed]

    def refusal_candidates(self) -> list:
        """问了但肯定没披露过的槽 —— 包括不在 reveal_order 里的（永不披露）+ 还没披露的"""
        return [s for s in SLOT_NAMES if s not in self.revealed]


def sample_persona(rng: random.Random, num_reveal: Optional[int] = None) -> Persona:
    """采样一个随机 persona。

    - 所有 8 槽全部采满（ground truth 完整）
    - reveal_order: 从 8 槽里随机选 4-6 个作为"本 episode 要披露"的槽，打乱顺序
    - 其余槽留作 refusal 命中源（用户问这些槽 → 模型应拒答）
    """
    p = Persona(
        name=random_name(rng),
        age=random_age(rng),
        city=random_city(rng),
        food=random_food(rng),
        job=random_job(rng),
        hobby=random_hobby(rng),
        pet=random_pet(rng),
        number=random_number(rng),
    )
    if num_reveal is None:
        num_reveal = rng.randint(4, 6)
    num_reveal = min(num_reveal, len(SLOT_NAMES))
    p.reveal_order = rng.sample(SLOT_NAMES, k=num_reveal)

    # 25% 概率带一个第三方人物（朋友/同事/家人）
    if rng.random() < 0.25:
        relation = rng.choice(["我朋友", "我同事", "我表弟", "我室友", "我老板"])
        tp_name = random_name(rng)
        tp_key = f"{relation}{tp_name}"
        # 第三方 1-2 个属性
        tp_slots = rng.sample(["job", "age", "city", "hobby"], k=rng.randint(1, 2))
        p.third_party[tp_key] = {s: _sample_slot(rng, s) for s in tp_slots}

    return p


def _sample_slot(rng: random.Random, slot: str) -> str:
    """采样单个槽的值（用于第三方 persona）"""
    samplers = {
        "name": random_name, "age": random_age, "city": random_city,
        "food": random_food, "job": random_job, "hobby": random_hobby,
        "pet": random_pet, "number": random_number,
    }
    return samplers[slot](rng)
