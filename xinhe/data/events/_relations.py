"""
Relation Spec：事件共享的关系定义。

每个 RelationSpec 描述一类记忆事实（如"喜欢的食物"、"宠物名字"、"项目代号"），
事件通过 spec 决定从哪儿采 value、写什么 key、是否允许 soft tier。

key = (subject, relation, scope)：
  - subject: 用户("user") / 第三方("xiaolin"/"老李") / 物件代号("project_alpha")
  - relation: spec.name
  - scope: spec.scope ("self" / "third_party" / "object")
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RelationSpec:
    name: str           # "fav_food"
    bank: str           # 词典类别
    scope: str          # "self" / "third_party" / "object"
    mode: str           # "scalar" / "set"
    soft_eligible: bool = True   # False 表示该 relation 的 value 禁用 Soft tier
    label: str = ""     # 中文短语，模板里 {relation_word} 用


# ── 常用关系池 ──
# 注意 soft_eligible：
#   编号 / 暗号 / 姓名 / 颜色细粒度 / 拒答事实 一律 hard-only
#   食物 / 城市 / 职业 / 爱好 / 品牌 / 组织 可以 soft（同义改写、简称）

RELATIONS: list[RelationSpec] = [
    # -- 用户自己 (scope=self) --
    RelationSpec("self_name",  "synthetic_full_name", "self", "scalar", False, "名字"),  # 姓名禁 soft
    RelationSpec("fav_food",   "foods",       "self",  "scalar", True,  "喜欢吃的食物"),
    RelationSpec("fav_color",  "colors",      "self",  "scalar", False, "喜欢的颜色"),  # 颜色禁 soft
    RelationSpec("fav_brand",  "brands",      "self",  "scalar", True,  "常用的品牌"),
    RelationSpec("fav_hobby",  "hobbies",     "self",  "set",    True,  "兴趣"),
    RelationSpec("home_city",  "cities",      "self",  "scalar", True,  "所在的城市"),
    RelationSpec("hometown",   "cities",      "self",  "scalar", True,  "老家"),
    RelationSpec("job",        "jobs",        "self",  "scalar", True,  "职业"),
    RelationSpec("pet_kind",   "pets",        "self",  "scalar", True,  "养的宠物种类"),
    RelationSpec("pet_name",   "given_names", "self",  "scalar", False, "宠物的名字"),  # 姓名禁 soft

    # -- 第三方实体 (scope=third_party)，subject 由 sample_third_party 给出 --
    RelationSpec("tp_pet_name",  "given_names", "third_party", "scalar", False, "宠物名字"),
    RelationSpec("tp_pet_kind",  "pets",        "third_party", "scalar", True,  "养的宠物种类"),
    RelationSpec("tp_fav_color", "colors",      "third_party", "scalar", False, "喜欢的颜色"),
    RelationSpec("tp_job",       "jobs",        "third_party", "scalar", True,  "职业"),
    RelationSpec("tp_city",      "cities",      "third_party", "scalar", True,  "所在城市"),

    # -- 物件 (scope=object)，subject 由项目代号或暗号引出 --
    RelationSpec("project_code",  "project_codes",  "object", "scalar", False, "项目代号"),  # 代号禁 soft
    RelationSpec("password",      "passwords",      "object", "scalar", False, "暗号"),
    RelationSpec("org_for_proj",  "organizations",  "object", "scalar", True,  "对接组织"),
]


class SyntheticNameBank:
    """虚拟 bank:sample_one 时合成 surname+given_name(空间 ~万级)。
    EntityBank 的 duck-typed 替身,不读 .txt,不做 entries 非空校验。
    """

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def sample_one(self, rng: random.Random) -> str:
        return sample_third_party_subject(rng, dict_split=self.split)

    def sample(self, rng: random.Random, n: int = 1, *, unique: bool = True) -> list[str]:
        if n == 1:
            return [self.sample_one(rng)]
        seen, out = set(), []
        for _ in range(n * 5):
            x = self.sample_one(rng)
            if unique and x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= n:
                break
        if len(out) < n:
            raise RuntimeError(f"SyntheticNameBank 抽不出 {n} 个 unique")
        return out

    def __len__(self) -> int:
        return 100_000  # 表征空间巨大


def sample_relation(
    rng: random.Random,
    *,
    scope: Optional[str] = None,
    mode: Optional[str] = None,
    soft_eligible: Optional[bool] = None,
    not_in: Optional[list[str]] = None,
) -> RelationSpec:
    """加权随机选一个 RelationSpec，可按 scope/mode/soft_eligible 过滤。"""
    pool = RELATIONS
    if scope is not None:
        pool = [r for r in pool if r.scope == scope]
    if mode is not None:
        pool = [r for r in pool if r.mode == mode]
    if soft_eligible is not None:
        pool = [r for r in pool if r.soft_eligible == soft_eligible]
    if not_in:
        pool = [r for r in pool if r.name not in not_in]
    if not pool:
        raise ValueError(f"找不到符合约束的 RelationSpec: scope={scope} mode={mode} not_in={not_in}")
    return rng.choice(pool)


def sample_third_party_subject(
    rng: random.Random,
    *,
    dict_split: str = "train",
    exclude: Optional[list[str]] = None,
) -> str:
    """第三方 subject 名字:动态合成 surname + given_name。

    SyntheticNameBank.sample_one 的实际实现;stage0 (EventContext.bank) 与 stage1
    (BeatPlanner._bank) 都走同一条路径,人物名分布完全一致。空间
    ~|surnames|×|given_names| ≈ 上万组合,远胜旧 16 个写死列表。

    策略:
      - given_name g 全 ASCII(英文名)→ 直接用
      - g 多字中文(已 ≥2 字)→ 50% 概率拼姓,50% 不拼(保留单/双字别名风格)
      - g 单字中文 → 必拼姓(单字孤立显得突兀)

    exclude 命中时最多 retry 5 次再合成,空间够大不易撞。
    """
    from xinhe.data.dicts.bank import load_bank
    exclude = exclude or []
    for _ in range(5):
        g = load_bank("given_names", dict_split).sample_one(rng)
        if all(c.isascii() for c in g):
            name = g
        elif len(g) >= 2 and rng.random() < 0.5:
            name = g
        else:
            s = load_bank("surnames", dict_split).sample_one(rng)
            name = s + g
        if name not in exclude:
            return name
    return name


def sample_object_subject(rng: random.Random) -> str:
    """物件 subject：项目代号风格的标识符。"""
    nums = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return rng.choice(nums) + rng.choice(nums) + "-" + str(rng.randint(10, 99))
