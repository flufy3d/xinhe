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


# ── third-party subject 名字池（可独立采样，不复用 given_names）──
THIRD_PARTY_NAMES = [
    "小林", "老李", "小张", "老王", "阿强", "小美",
    "邻居", "老同学", "前同事", "客户", "房东",
    "Alex", "Lily", "Tom", "Eric", "Sara",
]


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


def sample_third_party_subject(rng: random.Random, *, exclude: Optional[list[str]] = None) -> str:
    pool = THIRD_PARTY_NAMES if not exclude else [n for n in THIRD_PARTY_NAMES if n not in exclude]
    return rng.choice(pool or THIRD_PARTY_NAMES)


def sample_object_subject(rng: random.Random) -> str:
    """物件 subject：项目代号风格的标识符。"""
    nums = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return rng.choice(nums) + rng.choice(nums) + "-" + str(rng.randint(10, 99))
