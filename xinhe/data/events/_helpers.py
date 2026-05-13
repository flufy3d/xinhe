"""
事件实现共用的工具函数：

- relation_by_name：从 RELATIONS_BY_NAME 取 RelationSpec
- pick_template：按 ctx 提示选模板（pending_relation 支持骨架级强约束）
- get_or_seed_subject：第三方 / 物件 subject 维护
- build_user_key / build_third_party_key：构造 MemoryState key
"""
from __future__ import annotations

import random
from typing import Optional

from xinhe.data.events._relations import (
    RELATIONS,
    RelationSpec,
    sample_relation,
    sample_third_party_subject,
    sample_object_subject,
)
from xinhe.data.events.base import EventContext
from xinhe.data.memory_state import Key
from xinhe.data.templates.base import Template, TemplatePool


RELATIONS_BY_NAME: dict[str, RelationSpec] = {r.name: r for r in RELATIONS}


def relation_by_name(name: str) -> RelationSpec:
    if name not in RELATIONS_BY_NAME:
        raise KeyError(f"未知 relation: {name!r}")
    return RELATIONS_BY_NAME[name]


def pick_template(
    pool: TemplatePool,
    rng: random.Random,
    ctx: EventContext,
    *,
    relation: Optional[str] = None,
) -> Template:
    """从模板池选模板。优先级:
      1. relation 参数(显式传入)
      2. ctx.canonical_pool["__pending_relation"](一次性消费,pop)
      3. ctx.canonical_pool["__force_relation"](skeleton 级常驻,不 pop)
      4. 从 pool 全集随机

    若 relation 显式指定但 pool 内无匹配模板 → 抛 KeyError。
    (旧版本静默 fallback 全池随机,造成 S5 self_name 拿 job 模板的语义错位 bug,
     直接训进 jsonl 数据看不出来。改为 loud 失败逼调用方补模板或不传 relation。)
    """
    if relation is None:
        relation = ctx.canonical_pool.pop("__pending_relation", None)
    if relation is None:
        relation = ctx.canonical_pool.get("__force_relation")
    pool_list = pool.templates
    if not pool_list:
        raise ValueError(f"模板池 {pool.event_name} 为空")
    if relation:
        cands = [t for t in pool_list if t.meta.get("relation") == relation]
        if cands:
            return rng.choice(cands)
        raise KeyError(
            f"模板池 {pool.event_name!r} 内无 relation={relation!r} 的模板;"
            f" 现有 relations={sorted(set(t.meta.get('relation') for t in pool_list))}。"
            f" Hint: 给 {pool.event_name} 池补 {relation} 模板,或检查上游 skeleton/event 不要硬塞此 relation。"
        )
    return rng.choice(pool_list)


def get_or_seed_third_party(ctx: EventContext, rng: random.Random) -> str:
    """同一样本内第三方 subject 复用：第一次 I 事件创建后，G/H/L_partial 复用同一个名字。"""
    if "__third_party_subject" not in ctx.canonical_pool:
        existing = [
            v for k, v in ctx.canonical_pool.items()
            if isinstance(k, str) and k.startswith("__")
        ]
        ctx.canonical_pool["__third_party_subject"] = sample_third_party_subject(
            rng,
            dict_split=ctx.dict_split,
            exclude=[v for v in existing if isinstance(v, str)],
        )
    return ctx.canonical_pool["__third_party_subject"]


def get_or_seed_object_subject(ctx: EventContext, rng: random.Random) -> str:
    if "__object_subject" not in ctx.canonical_pool:
        ctx.canonical_pool["__object_subject"] = sample_object_subject(rng)
    return ctx.canonical_pool["__object_subject"]


def make_key(rel: RelationSpec, ctx: EventContext, rng: random.Random) -> Key:
    """根据 RelationSpec.scope 返回标准 Key。"""
    if rel.scope == "self":
        return ("user", rel.name, "self")
    if rel.scope == "third_party":
        subj = get_or_seed_third_party(ctx, rng)
        return (subj, rel.name, "third_party")
    if rel.scope == "object":
        subj = get_or_seed_object_subject(ctx, rng)
        return (subj, rel.name, "object")
    raise ValueError(f"未知 scope: {rel.scope}")


def render_subject_phrase(rel: RelationSpec, key: Key) -> str:
    """模板中 {subject_phrase} 占位的中文短语：

    - self      → ""（user 用 "我"，已在模板里硬编码）
    - third     → key[0]（如 "小林"）
    - object    → key[0]（如 "AB-12 项目"）
    """
    if rel.scope == "self":
        return "我"
    if rel.scope == "third_party":
        return key[0]
    return f"{key[0]} 这个项目"
