"""DeepSeek 响应 → Sample dict（含 schema 字段补全）。

LLM 输出形如:
  {"conversations": [
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "...", "train_loss": "true", "value": ["..."], "beat": 1},
     ...
  ]}

parser 的工作:
  1. 解析 JSON。
  2. 校验 user/assistant 严格交替 + train_loss / value 合法。
  3. 给每个 assistant turn 用 validator.tier 重新分级，得到 value_span / value_tier / weight_per_span。
  4. 收集 beat3 / beat4 turn indices 写回 plan，供 validator 调用。
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from xinhe.data.dicts.bank import dict_version
from xinhe.data.schema import Sample, normalize_train_loss
from xinhe.data.validator.tier import TierVerdict, classify_tier

if TYPE_CHECKING:
    from xinhe.data.generators.dialog.beat_planner import BeatPlan


class ParseError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    """容错抠 JSON：去 markdown 包裹 + 去前后噪声。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    # 找第一个 { 到最后一个 }
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e < 0:
        raise ParseError(f"找不到 JSON 边界: {text[:80]}")
    return json.loads(text[s:e + 1])


def _normalize_alternation(convs_raw: list[dict]) -> list[dict]:
    """LLM 输出常见两种瑕疵的容错修复：
       1. 末尾孤立 user 轮 → 丢
       2. 连续两个相同 role → 合并 content（保留前者的 schema 字段；后者 content 拼到前者末尾）
       3. 首轮非 user → 丢前置噪声直到首个 user
    """
    cleaned: list[dict] = []
    for item in convs_raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        # 合并连续同 role
        if cleaned and cleaned[-1].get("role") == role:
            prev = cleaned[-1]
            prev["content"] = (prev.get("content", "") + "\n" + content).strip()
            # 后者带 value/beat 等字段时，覆盖（取最近一个 LLM 标注）
            for k in ("train_loss", "value", "beat"):
                if k in item:
                    prev[k] = item[k]
            continue
        cleaned.append(dict(item))
    # 跳过前置非 user
    while cleaned and cleaned[0].get("role") != "user":
        cleaned.pop(0)
    # 末尾若是 user（孤立）丢
    if cleaned and cleaned[-1].get("role") == "user":
        cleaned.pop()
    return cleaned


def parse_response(
    raw_content: str,
    plan: "BeatPlan",
    *,
    sample_id: str | None = None,
    weight_table: dict | None = None,
    generator_model: str | None = None,
) -> Sample:
    obj = _extract_json(raw_content)
    convs_raw = obj.get("conversations") or obj.get("turns") or []
    if not isinstance(convs_raw, list):
        raise ParseError(f"conversations 不是 list: {type(convs_raw).__name__}")

    convs_raw = _normalize_alternation(convs_raw)
    if len(convs_raw) < 2:
        raise ParseError(f"normalize 后 conversations 过短: {len(convs_raw)} 条")

    weight_table = weight_table or {("1", "hard"): 3.0, ("1", "soft"): 1.5}

    # 严格交替规整 + 字段补全
    conversations: list[dict] = []
    beat3_indices: list[int] = []
    beat4_indices: list[int] = []

    canon_values: list[str] = [f.canonical_value for f in plan.canonical_facts]
    canon_aliases: dict[str, list[str]] = {
        f.canonical_value: list(f.aliases) for f in plan.canonical_facts
    }

    for i, item in enumerate(convs_raw):
        role = item.get("role")
        content = item.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ParseError(f"第 {i} 项 content 缺失")
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            raise ParseError(f"第 {i} 项 role={role!r} 期望 {expected!r}")

        if role == "user":
            conversations.append({"role": "user", "content": content})
            continue

        # assistant：补全 schema 字段
        try:
            train_loss = normalize_train_loss(item.get("train_loss", "true"))
        except (ValueError, TypeError) as e:
            raise ParseError(f"第 {i} 项 train_loss 非法: {e}") from e
        beat = item.get("beat")
        values_raw = item.get("value")

        if beat == 3:
            beat3_indices.append(i)
            # Beat 3 强制 lm_only（即便 LLM 给 "true" 也覆盖）
            train_loss = "lm_only"
            # Beat 3 不应有 value
            values_raw = None
        elif beat == 4:
            beat4_indices.append(i)

        # 整理 value list
        if isinstance(values_raw, str):
            values = [values_raw]
        elif isinstance(values_raw, list):
            values = [str(v) for v in values_raw if v]
        else:
            values = None

        # 跑 tier 分级
        spans: list[list[int]] = []
        actual_values: list[str] = []
        tier_decided: str | None = None
        weight = 0.0

        if values:
            tier_results = []
            for v in values:
                # 找匹配的 canonical（如果 v 本身就是 canonical 直接用，否则查 alias）
                canon_for_v = v
                if v not in canon_values:
                    for c, alist in canon_aliases.items():
                        if v in alist:
                            canon_for_v = c
                            break
                rel_name = next(
                    (f.relation for f in plan.canonical_facts if f.canonical_value == canon_for_v),
                    None,
                )
                soft_ok = next(
                    (f.soft_eligible for f in plan.canonical_facts if f.canonical_value == canon_for_v),
                    True,
                )
                res = classify_tier(
                    v, content, relation=rel_name, relation_soft_eligible=soft_ok,
                )
                tier_results.append((v, res))

            # 任一 reject → 整 turn reject（不能信）
            if any(r[1].verdict == TierVerdict.REJECT for r in tier_results):
                raise ParseError(
                    f"第 {i} 项 value 中存在 reject: {[(v, r.reason) for v, r in tier_results]}"
                )

            # 全 hard → hard；含 soft → 整体 soft
            if any(r[1].verdict == TierVerdict.SOFT for r in tier_results):
                tier_decided = "soft"
            else:
                tier_decided = "hard"

            base_weight = weight_table.get(("1", tier_decided), 1.0)
            weight = base_weight / max(1, len(tier_results))

            for v, r in tier_results:
                if r.span is None:
                    raise ParseError(f"value {v!r} 找不到 span")
                spans.append([r.span[0], r.span[1]])
                actual_values.append(r.matched_surface or v)

        a_turn = {
            "role": "assistant",
            "content": content,
            "train_loss": train_loss,
            "value": actual_values if values else None,
            "value_span": spans,
            "value_tier": tier_decided,
            "weight_per_span": weight,
        }
        conversations.append(a_turn)

    # 写回 plan 的 indices
    plan.beat3_turn_indices = beat3_indices
    plan.beat4_turn_indices = beat4_indices

    n_turns = len(conversations) // 2
    meta = {
        "n_turns": n_turns,
        "n_canonical": len(plan.canonical_facts),
        "recall_form": plan.recall_form,
        "dict_version": dict_version(),
        "substream": "1A",
        "generator_model": generator_model or "unknown",
        "canonical_facts": [
            {
                "subject": f.subject,
                "relation": f.relation,
                "scope": f.scope,
                "canonical_value": f.canonical_value,
                "aliases": f.aliases,
            }
            for f in plan.canonical_facts
        ],
    }
    return Sample(
        sample_id=sample_id or uuid.uuid4().hex[:12],
        stage="1",
        skeleton_id=None,
        meta=meta,
        conversations=conversations,
    )
