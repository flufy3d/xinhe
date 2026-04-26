"""验证器主入口：validate(sample, stage, plan=None) → ValidationResult

执行顺序（任何一步 fail 即整条 reject）:
  1. schema.validate_sample（结构 + 四元一致性 + span 越界）
  2. memory_state 重放（仅 Stage 0；Stage 1 由 fact_drift 替代）
  3. tier 分配验证：每个 value 在 content 中能定位为 hard 或 soft
  4. echo defense：assistant 不得整段复读上一轮 user
  5. Beat 3 结构（仅 Stage 1）：pair 数 ≥ min、字符数 ≥ min、Beat 4 存在
  6. Beat 3 纯洁性（仅 Stage 1）
  7. fact_drift（仅 Stage 1）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from xinhe.data.schema import SchemaError, validate_sample
from xinhe.data.validator.beat3_purity import check_beat3_purity, PurityResult
from xinhe.data.validator.beat3_repetition import check_beat3_repetition, RepetitionResult
from xinhe.data.validator.beat3_structure import check_beat3_structure, Beat3StructureResult
from xinhe.data.validator.beat4_pronoun import check_beat4_pronoun, PronounResult
from xinhe.data.validator.beat4_scope import check_beat4_scope, ScopeResult
from xinhe.data.validator.canonical_order import check_canonical_order, CanonicalOrderResult
from xinhe.data.validator.echo_check import check_echo, EchoResult
from xinhe.data.validator.fact_drift import check_fact_drift, FactDriftResult
from xinhe.data.validator.tier import TierVerdict, classify_tier


@dataclass
class ValidationResult:
    ok: bool
    sample: dict
    errors: list[str] = field(default_factory=list)
    purity: Optional[PurityResult] = None
    drift: Optional[FactDriftResult] = None
    echo: Optional[EchoResult] = None
    beat3_struct: Optional[Beat3StructureResult] = None
    pronoun: Optional[PronounResult] = None
    scope: Optional[ScopeResult] = None
    repetition: Optional[RepetitionResult] = None
    canonical_order: Optional[CanonicalOrderResult] = None


def _validate_tier_consistency(sample: dict) -> list[str]:
    """对每个带 value 的 assistant turn，重新分级，与样本声明的 tier 比对。"""
    errs: list[str] = []
    for i, turn in enumerate(sample.get("conversations", [])):
        if turn.get("role") != "assistant":
            continue
        values = turn.get("value")
        if not values:
            continue
        content = turn.get("content", "")
        declared_tier = turn.get("value_tier")
        for v in values:
            res = classify_tier(v, content)
            if res.verdict == TierVerdict.REJECT:
                errs.append(f"turn[{i}] value {v!r} 在 content 中无法定位（既非 hard 也非 soft）")
            elif declared_tier == "hard" and res.verdict == TierVerdict.SOFT:
                errs.append(f"turn[{i}] 声明 hard 但仅 soft 命中: {v!r}")
            # 反过来 declared soft 实际 hard 是允许的（hard 可降级为 soft 但不应该）
    return errs


def validate(
    sample: dict,
    *,
    stage: str = "0",
    plan: Optional[dict] = None,   # Stage 1 的 BeatPlanner.plan() 输出
) -> ValidationResult:
    errors: list[str] = []

    # 1. schema
    try:
        validate_sample(sample)
    except SchemaError as e:
        return ValidationResult(ok=False, sample=sample, errors=[f"schema: {e}"])

    # 2. tier consistency
    errors.extend(_validate_tier_consistency(sample))

    # 3. echo defense（两阶段都跑；Stage 0 模板基本不会触发，Stage 1 是主战场）
    echo = check_echo(sample.get("conversations", []))
    if not echo.ok:
        errors.append(f"echo: {echo.reason}")

    # 4. Stage 1 specific
    purity: Optional[PurityResult] = None
    drift: Optional[FactDriftResult] = None
    beat3_struct: Optional[Beat3StructureResult] = None
    pronoun: Optional[PronounResult] = None
    scope: Optional[ScopeResult] = None
    repetition: Optional[RepetitionResult] = None
    canonical_order: Optional[CanonicalOrderResult] = None
    if stage == "1" and plan:
        # user_injection: 每个 canonical_value(或 alias) 必须由某个 user turn 逐字说出
        # 防止 ling 把 user 暗示翻译成 canonical 让 assistant 凭空编造
        canonical_facts = plan.get("canonical_facts") or {}
        if canonical_facts:
            all_user_text = " ".join(
                t.get("content", "") for t in sample.get("conversations", [])
                if t.get("role") == "user"
            )
            missing = []
            for canonical, aliases in canonical_facts.items():
                forms = [canonical] + list(aliases or [])
                if not any(form and form in all_user_text for form in forms):
                    missing.append(canonical)
            if missing:
                errors.append(f"user_injection: canonical 未在任何 user turn 出现 {missing}")
        # Beat 3 结构（pair 数 / 字符数 / Beat 4 存在性）
        beat3_idxs = plan.get("beat3_turn_indices") or []
        beat4_idxs = plan.get("beat4_turn_indices") or []
        # 只在 5-Beat 1A（plan 含 canonical_facts）上跑结构校验
        if plan.get("canonical_facts"):
            beat3_struct = check_beat3_structure(
                sample.get("conversations", []),
                beat3_idxs,
                beat4_idxs,
                min_pairs=int(plan.get("beat3_min_turns", 4)),
                min_chars=int(plan.get("beat3_min_chars", 1500)),
            )
            if not beat3_struct.ok:
                errors.append(f"beat3 struct: {beat3_struct.reason}")

        # Beat 3 重复检测（LLM degeneracy）
        beat3_turn_indices = plan.get("beat3_turn_indices") or []
        if beat3_turn_indices:
            repetition = check_beat3_repetition(
                sample.get("conversations", []),
                beat3_turn_indices,
            )
            if not repetition.ok:
                errors.append(f"beat3 repetition: {repetition.reason}")

        # Beat 3 纯洁性
        beat3_turn_indices = plan.get("beat3_turn_indices") or []
        beat3_texts = [
            sample["conversations"][i]["content"]
            for i in beat3_turn_indices
            if 0 <= i < len(sample["conversations"])
        ]
        banned = list(plan.get("banned_terms") or [])
        if beat3_texts and banned:
            purity = check_beat3_purity(beat3_texts, banned)
            if not purity.ok:
                errors.append(f"beat3 purity: {purity.reason} 泄漏={purity.leaked_terms}")

        # fact drift（Beat 4 召回 value 必须映射 canonical）
        beat4_turn_indices = plan.get("beat4_turn_indices") or []
        recalled = []
        for i in beat4_turn_indices:
            if 0 <= i < len(sample["conversations"]):
                turn = sample["conversations"][i]
                if turn.get("role") == "assistant" and turn.get("value"):
                    recalled.extend(turn["value"])
        canonical_facts = plan.get("canonical_facts") or {}
        if recalled and canonical_facts:
            drift = check_fact_drift(recalled, canonical_facts)
            if not drift.ok:
                errors.append(f"fact drift: {drift.drift_facts}")

        # Beat 4 人称反转（self-scope fact 召回时 user 不准用"你"指 assistant）
        facts_scope = plan.get("facts_scope") or {}
        if beat4_turn_indices and facts_scope:
            pronoun = check_beat4_pronoun(
                sample.get("conversations", []),
                beat4_turn_indices,
                facts_scope,
            )
            if not pronoun.ok:
                errors.append(f"beat4 pronoun: {pronoun.reason}")

        # Beat 4 scope 错配（self/third_party 召回归属与 user 提问句人称对齐）
        facts_meta = plan.get("facts_meta") or []
        if beat4_turn_indices and facts_meta:
            scope = check_beat4_scope(
                sample.get("conversations", []),
                beat4_turn_indices,
                facts_meta,
            )
            if not scope.ok:
                errors.append(f"beat4 scope: {scope.reason}")

        # Canonical 出现顺序（user 必须先于 assistant 提到 canonical）
        if canonical_facts:
            canonical_order = check_canonical_order(
                sample.get("conversations", []),
                list(canonical_facts.keys()),
            )
            if not canonical_order.ok:
                errors.append(f"canonical_order: {canonical_order.reason}")

    ok = len(errors) == 0
    return ValidationResult(
        ok=ok, sample=sample, errors=errors, purity=purity, drift=drift, echo=echo,
        beat3_struct=beat3_struct, pronoun=pronoun, scope=scope, repetition=repetition,
        canonical_order=canonical_order,
    )
