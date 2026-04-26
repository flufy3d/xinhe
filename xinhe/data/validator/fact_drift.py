"""Stage 1 fact drift：召回值是否归一到 plan.canonical 或 alias 表。"""
from __future__ import annotations

from dataclasses import dataclass

from xinhe.data.validator.normalize import fold


@dataclass
class FactDriftResult:
    ok: bool
    drift_facts: list[dict]   # [{"recalled": str, "canonical": str, "reason": str}, ...]


def check_fact_drift(
    recalled_values: list[str],
    canonical_facts: dict[str, list[str]],  # {canonical: [aliases]}
) -> FactDriftResult:
    """每个召回值必须能映射回 canonical（直接或经 alias 折叠）。

    canonical_facts: 由 BeatPlanner 生成时记录，validator 拿来反查。
    """
    drift = []
    for recalled in recalled_values:
        if not recalled:
            continue
        norm_recalled = fold(recalled)
        matched = False
        for canon, aliases in canonical_facts.items():
            if recalled == canon or fold(canon) == norm_recalled:
                matched = True
                break
            for a in aliases:
                if recalled == a or fold(a) == norm_recalled:
                    matched = True
                    break
            if matched:
                break
        if not matched:
            drift.append({"recalled": recalled, "canonical": list(canonical_facts.keys())})
    if drift:
        return FactDriftResult(ok=False, drift_facts=drift)
    return FactDriftResult(ok=True, drift_facts=[])
