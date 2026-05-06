"""Stage 配置校验。

每个 curriculum stage 在 train.py / generate_data.py 入口调一次：
- 必填字段缺失 / 类型错 / 字段超约束 → 抛 ConfigError，错误信息含 Hint:
- 多条错误一次报齐，避免改一条跑一次的循环
- 默认 tbptt_turns 缺省时派生 = max_turns_per_episode（无截断）

术语统一（仅一种业务概念：turn = 一个 user-asst pair）：
  max_turns_per_episode  单 episode 最多几个 turn
  turn_max_tokens         单 turn 内 token 上限
  tbptt_turns             每多少个 turn backward 一次（默认 = max_turns_per_episode 即无截断）
"""
from __future__ import annotations

import logging
from typing import Any

from xinhe.config.errors import ConfigError

logger = logging.getLogger(__name__)

# turn_max_tokens 合理区间(防呆,避免 8192 / 64 这种异常值)
_TURN_MAX_TOKENS_MIN = 128
_TURN_MAX_TOKENS_MAX = 4096


def validate_stage_config(stage_name: str, stage_cfg: dict[str, Any]) -> dict[str, Any]:
    """校验单个 stage 配置；通过则原地写入派生字段并返回，否则抛 ConfigError。

    必校验字段（在 stage_cfg["training"] 里）：
      - max_turns_per_episode (int)：单 episode 最多几个 turn
      - turn_max_tokens (int)：单 turn token 上限

    可选字段：
      - tbptt_turns (int)：默认派生 = max_turns_per_episode；显式给小值会 warning

    依据 stage_cfg["data"]["kind"] 触发额外规则:
      - dialog: n_turns_range[1] <= max_turns_per_episode
      - dialog: beat3_min_chars × 1.5 (zh→token 估算) <= turn_max_tokens
    """
    errors: list[str] = []
    warns: list[str] = []

    training = stage_cfg.get("training", {})
    data = stage_cfg.get("data", {})
    kind = data.get("kind", "skeleton")

    # ── 必填: max_turns_per_episode ──
    max_turns = training.get("max_turns_per_episode")
    if max_turns is None:
        errors.append(
            "'max_turns_per_episode' missing under training:. "
            "Hint: add 'max_turns_per_episode: 12' (skeleton) or 16 (dialog/mix)."
        )
    elif not isinstance(max_turns, int) or max_turns < 1:
        errors.append(
            f"max_turns_per_episode={max_turns!r} must be positive int. "
            f"Hint: typical values 8–16."
        )
        max_turns = None

    # ── 必填: turn_max_tokens ──
    turn_max_tokens = training.get("turn_max_tokens")
    if turn_max_tokens is None:
        errors.append(
            "'turn_max_tokens' missing under training: (no fallback in base.yaml). "
            "Hint: add 'turn_max_tokens: 256' (skeleton) or 768 (dialog/mix)."
        )
    elif not isinstance(turn_max_tokens, int):
        errors.append(
            f"turn_max_tokens={turn_max_tokens!r} must be int. "
            f"Hint: use integer like 256, 512, 768."
        )
        turn_max_tokens = None
    elif not (_TURN_MAX_TOKENS_MIN <= turn_max_tokens <= _TURN_MAX_TOKENS_MAX):
        errors.append(
            f"turn_max_tokens={turn_max_tokens} outside [{_TURN_MAX_TOKENS_MIN}, {_TURN_MAX_TOKENS_MAX}]. "
            f"Hint: out of typical range; if intentional override the assertion in xinhe/config/validate.py."
        )

    # ── tbptt_turns 关系（缺省派生 = max_turns_per_episode） ──
    tbptt_turns = training.get("tbptt_turns")
    if tbptt_turns is not None and max_turns is not None:
        if tbptt_turns > max_turns:
            errors.append(
                f"tbptt_turns={tbptt_turns} > max_turns_per_episode={max_turns}. "
                f"Hint: tbptt window cannot exceed episode length; remove tbptt_turns to default to max_turns_per_episode."
            )
        elif tbptt_turns < max_turns:
            warns.append(
                f"tbptt_turns={tbptt_turns} < max_turns_per_episode={max_turns}. "
                f"Will introduce detach boundary at turn {tbptt_turns} — "
                f"long-range Beat 1↔Beat 4 gradients will be cut. "
                f"Hint: set tbptt_turns={max_turns} or remove field for no-truncation default."
            )

    # ── skeleton: turn_count_hi（可选 yaml 字段，future-proof）──
    if kind == "skeleton" and max_turns is not None:
        turn_count_hi = data.get("turn_count_hi")
        if turn_count_hi is not None and turn_count_hi > max_turns:
            errors.append(
                f"skeleton turn_count_hi={turn_count_hi} > max_turns_per_episode={max_turns}. "
                f"Generator would emit episodes longer than dataloader can handle, "
                f"causing silent truncation in conversation.py. "
                f"Hint: lower turn_count_hi to ≤{max_turns} or raise max_turns_per_episode."
            )

    # ── dialog: n_turns_range[1] <= max_turns ──
    if kind == "dialog":
        n_turns_range = data.get("n_turns_range")
        if n_turns_range is not None and max_turns is not None:
            try:
                n_turns_hi = int(n_turns_range[1])
            except (TypeError, IndexError, ValueError):
                errors.append(
                    f"dialog n_turns_range={n_turns_range!r} not parseable as [lo, hi]. "
                    f"Hint: use 'n_turns_range: [10, 14]'."
                )
            else:
                if n_turns_hi > max_turns:
                    errors.append(
                        f"dialog n_turns_range={tuple(n_turns_range)} > max_turns_per_episode={max_turns}. "
                        f"Hint: reduce n_turns_range upper bound to ≤{max_turns} "
                        f"or raise max_turns_per_episode."
                    )

        # NOTE: beat3_min_chars 是"beat3 段(多 turn 累计)的总字数门槛"，
        # 不是单 turn 字数。BeatPlan.beat3_turn_indices 是 list[int],
        # 实测每个子 turn 仍 < 256 token（fact-bearing max=124, free-text max=123 in 200 samples）。
        # 所以不能从 beat3_min_chars 推 turn_max_tokens 下界 —— 实际 turn 长度由
        # ConversationDataset._stats.turn_truncation_rate 在加载期监测,超 5% 自动 warning。

    # ── distance_bucket 概率和 ≈ 1.0 ──
    distance_bucket = data.get("distance_bucket")
    if distance_bucket is not None:
        if not isinstance(distance_bucket, dict):
            errors.append(
                f"distance_bucket must be dict, got {type(distance_bucket).__name__}. "
                f"Hint: 'distance_bucket: {{near: 0.20, mid: 0.35, far: 0.30, very_far: 0.15}}'."
            )
        else:
            try:
                total = sum(float(v) for v in distance_bucket.values())
            except (TypeError, ValueError):
                errors.append(
                    f"distance_bucket has non-numeric values: {distance_bucket!r}. "
                    f"Hint: all weights must be float."
                )
            else:
                if abs(total - 1.0) > 1e-3:
                    errors.append(
                        f"distance_bucket weights sum to {total:.4f}, expected 1.0. "
                        f"Got: {distance_bucket}. "
                        f"Hint: adjust weights so they sum to 1.0 (current sum off by {total - 1.0:+.4f})."
                    )

    # ── 一次性报齐 ──
    if errors:
        if len(errors) == 1:
            msg = f"in stage {stage_name!r}: {errors[0]}"
        else:
            joined = "\n".join(f"  {i + 1}. {e}" for i, e in enumerate(errors))
            msg = f"{len(errors)} issues in stage {stage_name!r}:\n{joined}"
        raise ConfigError(msg)

    # ── warning 一次性输出 ──
    for w in warns:
        logger.warning(f"stage {stage_name!r}: {w}")

    # ── 派生回填：tbptt_turns 缺省即 max_turns_per_episode ──
    if tbptt_turns is None and max_turns is not None:
        training["tbptt_turns"] = max_turns

    return stage_cfg
