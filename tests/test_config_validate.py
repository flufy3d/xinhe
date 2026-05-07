"""validate_stage_config 单测：每条规则正反测，错误信息含 Hint:。"""
from __future__ import annotations

import logging

import pytest

from xinhe.config import validate_stage_config
from xinhe.config.errors import ConfigError


def _stage(kind: str, training: dict | None = None, data: dict | None = None) -> dict:
    """构造一个 stage_cfg 骨架。"""
    return {
        "name": "test_stage",
        "data": {"kind": kind, **(data or {})},
        "training": training or {},
    }


# ────────────────────────────────────────────────────────────
# 派生 + happy path
# ────────────────────────────────────────────────────────────

def test_derived_tbptt_turns_default():
    """tbptt_turns 缺省时派生 = max_turns_per_episode。"""
    cfg = _stage("skeleton", training={"max_turns_per_episode": 12, "turn_max_tokens": 256})
    out = validate_stage_config("s0", cfg)
    assert out["training"]["max_turns_per_episode"] == 12
    assert out["training"]["tbptt_turns"] == 12


def test_explicit_tbptt_turns_equal_to_max_turns_no_warning(caplog):
    cfg = _stage("skeleton", training={
        "max_turns_per_episode": 12, "turn_max_tokens": 256, "tbptt_turns": 12,
    })
    with caplog.at_level(logging.WARNING):
        validate_stage_config("s0", cfg)
    assert not any("detach boundary" in r.message for r in caplog.records)


# ────────────────────────────────────────────────────────────
# 规则 1: max_turns_per_episode 缺失 / 非法
# ────────────────────────────────────────────────────────────

def test_missing_max_turns_per_episode():
    cfg = _stage("skeleton", training={"turn_max_tokens": 256})
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "'max_turns_per_episode' missing" in str(exc.value)
    assert "Hint:" in str(exc.value)


def test_max_turns_must_be_positive_int():
    cfg = _stage("skeleton", training={"max_turns_per_episode": -1, "turn_max_tokens": 256})
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "Hint:" in str(exc.value)


# ────────────────────────────────────────────────────────────
# 规则 2: turn_max_tokens 缺失 / 范围
# ────────────────────────────────────────────────────────────

def test_missing_turn_max_tokens():
    cfg = _stage("skeleton", training={"max_turns_per_episode": 12})
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "'turn_max_tokens' missing" in str(exc.value)
    assert "Hint:" in str(exc.value)


def test_turn_max_tokens_out_of_range():
    cfg = _stage("skeleton", training={"max_turns_per_episode": 12, "turn_max_tokens": 8192})
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "outside" in str(exc.value)
    assert "Hint:" in str(exc.value)


# ────────────────────────────────────────────────────────────
# 规则 3: tbptt_turns > max_turns_per_episode 抛错
# ────────────────────────────────────────────────────────────

def test_tbptt_turns_greater_than_max_turns():
    cfg = _stage("skeleton", training={
        "max_turns_per_episode": 12, "turn_max_tokens": 256, "tbptt_turns": 32,
    })
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "tbptt_turns=32 > max_turns_per_episode=12" in str(exc.value)
    assert "Hint:" in str(exc.value)


# ────────────────────────────────────────────────────────────
# 规则 4: tbptt_turns < max_turns_per_episode 在 v9 Titans 架构下不再 warn
#   (Hippo 是 per-token inner SGD,gate/alpha 是局部决策,长程 BPTT 本就不需要)
#   只在 tbptt_turns < 2 时才提示
# ────────────────────────────────────────────────────────────

def test_tbptt_turns_less_than_max_turns_no_warn(caplog):
    cfg = _stage("skeleton", training={
        "max_turns_per_episode": 12, "turn_max_tokens": 256, "tbptt_turns": 6,
    })
    with caplog.at_level(logging.WARNING):
        validate_stage_config("s0", cfg)
    # tbptt < max_turns 在 Titans 架构下是常规配置,不应该有 warn
    msgs = " ".join(r.message for r in caplog.records)
    assert "tbptt" not in msgs.lower() or "tbptt_turns=6" not in msgs


def test_tbptt_turns_less_than_2_warns(caplog):
    cfg = _stage("skeleton", training={
        "max_turns_per_episode": 12, "turn_max_tokens": 256, "tbptt_turns": 1,
    })
    with caplog.at_level(logging.WARNING):
        validate_stage_config("s0", cfg)
    msgs = " ".join(r.message for r in caplog.records)
    assert "tbptt_turns=1" in msgs
    assert "Hint:" in msgs


# ────────────────────────────────────────────────────────────
# 规则 5: dialog n_turns_range[1] > max_turns
# ────────────────────────────────────────────────────────────

def test_dialog_n_turns_range_exceeds_max_turns():
    cfg = _stage(
        "dialog",
        training={"max_turns_per_episode": 10, "turn_max_tokens": 512},
        data={"n_turns_range": [8, 14]},
    )
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s1", cfg)
    assert "n_turns_range" in str(exc.value)
    assert "Hint:" in str(exc.value)


def test_dialog_n_turns_range_within_budget_ok():
    cfg = _stage(
        "dialog",
        training={"max_turns_per_episode": 14, "turn_max_tokens": 512},
        data={"n_turns_range": [10, 14]},
    )
    validate_stage_config("s1", cfg)  # 14 ≤ 14,OK


# NOTE: 旧 plan 曾加过 "beat3_min_chars × 1.5 > turn_max_tokens 抛错" 规则,
# 已移除 —— beat3_min_chars 是"多 turn 累计"门槛,不是单 turn 字数。
# 单 turn 长度由 ConversationDataset._stats.turn_truncation_rate 在加载期监测。

# ────────────────────────────────────────────────────────────
# 规则 6: dialog beat3 长干扰不应阻止 turn_max_tokens=256
# ────────────────────────────────────────────────────────────

def test_dialog_beat3_min_chars_does_not_block_short_turn_max_tokens():
    """beat3 是多 turn flush W,单 turn 长度无关 beat3_min_chars 门槛。"""
    cfg = _stage(
        "dialog",
        training={"max_turns_per_episode": 16, "turn_max_tokens": 256},
        data={"n_turns_range": [10, 14], "beat3_min_chars": 450},
    )
    validate_stage_config("s1", cfg)  # 不应抛错


# ────────────────────────────────────────────────────────────
# 规则 7: skeleton turn_count_hi > max_turns
# ────────────────────────────────────────────────────────────

def test_skeleton_turn_count_hi_exceeds_max_turns():
    cfg = _stage(
        "skeleton",
        training={"max_turns_per_episode": 12, "turn_max_tokens": 256},
        data={"turn_count_hi": 14},
    )
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "turn_count_hi" in str(exc.value)
    assert "Hint:" in str(exc.value)


# ────────────────────────────────────────────────────────────
# 规则 8: distance_bucket 概率和 != 1.0
# ────────────────────────────────────────────────────────────

def test_distance_bucket_sum_not_1():
    cfg = _stage(
        "skeleton",
        training={"max_turns_per_episode": 12, "turn_max_tokens": 256},
        data={"distance_bucket": {"near": 0.20, "mid": 0.35, "far": 0.30, "very_far": 0.10}},
    )
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    assert "0.95" in str(exc.value) or "0.9500" in str(exc.value)
    assert "Hint:" in str(exc.value)


def test_distance_bucket_valid_sum():
    cfg = _stage(
        "skeleton",
        training={"max_turns_per_episode": 12, "turn_max_tokens": 256},
        data={"distance_bucket": {"near": 0.20, "mid": 0.35, "far": 0.30, "very_far": 0.15}},
    )
    validate_stage_config("s0", cfg)


# ────────────────────────────────────────────────────────────
# 多条错误一次报齐
# ────────────────────────────────────────────────────────────

def test_multiple_errors_aggregated():
    cfg = _stage(
        "skeleton",
        training={},  # 缺 max_turns + turn_max_tokens
        data={"distance_bucket": {"near": 0.5, "mid": 0.4}},  # sum=0.9
    )
    with pytest.raises(ConfigError) as exc:
        validate_stage_config("s0", cfg)
    msg = str(exc.value)
    # 必须含 3 条错误（max_turns / turn_max_tokens / distance_bucket）
    assert "3 issues" in msg
    assert "max_turns_per_episode" in msg
    assert "turn_max_tokens" in msg
    assert "distance_bucket" in msg
    # 每条都有 Hint
    assert msg.count("Hint:") >= 3


# ────────────────────────────────────────────────────────────
# 派生回填只在通过校验后发生
# ────────────────────────────────────────────────────────────

def test_no_derive_on_invalid_config():
    cfg = _stage("skeleton", training={"max_turns_per_episode": 12})  # 缺 turn_max_tokens
    with pytest.raises(ConfigError):
        validate_stage_config("s0", cfg)
    assert "tbptt_turns" not in cfg["training"]
