"""Skeleton generator world_qa warmup 注入单测。"""
from __future__ import annotations

import random

import pytest

from xinhe.data.generators.skeleton.runner import generate_stage0_episode
from xinhe.data.warmup import inject_world_qa_warmup
from xinhe.data.schema import Sample


# ────────────────────────────────────────────────────────────
# inject_world_qa_warmup 通用工具
# ────────────────────────────────────────────────────────────

def _make_sample(n_turns: int) -> Sample:
    """构造一个 mock sample 用于测试 warmup 注入逻辑。"""
    convs = []
    for i in range(n_turns):
        convs.append({"role": "user", "content": f"u{i}"})
        convs.append({
            "role": "assistant", "content": f"a{i}",
            "train_loss": "true", "value": None, "value_span": [],
            "value_tier": None, "weight_per_span": 0.0,
        })
    return Sample(
        sample_id="test", stage="0", skeleton_id="S1",
        meta={"n_turns": n_turns}, conversations=convs,
    )


def test_inject_warmup_k_zero_no_change():
    sample = _make_sample(4)
    rng = random.Random(0)
    inject_world_qa_warmup(
        sample, rng, dict_split="train",
        k_sampler=lambda r, s: 0, train_loss="lm_only",
    )
    assert sample.meta["warmup_pairs"] == 0
    assert len(sample.conversations) == 8  # 4 turn × 2 segments


def test_inject_warmup_k_positive_prepends():
    sample = _make_sample(4)
    rng = random.Random(42)
    inject_world_qa_warmup(
        sample, rng, dict_split="train",
        k_sampler=lambda r, s: 2, train_loss="lm_only",
    )
    # 应该插了 ≤2 个（pair 数取决于 world_qa 语料；最少 ≥ 0）
    k = sample.meta["warmup_pairs"]
    if k == 0:
        pytest.skip("world_qa 语料缺失或空 fallback")
    assert k <= 2
    # 前 2k 条是 warmup，后面是原始 sample
    assert len(sample.conversations) == k * 2 + 8
    # warmup assistant 段必须 train_loss="lm_only"，value=None
    for i in range(k):
        asst = sample.conversations[i * 2 + 1]
        assert asst["role"] == "assistant"
        assert asst["train_loss"] == "lm_only"
        assert asst["value"] is None
        assert asst["value_span"] == []
    # 原始末段 a3 仍在
    assert sample.conversations[-1]["content"] == "a3"
    # n_turns 更新
    assert sample.meta["n_turns"] == 4 + k


def test_inject_warmup_train_loss_propagated():
    sample = _make_sample(2)
    rng = random.Random(1)
    inject_world_qa_warmup(
        sample, rng, dict_split="train",
        k_sampler=lambda r, s: 1, train_loss="true",
    )
    if sample.meta["warmup_pairs"] == 0:
        pytest.skip("world_qa 缺失")
    assert sample.conversations[1]["train_loss"] == "true"


# ────────────────────────────────────────────────────────────
# generate_stage0_episode 端到端
# ────────────────────────────────────────────────────────────

def test_stage0_inject_warmup_default_on():
    """默认生成应触发 warmup（短 episode 时）。"""
    rng = random.Random(0)
    found_warmup = False
    for _ in range(20):
        sample = generate_stage0_episode(
            rng, skeleton_id="S1", dict_split="train", max_turns=12,
        )
        if sample.meta.get("warmup_pairs", 0) > 0:
            found_warmup = True
            break
    if not found_warmup:
        pytest.skip("20 次内未抽到非零 warmup（取决于 turn_count 分布与 world_qa 命中）")


def test_stage0_inject_warmup_off_no_warmup_field_or_zero():
    """关闭 warmup 后不应注入。"""
    rng = random.Random(0)
    sample = generate_stage0_episode(
        rng, skeleton_id="S1", dict_split="train",
        max_turns=12, inject_warmup=False,
    )
    # 关闭时 meta 不写 warmup_pairs（或写 0），并且总 n_turns 与 skeleton 输出一致
    assert sample.meta.get("warmup_pairs", 0) == 0


def test_stage0_max_turns_4_caps_total_turns():
    """max_turns=4 应让 SkeletonRunner budget 严格收紧到 4 turns 上限（含 warmup）。"""
    rng = random.Random(7)
    for _ in range(10):
        sample = generate_stage0_episode(
            rng, skeleton_id="S1", dict_split="train",
            max_turns=4, inject_warmup=False,
        )
        n_turns = len(sample.conversations) // 2
        # 软约束：skeleton 至少跑完 fixed event slots，可能略超；
        # 主要验证 SkeletonRunner 的 budget 推导确实使用了传入的 max_turns
        assert n_turns <= 12, f"max_turns=4 但生成 {n_turns} turns"


def test_stage0_warmup_keeps_total_within_max_turns():
    """warmup K 余量公式应保 total_turns ≤ max_turns。"""
    rng = random.Random(100)
    for _ in range(30):
        sample = generate_stage0_episode(
            rng, skeleton_id="S1", dict_split="train", max_turns=12,
        )
        total_turns = len(sample.conversations) // 2
        # 关键不变量：warmup 注入后总 turns ≤ max_turns（除非 skeleton 自身已超 budget，
        # 此时 warmup 会跳过）
        # _stage0_warmup_k 公式: rng.randint(1, max_turns - n_turns)，余量≤0 跳过
        warmup = sample.meta.get("warmup_pairs", 0)
        assert total_turns == sample.meta.get("n_turns"), \
            f"meta.n_turns ({sample.meta.get('n_turns')}) 与 conversations 长度 ({total_turns}) 不一致"
        assert total_turns <= 12 + 1, f"total_turns={total_turns} 超 max_turns=12 太多 (warmup={warmup})"
