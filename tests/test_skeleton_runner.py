"""SkeletonRunner 端到端测试：每个骨架至少 5 条样本通过 validator。"""
import random

import pytest

from xinhe.data.skeletons.library import SKELETONS
from xinhe.data.skeletons.runner import SkeletonRunner
from xinhe.data.validator.api import validate


@pytest.mark.parametrize("skeleton_id", list(SKELETONS.keys()))
def test_skeleton_produces_valid_samples(skeleton_id):
    sk = SKELETONS[skeleton_id]
    runner = SkeletonRunner(dict_split="train", stage="0")
    rng = random.Random(hash(skeleton_id) % (2**31))

    n_ok = 0
    for _ in range(5):
        sample = runner.run(sk, rng)
        d = sample.to_dict()
        result = validate(d, stage="0")
        if result.ok:
            n_ok += 1
        else:
            pytest.fail(f"{skeleton_id}: {result.errors}")
    assert n_ok == 5


def test_skeleton_n_turns_within_bounds():
    """骨架生成的 n_turns 大致在 4..12 范围。"""
    rng = random.Random(0)
    runner = SkeletonRunner(dict_split="train", stage="0")
    sk = SKELETONS["S5"]
    for _ in range(20):
        sample = runner.run(sk, rng)
        n_turns = len(sample.conversations) // 2
        assert 1 <= n_turns <= 12, f"n_turns={n_turns} out of bounds"


def test_skeleton_S5_produces_stale_correction():
    """S5 = [A, D, {En}, K]：K 的 asst 应包含新值（覆盖后纠正）。"""
    rng = random.Random(42)
    runner = SkeletonRunner(dict_split="train", stage="0")
    sk = SKELETONS["S5"]
    for _ in range(10):
        sample = runner.run(sk, rng)
        d = sample.to_dict()
        # 最后一轮 assistant 应有 value（K 的新值）
        last_asst = d["conversations"][-1]
        assert last_asst["role"] == "assistant"
        if last_asst.get("value"):
            return  # 至少一次找到
    pytest.fail("10 次 S5 都没产出 K 末轮 value")


def test_skeleton_S10_produces_refusal_after_erase():
    """S10 = [A, {En_long}, L, {En_short}, C_prime]：末轮 asst.value 应为 None（拒答）。"""
    rng = random.Random(7)
    runner = SkeletonRunner(dict_split="train", stage="0")
    sk = SKELETONS["S10"]
    for _ in range(10):
        sample = runner.run(sk, rng)
        d = sample.to_dict()
        last_asst = d["conversations"][-1]
        assert last_asst["role"] == "assistant"
        if last_asst.get("value") is None:
            return
    pytest.fail("10 次 S10 末轮都不是拒答")
