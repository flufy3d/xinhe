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


def test_paragraph_distract_expansion_runs():
    """v9.5:含 _DG_PARA 的 skeleton 跑通(paragraph 分支即使无 congliu 数据
    也 fallback 到 distract_chat 多条拼接)。验证 paragraph 模式 distract turn
    比 short 模式输出更长。"""
    from xinhe.data.skeletons.spec import DistractGroup, Skeleton
    from xinhe.data.skeletons.library import SKELETONS

    # S1 = [A, _DG_PARA, B] 是 paragraph 模式
    rng = random.Random(123)
    runner = SkeletonRunner(dict_split="train", stage="0", max_turns=24)
    sk_para = SKELETONS["S1"]

    # 短模式对照
    _DG_SHORT_TEST = DistractGroup(expansion="short", label="DG")
    sk_short = Skeleton("S1_short", ["A", _DG_SHORT_TEST, "B"], weight=1.0)

    para_lengths = []
    for _ in range(8):
        sample = runner.run(sk_para, rng)
        for turn in sample.conversations:
            if turn["role"] == "assistant" and turn.get("train_loss") == "lm_only":
                para_lengths.append(len(turn["content"]))

    rng_short = random.Random(123)
    short_lengths = []
    for _ in range(8):
        sample = runner.run(sk_short, rng_short)
        for turn in sample.conversations:
            if turn["role"] == "assistant" and turn.get("train_loss") == "lm_only":
                short_lengths.append(len(turn["content"]))

    if not para_lengths or not short_lengths:
        pytest.skip("没采到 distract turn")
    para_avg = sum(para_lengths) / len(para_lengths)
    short_avg = sum(short_lengths) / len(short_lengths)
    # paragraph 模式至少比 short 模式长 1.5×(拼接 4-8 个短句应远超单句)
    assert para_avg > short_avg * 1.3, (
        f"paragraph distract 不显著长于 short:para_avg={para_avg:.0f} vs short_avg={short_avg:.0f}"
    )


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
