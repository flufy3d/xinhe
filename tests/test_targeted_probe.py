"""Phase 1 targeted_probe 的 CPU 自测:只测内存 NIH 生成器 + leakage 计数。

不加载模型、不需 GPU/ckpt(probe 的 model-forward 诊断由用户在 GPU 上跑)。
probe 把模型栈做成 lazy import,故这里 import 只拉 torch + generate_recall_probe(轻量)。
"""
import random
import sys
from pathlib import Path

import pytest

# scripts 不是已安装包(xinhe 才是),import probe 前须把 repo root 放上 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.probe.targeted_probe import make_nih_episode, count_leakage, PATTERN_BY_TYPE


def _roles(convs):
    return [c["role"] for c in convs]


def test_make_nih_episode_schema():
    rng = random.Random(0)
    n = 3
    ep = make_nih_episode(rng, n_distract=n, value_type="name_en")
    convs, meta = ep["conversations"], ep["meta"]

    # N distract + target + query = N+2 个 user/asst 对,严格交替
    assert len(convs) == 2 * (n + 2)
    assert _roles(convs) == ["user", "assistant"] * (n + 2)

    # distractor:N 个互异,且都 != target(同类型不同实体)
    assert meta["n_distract"] == n
    assert len(set(meta["distractors"])) == n
    assert meta["entity"] not in meta["distractors"]

    # query 轮:真 recall(user 不含 entity)+ value_span 命中 target
    q_user, q_asst = convs[-2]["content"], convs[-1]["content"]
    assert meta["entity"] not in q_user
    s, e = convs[-1]["value_span"][0]
    assert q_asst[s:e] == meta["entity"]

    # target 写入轮默认落在所有 distract 之后(target_position == n)
    assert meta["target_position"] == n
    ta = convs[2 * meta["target_position"] + 1]
    ts, te = ta["value_span"][0]
    assert ta["content"][ts:te] == meta["entity"]


@pytest.mark.parametrize("vtype", sorted(PATTERN_BY_TYPE))
def test_all_value_types_valid(vtype):
    rng = random.Random(7)
    ep = make_nih_episode(rng, n_distract=2, value_type=vtype)
    convs, meta = ep["conversations"], ep["meta"]
    assert meta["entity_type"] == vtype
    q_user, q_asst = convs[-2]["content"], convs[-1]["content"]
    assert meta["entity"] not in q_user                 # 真 recall
    s, e = convs[-1]["value_span"][0]
    assert q_asst[s:e] == meta["entity"]                # span 命中


def test_target_position_places_target():
    rng = random.Random(1)
    ep = make_nih_episode(rng, n_distract=4, value_type="name_en", target_position=0)
    convs, target = ep["conversations"], ep["meta"]["entity"]
    ta = convs[1]                                       # 第 0 个写入轮的 assistant
    s, e = ta["value_span"][0]
    assert ta["content"][s:e] == target


def test_pool_too_small_raises():
    rng = random.Random(2)
    # food 池 14 个,n_distract+1=20 > 14 → 必须 raise(fail loud,不 silent fallback)
    with pytest.raises(ValueError):
        make_nih_episode(rng, n_distract=19, value_type="food")


def test_unknown_value_type_raises():
    rng = random.Random(3)
    with pytest.raises(ValueError):
        make_nih_episode(rng, n_distract=2, value_type="nope")


def test_count_leakage():
    assert count_leakage("我叫Bob,不是Carol", ["Bob", "Carol"], "Alice") == 1.0
    assert count_leakage("我叫Alice", ["Bob", "Carol"], "Alice") == 0.0
    assert count_leakage("Bob", ["Bob", "Carol"], "Alice") == 0.5
    assert count_leakage("anything", [], "Alice") == 0.0
    assert count_leakage("Alice Bob", ["Alice", "Bob"], "Alice") == 0.5  # target 不计入泄漏


def test_make_nih_episode_no_disk_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = random.Random(4)
    for _ in range(50):
        make_nih_episode(rng, n_distract=3, value_type="name_en")
    assert list(tmp_path.iterdir()) == []               # 生成器纯内存,不落盘
