"""
Curriculum data (v7.1) —— 每个 turn kind + pattern 基础单元测试

检查：
  - 注册表完整（9 turn_kinds + 10 patterns + 8 val）
  - 每个 turn kind 能生成合理 dict
  - 每个 pattern 能生成非空 episode
  - 每个 val 生成器能产生带 value 的 episode
"""
import random
import pytest

from xinhe.data.registry import TURN_KIND_FNS, PATTERN_FNS, VAL_FNS
from xinhe.data.persona import sample_persona
from xinhe.data.curriculum_data import generate_episode, TurnCache


EXPECTED_TURN_KINDS = {
    "reveal_single", "reveal_multi", "recall", "refusal", "overwrite",
    "general_chat", "world_qa", "compositional", "third_party",
}
EXPECTED_PATTERNS = {
    "stress_retention", "multi_slot_retention", "rapid_overwrite", "fact_vs_transient",
    "verbatim_recall", "adversarial_temporal",
    "reference_back", "context_followup", "topic_continuation", "entity_tracking",
}
EXPECTED_VAL = {
    "multi_slot_retention", "rapid_overwrite", "irrelevant_forget",
    "verbatim",
    "reference_back", "context_followup", "topic_continuation", "entity_tracking",
}


def _fake_cache():
    c = TurnCache()
    c.chat_turns = [{"user": "今天天气", "assistant": "不错哦"}]
    c.qa_turns = [{"user": "1+1=?", "assistant": "等于 2"}]
    return c


# ═══════════════════════════════════════════════════════════════════
# 注册完整性
# ═══════════════════════════════════════════════════════════════════

def test_all_turn_kinds_registered():
    missing = EXPECTED_TURN_KINDS - set(TURN_KIND_FNS.keys())
    assert not missing, f"缺 turn_kind: {missing}"


def test_all_patterns_registered():
    missing = EXPECTED_PATTERNS - set(PATTERN_FNS.keys())
    assert not missing, f"缺 pattern: {missing}"


def test_all_val_registered():
    missing = EXPECTED_VAL - set(VAL_FNS.keys())
    assert not missing, f"缺 val: {missing}"


def test_val_has_eval_fn_after_patch():
    """persona_joint.py 的 _patch_val_eval_fns 应给所有 val 注入 eval_fn。"""
    # Force persona_joint.py load to trigger patching
    from xinhe.evaluation import persona_joint  # noqa: F401
    for name, (_gen, eval_fn) in VAL_FNS.items():
        assert eval_fn is not None, f"val {name} 缺 eval_fn"


# ═══════════════════════════════════════════════════════════════════
# turn kinds 基本生成
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kind", sorted(EXPECTED_TURN_KINDS))
def test_turn_kind_generates(kind):
    rng = random.Random(0)
    persona = sample_persona(rng, num_reveal=5)
    cache = _fake_cache()
    fn = TURN_KIND_FNS[kind]
    # some kinds need revealed slots; reveal a few first
    if kind in ("recall", "overwrite", "compositional"):
        TURN_KIND_FNS["reveal_single"](rng, persona, cache)
        TURN_KIND_FNS["reveal_single"](rng, persona, cache)

    attempts = 0
    t = None
    while t is None and attempts < 20:
        attempts += 1
        t = fn(rng, persona, cache)
    # 不强制每次都能生成（某些 kind 在 persona 状态不对时返回 None）
    if t is not None:
        assert "user" in t and "assistant" in t
        assert isinstance(t.get("train_loss", True), bool)


# ═══════════════════════════════════════════════════════════════════
# patterns 基本生成
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(EXPECTED_PATTERNS))
def test_pattern_generates_episode(name):
    rng = random.Random(42)
    cache = _fake_cache()
    fn = PATTERN_FNS[name]
    # 尝试多次（某些 pattern 需要 persona.reveal_order 覆盖特定 slot）
    attempts = 0
    episode = []
    while not episode and attempts < 20:
        attempts += 1
        episode = fn(rng, None, cache)
    assert episode, f"pattern {name} 20 次尝试均返回空"
    assert all("user" in t and "assistant" in t for t in episode)


# ═══════════════════════════════════════════════════════════════════
# val 生成器
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(EXPECTED_VAL))
def test_val_generator(name):
    rng = random.Random(100)
    cache = _fake_cache()
    gen_fn, _ = VAL_FNS[name]
    episodes = gen_fn(rng, cache, n_samples=3)
    assert len(episodes) >= 1, f"val {name} 无法生成 episode"
    # 最后一个 turn 应该带 value
    for ep in episodes:
        last = ep[-1]
        assert last.get("value"), f"val {name} 最后 turn 无 value"


# ═══════════════════════════════════════════════════════════════════
# 端到端 generate_episode
# ═══════════════════════════════════════════════════════════════════

def test_generate_episode_basic_rw():
    rng = random.Random(0)
    cache = _fake_cache()
    cfg = {
        "turn_kinds": {"reveal_single": 0.5, "recall": 0.5},
        "patterns": {},
        "min_turns": 2, "max_turns": 2,
    }
    ep = generate_episode(rng, cfg, cache)
    assert len(ep) <= 2


def test_generate_episode_verbatim_gate():
    rng = random.Random(0)
    cache = _fake_cache()
    cfg = {
        "turn_kinds": {},
        "patterns": {"verbatim_recall": 1.0},
        "min_turns": 2, "max_turns": 3,
    }
    ep = generate_episode(rng, cfg, cache)
    # 最后一个 turn 应该是 recall，有 value
    assert ep[-1].get("value")


def test_generate_episode_mix_with_patterns():
    """混合 kinds + patterns：有 pattern_prob 概率走 pattern，其余走 turn_kinds。"""
    rng = random.Random(0)
    cache = _fake_cache()
    cfg = {
        "turn_kinds": {"reveal_single": 0.5, "recall": 0.5},
        "patterns": {"reference_back": 0.3},
        "min_turns": 4, "max_turns": 6,
    }
    # 跑多次，统计
    n_with_value = 0
    for _ in range(20):
        ep = generate_episode(rng, cfg, cache)
        if ep and ep[-1].get("value"):
            n_with_value += 1
    assert n_with_value > 0, "20 次采样没一次生成带 value 的 episode"


# ═══════════════════════════════════════════════════════════════════
# 配置 yaml 解析
# ═══════════════════════════════════════════════════════════════════

def test_curriculum_yaml_loads():
    from xinhe.model.config import XinheConfig
    cfg, curriculum = XinheConfig.from_yaml("configs/persona_unified_0.8b.yaml")
    assert len(curriculum) == 5, f"期望 5 stages，得到 {len(curriculum)}"
    stage_names = [s["name"] for s in curriculum]
    assert stage_names == [
        "0a_basic_rw", "0b_verbatim_rw", "0c_context_review",
        "0d_irrelevant_forget", "1_persona_unified",
    ]
