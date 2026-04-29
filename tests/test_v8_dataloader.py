"""DataLoader 端到端：v8 schema → 正确的 token weight 分配。"""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from xinhe.data.conversation import (
    ConversationDataset,
    tokenize_turn,
    _resolve_lm_weight,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def test_resolve_lm_weight():
    assert _resolve_lm_weight("true") == (1.0, True)
    assert _resolve_lm_weight(True) == (1.0, True)
    assert _resolve_lm_weight("lm_only") == (0.1, False)
    assert _resolve_lm_weight("false") == (0.0, False)
    assert _resolve_lm_weight(False) == (0.0, False)


def test_tokenize_turn_value_span_basic(tokenizer):
    """value_span 5.0 weight 加到对应 token。"""
    user = "我喜欢吃什么？"
    asst = "你喜欢苹果。"
    # value_span: 苹果 在 asst 中是 char [3, 5]
    ids, labels, weights = tokenize_turn(
        tokenizer, user, asst, turn_max_tokens=64,
        train_loss="true",
        value_spans=[[3, 5]],
        weight_per_span=5.0,
    )
    # 至少应有一个 weight=5.0 的 token（VALUE token）
    assert (weights >= 4.9).any()
    # 其他 assistant token weight=1.0
    assert (weights == 1.0).any()


def test_tokenize_turn_lm_only(tokenizer):
    """lm_only 段 weight 全 0.1，无 value 加权。"""
    user = "今天怎么样？"
    asst = "还行,看着挺舒服。"
    ids, labels, weights = tokenize_turn(
        tokenizer, user, asst, turn_max_tokens=64,
        train_loss="lm_only",
        value_spans=[],
        weight_per_span=0.0,
    )
    nonzero = weights[weights > 0]
    assert len(nonzero) > 0
    assert (nonzero == 0.1).all()


def test_tokenize_turn_false_no_loss(tokenizer):
    """train_loss=false 时 labels 全 -100，weights 全 0。"""
    user = "你好"
    asst = "你好"
    ids, labels, weights = tokenize_turn(
        tokenizer, user, asst, turn_max_tokens=64,
        train_loss="false",
    )
    assert (labels == -100).all()
    assert (weights == 0).all()


def test_tokenize_turn_multi_value_budget(tokenizer):
    """多 value 时 weight = base / N（per-sample 守恒）。"""
    user = "我家有什么？"
    asst = "你家猫叫Leo,狗叫Max。"
    # 假设 weight=5.0/2 = 2.5 per span
    ids, labels, weights = tokenize_turn(
        tokenizer, user, asst, turn_max_tokens=64,
        train_loss="true",
        value_spans=[[3, 6], [9, 12]],   # Leo, Max
        weight_per_span=2.5,
    )
    # 应有 token weight = 2.5
    assert (torch.isclose(weights, torch.tensor(2.5))).any()


def test_dataset_loads_v8_jsonl(tmp_path, tokenizer):
    """ConversationDataset 应能正确读 v8 schema 多 turn 数据。"""
    sample = {
        "sample_id": "test1",
        "stage": "0",
        "skeleton_id": "S1",
        "meta": {"n_turns": 2},
        "conversations": [
            {"role": "user", "content": "我喜欢吃什么？"},
            {
                "role": "assistant", "content": "你喜欢苹果。",
                "train_loss": "true",
                "value": ["苹果"], "value_span": [[3, 5]],
                "value_tier": "hard", "weight_per_span": 5.0,
            },
            {"role": "user", "content": "今天天气如何？"},
            {
                "role": "assistant", "content": "还可以。",
                "train_loss": "lm_only",
                "value": None, "value_span": [],
                "value_tier": None, "weight_per_span": 0.0,
            },
        ],
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    ds = ConversationDataset(str(path), tokenizer, turn_max_tokens=64, max_turns_per_episode=4)
    assert len(ds) == 1
    ep = ds[0]
    assert len(ep) == 4   # padded to max_turns_per_episode
    # 第 1 个 turn: hard value，应有 5.0 weight
    _, _, w0 = ep[0]
    assert (w0 >= 4.9).any()
    # 第 2 个 turn: lm_only，只有 0.1 weight
    _, _, w1 = ep[1]
    nonzero = w1[w1 > 0]
    if len(nonzero) > 0:
        assert (nonzero == 0.1).all()
