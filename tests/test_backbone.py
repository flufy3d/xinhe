"""
测试 Backbone 接口
"""
import torch
import pytest
import torch.nn as nn

from xinhe.model.backbone import BackboneBase


class DummyBackbone(nn.Module, BackboneBase):
    """最简 backbone 实现，验证接口"""

    def __init__(self, hidden_size=32, vocab_size=50):
        nn.Module.__init__(self)
        self._hidden_size = hidden_size
        self._embed = nn.Embedding(vocab_size, hidden_size)
        self._blocks = nn.Linear(hidden_size, hidden_size)
        self._head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed(self, input_ids):
        return self._embed(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None, position_ids=None):
        return self._blocks(hidden_states)

    def get_lm_head(self):
        return self._head

    def get_hidden_size(self):
        return self._hidden_size


def test_interface_contract():
    """验证 BackboneBase 接口"""
    backbone = DummyBackbone()

    # embed
    ids = torch.randint(0, 50, (2, 10))
    emb = backbone.embed(ids)
    assert emb.shape == (2, 10, 32)

    # forward_blocks
    out = backbone.forward_blocks(emb)
    assert out.shape == (2, 10, 32)

    # get_lm_head
    head = backbone.get_lm_head()
    logits = head(out)
    assert logits.shape == (2, 10, 50)

    # get_hidden_size
    assert backbone.get_hidden_size() == 32


def test_forward_blocks_with_mask():
    """forward_blocks 接受 attention_mask"""
    backbone = DummyBackbone()
    emb = torch.randn(2, 10, 32)
    mask = torch.zeros(1, 1, 10, 10)

    # 不应报错
    out = backbone.forward_blocks(emb, attention_mask=mask)
    assert out.shape == (2, 10, 32)


def test_forward_blocks_with_position_ids():
    """forward_blocks 接受 position_ids"""
    backbone = DummyBackbone()
    emb = torch.randn(2, 10, 32)
    pos_ids = torch.zeros(1, 10, dtype=torch.long)

    out = backbone.forward_blocks(emb, position_ids=pos_ids)
    assert out.shape == (2, 10, 32)
