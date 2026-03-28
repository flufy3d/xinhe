"""
测试 XinheModel 整体前向传播

使用 mock backbone 替代 MiniMind，避免依赖预训练权重。
"""
import torch
import pytest
import torch.nn as nn

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.state_plugin import StatePlugin
from xinhe.model.xinhe_model import XinheModel


class MockBackbone(nn.Module, BackboneBase):
    """用于测试的 mock backbone"""

    def __init__(self, hidden_size=64, vocab_size=100):
        nn.Module.__init__(self)
        self._hidden_size = hidden_size
        self.embed_layer = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed(self, input_ids):
        return self.embed_layer(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None):
        return self.linear(hidden_states)

    def get_lm_head(self):
        return self._lm_head

    def get_hidden_size(self):
        return self._hidden_size


@pytest.fixture
def model():
    """创建带 mock backbone 的测试模型"""
    config = XinheConfig(
        hidden_size=64,
        n_state=4,
        state_dim=64,
        state_scale_init=-5.0,
        lora_rank=0,  # 不用 LoRA
        freeze_backbone=False,
    )
    backbone = MockBackbone(hidden_size=64, vocab_size=100)
    m = XinheModel(config, backbone=backbone)
    return m


def test_forward_shape(model):
    """forward 输出形状正确"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state)

    assert result["logits"].shape == (B, T, 100)
    assert result["state_next"].shape == (B, 4, 64)


def test_forward_with_labels(model):
    """提供 labels 时返回 loss"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state, labels=labels)

    assert "loss" in result
    assert result["loss"].dim() == 0  # scalar
    assert result["loss"].item() > 0


def test_state_persistence(model):
    """状态在多个 segment 间传递"""
    B, T = 1, 8
    state = model.init_state(B)
    states = [state.clone()]

    for _ in range(3):
        input_ids = torch.randint(0, 100, (B, T))
        result = model(input_ids, state)
        state = result["state_next"]
        states.append(state.clone())

    # 状态应该在变化
    for i in range(1, len(states)):
        assert not torch.allclose(states[i], states[0], atol=1e-6), \
            f"状态在 segment {i} 没有变化"


def test_sleep(model):
    """sleep pass 不改变状态形状"""
    state = model.init_state(2)
    state_after = model.sleep(state)
    assert state_after.shape == state.shape


def test_gradient_flow(model):
    """梯度能通过状态反向传播"""
    B, T = 1, 8
    state = model.init_state(B)

    # 两个 segment，不 detach
    input_ids_1 = torch.randint(0, 100, (B, T))
    result_1 = model(input_ids_1, state)
    state_1 = result_1["state_next"]  # 不 detach

    input_ids_2 = torch.randint(0, 100, (B, T))
    labels_2 = torch.randint(0, 100, (B, T))
    result_2 = model(input_ids_2, state_1, labels=labels_2)

    loss = result_2["loss"]
    loss.backward()

    # StatePlugin 参数应该有梯度
    assert model.plugin.gate_bias.grad is not None
    assert model.plugin.state_emb.grad is not None


def test_generate(model):
    """generate_with_state 能生成 token"""
    B = 1
    state = model.init_state(B)
    input_ids = torch.randint(0, 100, (B, 4))

    with torch.no_grad():
        generated, new_state = model.generate_with_state(
            input_ids, state, max_new_tokens=8, temperature=1.0, top_p=1.0,
        )

    assert generated.shape[1] > input_ids.shape[1]
    assert new_state.shape == state.shape


def test_trainable_params(model):
    """可训练参数数量 > 0"""
    params = model.get_trainable_params()
    assert len(params) > 0
    count = model.get_trainable_param_count()
    assert count > 0
