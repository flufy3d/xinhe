"""
测试 XinheModel 整体前向传播 (v7: 单 Hippocampus, 单 W)

使用 mock backbone 替代真实模型，避免依赖预训练权重。
"""
import torch
import pytest
import torch.nn as nn

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.xinhe_model import XinheModel


N_LAYERS = 4


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

    def forward_blocks(self, hidden_states, attention_mask=None, position_ids=None, layer_hook=None):
        if layer_hook is not None:
            for i in range(N_LAYERS):
                hidden_states = layer_hook(hidden_states, i)
        return self.linear(hidden_states)

    def get_lm_head(self):
        return self._lm_head

    def get_hidden_size(self):
        return self._hidden_size

    def get_num_layers(self):
        return N_LAYERS


@pytest.fixture
def model():
    """创建带 mock backbone 的测试模型 (v7)"""
    config = XinheConfig(
        hidden_size=64,
        n_heads=4,
        head_dim=16,
        read_scale_init=-5.0,
        lora_rank=0,  # 不用 LoRA
        freeze_backbone=False,
    )
    backbone = MockBackbone(hidden_size=64, vocab_size=100)
    m = XinheModel(config, backbone=backbone)
    return m


def test_forward_shape(model):
    """forward 输出形状正确 (v7: W=(B,H,d_v,d_k) 单张量)"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state)

    assert result["logits"].shape == (B, T, 100)
    assert result["state_next"].shape == (B, 4, 16, 16)


def test_forward_with_labels(model):
    """提供 labels 时返回 loss"""
    B, T = 2, 16
    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    state = model.init_state(B)

    result = model(input_ids, state, labels=labels)

    assert "loss" in result
    assert result["loss"].dim() == 0
    assert result["loss"].item() > 0


def test_state_persistence(model):
    """状态在多个 segment 间传递（v7: 单 W 持续演化）"""
    B, T = 1, 8
    state = model.init_state(B)
    states = [state.clone()]

    for _ in range(3):
        input_ids = torch.randint(0, 100, (B, T))
        result = model(input_ids, state)
        state = result["state_next"]
        states.append(state.clone())

    # W 应该在每步变化
    for i in range(1, len(states)):
        assert not torch.allclose(states[i], states[0], atol=1e-6), \
            f"W 在 segment {i} 没有变化"


def test_gradient_flow(model):
    """梯度能通过状态反向传播"""
    B, T = 1, 8
    state = model.init_state(B)

    input_ids_1 = torch.randint(0, 100, (B, T))
    result_1 = model(input_ids_1, state)
    state_1 = result_1["state_next"]

    input_ids_2 = torch.randint(0, 100, (B, T))
    labels_2 = torch.randint(0, 100, (B, T))
    result_2 = model(input_ids_2, state_1, labels=labels_2)

    loss = result_2["loss"]
    loss.backward()

    # Hippocampus 参数应该有梯度
    assert model.hippocampus.k_proj.weight.grad is not None
    assert model.hippocampus.v_proj.weight.grad is not None
    assert model.hippocampus.beta_proj.weight.grad is not None


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

    new_tokens = generated[0, input_ids.shape[1]:]
    assert not (new_tokens == new_tokens[0]).all()


def test_trainable_params(model):
    """可训练参数数量 > 0"""
    params = model.get_trainable_params()
    assert len(params) > 0
    count = model.get_trainable_param_count()
    assert count > 0


def test_forward_with_decoupled_dims():
    """n_heads*head_dim != hidden_size 时全流程正确 (v7)"""
    config = XinheConfig(
        hidden_size=64,
        n_heads=4,
        head_dim=8,            # 4*8=32 != hidden_size=64
        read_scale_init=0.0,
        lora_rank=0,
        freeze_backbone=False,
    )
    backbone = MockBackbone(hidden_size=64, vocab_size=100)
    model = XinheModel(config, backbone=backbone)

    B, T = 2, 16
    state = model.init_state(B)
    assert state.shape == (B, 4, 8, 8)

    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    result = model(input_ids, state, labels=labels)

    assert result["logits"].shape == (B, T, 100)
    assert result["state_next"].shape == (B, 4, 8, 8)
    assert result["loss"].item() > 0

    state_next = result["state_next"]
    state_loss = state_next.sum() + result["loss"]
    state_loss.backward()
    assert model.hippocampus.q_projs[0].weight.grad is not None
    assert model.hippocampus.v_proj.weight.grad is not None
