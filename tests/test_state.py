"""
测试 StatePlugin 的各个方法
"""
import torch
import pytest
from xinhe.model.state_plugin import StatePlugin


@pytest.fixture
def plugin():
    return StatePlugin(n_state=8, state_dim=64)


def test_blank_state_shape(plugin):
    """空白状态形状正确"""
    state = plugin.blank_state(batch_size=2)
    assert state.shape == (2, 8, 64)


def test_inject_shape(plugin):
    """注入后序列长度 = n_state + T"""
    state = plugin.blank_state(2)
    content = torch.randn(2, 16, 64)

    hidden, mask = plugin.inject(state, content)

    assert hidden.shape == (2, 8 + 16, 64)
    assert mask.shape == (1, 1, 24, 24)


def test_mask_structure(plugin):
    """mask 结构正确: 状态双向 + 内容因果"""
    mask = plugin.build_mask(n_state=4, n_content=6)
    assert mask.shape == (1, 1, 10, 10)

    m = mask[0, 0]

    # 状态→状态: 全可见 (0)
    assert (m[:4, :4] == 0).all()

    # 状态→内容: 全可见 (0)
    assert (m[:4, 4:] == 0).all()

    # 内容→状态: 全可见 (0)
    assert (m[4:, :4] == 0).all()

    # 内容→内容: 因果 (下三角为0, 上三角为-inf)
    causal = m[4:, 4:]
    # 对角线和下方应该是 0
    for i in range(6):
        for j in range(6):
            if j <= i:
                assert causal[i, j] == 0, f"causal[{i},{j}] should be 0"
            else:
                assert causal[i, j] == float("-inf"), f"causal[{i},{j}] should be -inf"


def test_extract_and_update(plugin):
    """提取 + gate 更新，形状正确"""
    state_old = plugin.blank_state(2)
    output = torch.randn(2, 8 + 16, 64)

    content_out, state_next = plugin.extract_and_update(output, state_old)

    assert content_out.shape == (2, 16, 64)
    assert state_next.shape == (2, 8, 64)


def test_gate_range(plugin):
    """gate 值在 [0, 1] 范围内"""
    state_old = torch.randn(2, 8, 64)
    state_new = torch.randn(2, 8, 64)

    combined = torch.cat([state_old, state_new], dim=-1)
    dynamic_logit = plugin.gate_proj(combined)
    gate = torch.sigmoid(plugin.gate_bias.unsqueeze(0) + dynamic_logit)

    assert (gate >= 0).all()
    assert (gate <= 1).all()


def test_sleep_forward(plugin):
    """sleep pass 不改变状态形状"""
    state = plugin.blank_state(2)

    # 模拟 backbone forward
    def mock_backbone(hidden_states, attention_mask=None):
        return hidden_states + torch.randn_like(hidden_states) * 0.01

    state_next = plugin.sleep_forward(mock_backbone, state)
    assert state_next.shape == state.shape


def test_state_stats(plugin):
    """状态统计信息包含所有字段"""
    state = plugin.blank_state(1)
    stats = plugin.get_state_stats(state)

    assert "scale" in stats
    assert "gate_mean" in stats
    assert "slow_dims" in stats
    assert "fast_dims" in stats
    assert "effective_rank" in stats
    assert stats["effective_rank"] > 0


def test_scale_init_near_zero():
    """初始 scale 接近 0 (不影响模型)"""
    plugin = StatePlugin(n_state=8, state_dim=64, state_scale_init=-5.0)
    scale = torch.sigmoid(plugin.state_scale).item()
    assert scale < 0.01, f"初始 scale 应接近 0，实际为 {scale}"
