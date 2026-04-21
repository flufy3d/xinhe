"""
测试 StateInterface (v5b: 对称 cross-attn + 动态 gate + contrastive value head)
"""
import torch
import pytest
from xinhe.model.state_plugin import StateInterface


@pytest.fixture
def interface():
    return StateInterface(n_state=8, state_dim=64, hidden_size=64, n_layers=4)


def test_blank_state_shape(interface):
    """空白状态形状正确"""
    state = interface.blank_state(batch_size=2)
    assert state.shape == (2, 8, 64)


def test_generate_read_kv_shape(interface):
    """generate_read_kv 返回正确数量和形状的 K/V 对"""
    state = interface.blank_state(2)
    kv_pairs = interface.generate_read_kv(state)

    assert len(kv_pairs) == 4  # n_layers=4
    for K, V in kv_pairs:
        assert K.shape == (2, 8, 64)
        assert V.shape == (2, 8, 64)


def test_read_layer_shape(interface):
    """read_layer 输出形状与输入相同"""
    state = interface.blank_state(2)
    kv_pairs = interface.generate_read_kv(state)

    hidden = torch.randn(2, 16, 64)
    output = interface.read_layer(hidden, kv_pairs[0])

    assert output.shape == (2, 16, 64)


def test_read_layer_residual():
    """read_scale 初始近零时, read_layer 输出接近输入 (残差连接)"""
    iface = StateInterface(
        n_state=8, state_dim=64, hidden_size=64, n_layers=4,
        state_scale_init=-5.0,
    )
    state = iface.blank_state(1)
    kv_pairs = iface.generate_read_kv(state)

    hidden = torch.randn(1, 16, 64)
    output = iface.read_layer(hidden, kv_pairs[0])

    diff = (output - hidden).abs().max().item()
    assert diff < 0.1, f"read_layer 应接近恒等, 最大差异={diff}"


def test_write_from_content_shape(interface):
    """write_from_content 输出正确形状, 返回 (state_next, write_attn)"""
    state = interface.blank_state(2)
    content = torch.randn(2, 16, 64)
    state_next, write_attn = interface.write_from_content(state, content)
    assert state_next.shape == (2, 8, 64)
    assert write_attn.shape == (2, 8, 16)  # (B, n_state, T)
    # attn 每 slot 对 T 的分布 sum=1
    assert torch.allclose(write_attn.sum(dim=-1),
                          torch.ones(2, 8), atol=1e-5)


def test_gate_range(interface):
    """gate 值在 [0, 1] 范围内"""
    state_old = torch.randn(2, 8, 64)
    state_new = torch.randn(2, 8, 64)
    combined = torch.cat([state_old, state_new], dim=-1)
    gate = torch.sigmoid(interface.gate_proj(combined))
    assert (gate >= 0).all()
    assert (gate <= 1).all()


def test_scale_init_near_zero():
    """初始 read_scale 接近 0"""
    iface = StateInterface(
        n_state=8, state_dim=64, hidden_size=64, n_layers=4,
        state_scale_init=-5.0,
    )
    scale = torch.sigmoid(iface.read_scale).item()
    assert scale < 0.01, f"初始 read_scale 应接近 0, 实际为 {scale}"


def test_state_stats(interface):
    """状态统计信息包含所有字段"""
    state = interface.blank_state(1)
    stats = interface.get_state_stats(state)
    assert "read_scale" in stats
    assert "state_norm" in stats
    assert "effective_rank" in stats
    assert stats["effective_rank"] > 0


def test_decoupled_dims():
    """state_dim != hidden_size 时全流程工作正确"""
    iface = StateInterface(n_state=8, state_dim=32, hidden_size=64, n_layers=4)
    state = iface.blank_state(2)
    assert state.shape == (2, 8, 32)

    kv_pairs = iface.generate_read_kv(state)
    K, V = kv_pairs[0]
    assert K.shape == (2, 8, 64)

    hidden = torch.randn(2, 16, 64)
    output = iface.read_layer(hidden, kv_pairs[0])
    assert output.shape == (2, 16, 64)

    content = torch.randn(2, 16, 64)
    state_next, _ = iface.write_from_content(state, content)
    assert state_next.shape == (2, 8, 32)


def test_gradient_flow():
    """梯度能通过 read + write 流动"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    state = iface.blank_state(1)
    kv_pairs = iface.generate_read_kv(state)

    hidden = torch.randn(1, 8, 32)
    output = iface.read_layer(hidden, kv_pairs[0])
    state_next, _ = iface.write_from_content(state, output)

    loss = state_next.sum()
    loss.backward()

    assert iface.read_k_projs[0].weight.grad is not None
    assert iface.write_out.weight.grad is not None
    assert iface.state_emb.grad is not None


def test_v5a_minimal_surface():
    """v5a 确认已删除 v4 的 EKS / SlotAttn / 参数分组 API"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    # 删除的字段
    for attr in ["slot_keys", "key_proj", "temperature", "eks_alpha",
                 "slot_attn_write", "slot_attn_read", "last_write_routing"]:
        assert not hasattr(iface, attr), f"v5a 应已删除 {attr}"
    # 删除的 API
    for api in ["core_parameters", "projection_parameters", "slot_attn_parameters",
                "freeze_core", "unfreeze_core", "core_state_dict"]:
        assert not hasattr(iface, api), f"v5a 应已删除 {api}() 方法"


def test_v5b_value_head_exists():
    """v5b: value_head 存在且是 state_dim → hidden_size 线性层"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    assert hasattr(iface, "value_head"), "v5b: 必须有 value_head"
    assert iface.value_head.weight.shape == (32, 16)  # (hidden, state_dim)


def test_v5b_gradient_flow_through_value_head():
    """value_head 可以接收梯度"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    state = iface.blank_state(1)
    slot_repr = iface.value_head(state)
    loss = slot_repr.sum()
    loss.backward()
    assert iface.value_head.weight.grad is not None
    assert iface.value_head.weight.grad.abs().sum().item() > 0


def test_param_count_budget():
    """v5b state plugin 参数量应 ~17M 在 state_dim=1024, hidden=1024, n_layers=6 下"""
    iface = StateInterface(n_state=32, state_dim=1024, hidden_size=1024, n_layers=6)
    total = sum(p.numel() for p in iface.parameters())
    assert 15_000_000 < total < 20_000_000, \
        f"参数量超预算, 实际 {total:,} (目标 ~17M)"
