"""
测试 StateInterface (v2 对称 cross-attention 架构)
"""
import torch
import pytest
from xinhe.model.state_plugin import StateInterface, CORE_PARAM_PREFIXES, PROJECTION_PARAM_PREFIXES


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
        assert K.shape == (2, 8, 64)  # (B, n_state, hidden_size)
        assert V.shape == (2, 8, 64)


def test_read_layer_shape(interface):
    """read_layer 输出形状与输入相同"""
    state = interface.blank_state(2)
    kv_pairs = interface.generate_read_kv(state)

    hidden = torch.randn(2, 16, 64)
    output = interface.read_layer(hidden, kv_pairs[0])

    assert output.shape == (2, 16, 64)


def test_read_layer_residual(interface):
    """read_scale 初始近零时，read_layer 输出接近输入 (残差连接)"""
    # state_scale_init=-5.0 → sigmoid ≈ 0.007
    iface = StateInterface(n_state=8, state_dim=64, hidden_size=64, n_layers=4, state_scale_init=-5.0)
    state = iface.blank_state(1)
    kv_pairs = iface.generate_read_kv(state)

    hidden = torch.randn(1, 16, 64)
    output = iface.read_layer(hidden, kv_pairs[0])

    # cross-attn 输出应很小 (K/V 近零)，output ≈ hidden
    diff = (output - hidden).abs().max().item()
    assert diff < 0.1, f"read_layer 应接近恒等, 最大差异={diff}"


def test_write_from_content_shape(interface):
    """write_from_content 输出正确形状"""
    state = interface.blank_state(2)
    content = torch.randn(2, 16, 64)

    state_next = interface.write_from_content(state, content)
    assert state_next.shape == (2, 8, 64)


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
    iface = StateInterface(n_state=8, state_dim=64, hidden_size=64, n_layers=4, state_scale_init=-5.0)
    scale = torch.sigmoid(iface.read_scale).item()
    assert scale < 0.01, f"初始 read_scale 应接近 0，实际为 {scale}"


def test_state_stats(interface):
    """状态统计信息包含所有字段"""
    state = interface.blank_state(1)
    stats = interface.get_state_stats(state)

    assert "read_scale" in stats
    assert "state_norm" in stats
    assert "effective_rank" in stats
    assert stats["effective_rank"] > 0


# --- 维度解耦测试 ---

def test_decoupled_dims():
    """state_dim != hidden_size 时全流程工作正确"""
    iface = StateInterface(n_state=8, state_dim=32, hidden_size=64, n_layers=4)
    state = iface.blank_state(2)          # (2, 8, 32) — state_dim
    assert state.shape == (2, 8, 32)

    # 读侧: K/V 在 hidden_size 空间
    kv_pairs = iface.generate_read_kv(state)
    K, V = kv_pairs[0]
    assert K.shape == (2, 8, 64)   # hidden_size

    # read_layer: hidden_states 在 hidden_size 空间
    hidden = torch.randn(2, 16, 64)
    output = iface.read_layer(hidden, kv_pairs[0])
    assert output.shape == (2, 16, 64)

    # 写侧: content 在 hidden_size 空间, 输出在 state_dim 空间
    content = torch.randn(2, 16, 64)
    state_next = iface.write_from_content(state, content)
    assert state_next.shape == (2, 8, 32)  # state_dim


def test_gradient_flow():
    """梯度能通过 read + write 流动"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    state = iface.blank_state(1)
    kv_pairs = iface.generate_read_kv(state)

    hidden = torch.randn(1, 8, 32)
    output = iface.read_layer(hidden, kv_pairs[0])
    state_next = iface.write_from_content(state, output)

    loss = state_next.sum()
    loss.backward()

    assert iface.read_k_projs[0].weight.grad is not None
    assert iface.write_out.weight.grad is not None
    assert iface.state_emb.grad is not None


# --- 参数分类与迁移测试 ---

def test_core_parameters_classification():
    """core vs projection 参数分类正确"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)

    core_params = iface.core_parameters()
    proj_params = iface.projection_parameters()

    # core 包含 state_emb + gate_proj
    core_names = {n for n, _ in iface.named_parameters()
                  if any(n.startswith(p) for p in CORE_PARAM_PREFIXES)}
    assert len(core_params) == len(core_names)
    assert len(core_params) > 0

    # projection 包含 read_k/v_projs + read_scale + write_q + write_out
    proj_names = {n for n, _ in iface.named_parameters()
                  if any(n.startswith(p) for p in PROJECTION_PARAM_PREFIXES)}
    assert len(proj_params) == len(proj_names)
    assert len(proj_params) > 0

    # 无重叠
    core_ids = {id(p) for p in core_params}
    proj_ids = {id(p) for p in proj_params}
    assert core_ids.isdisjoint(proj_ids)

    # 覆盖所有参数
    all_ids = {id(p) for p in iface.parameters()}
    assert core_ids | proj_ids == all_ids


def test_freeze_core():
    """freeze/unfreeze core 正确设置 requires_grad"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)

    iface.freeze_core()
    for p in iface.core_parameters():
        assert not p.requires_grad
    # projection 不受影响
    for p in iface.projection_parameters():
        assert p.requires_grad

    iface.unfreeze_core()
    for p in iface.core_parameters():
        assert p.requires_grad


def test_core_state_dict():
    """core_state_dict 只返回核心参数"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    core = iface.core_state_dict()

    for k in core:
        assert any(k.startswith(p) for p in CORE_PARAM_PREFIXES), f"非 core key: {k}"

    # 不含 projection
    for k in core:
        assert not any(k.startswith(p) for p in PROJECTION_PARAM_PREFIXES)


def test_core_state_dict_roundtrip():
    """从小 interface 提取 core 加载到大 interface (不同 hidden_size/n_layers)"""
    # 源: state_dim=16, hidden_size=16, n_layers=2
    src = StateInterface(n_state=4, state_dim=16, hidden_size=16, n_layers=2)
    with torch.no_grad():
        src.state_emb.fill_(1.23)

    core = src.core_state_dict()

    # 目标: state_dim=16, hidden_size=64, n_layers=4
    dst = StateInterface(n_state=4, state_dim=16, hidden_size=64, n_layers=4)

    result = dst.load_state_dict(core, strict=False)
    # read/write projections 应在 missing_keys 中
    assert any("read_k_projs" in k for k in result.missing_keys)
    assert any("write_q" in k for k in result.missing_keys)

    # core 参数应该已加载
    assert torch.allclose(dst.state_emb, torch.full_like(dst.state_emb, 1.23))
