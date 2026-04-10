"""
测试 StatePlugin 的各个方法 (读写分离架构)
"""
import torch
import pytest
from xinhe.model.state_plugin import StatePlugin, CORE_PARAM_PREFIXES, PROJECTION_PARAM_PREFIXES


@pytest.fixture
def plugin():
    return StatePlugin(n_state=8, state_dim=64)


def test_blank_state_shape(plugin):
    """空白状态形状正确"""
    state = plugin.blank_state(batch_size=2)
    assert state.shape == (2, 8, 64)


def test_inject_shape(plugin):
    """注入后序列长度 = n_read + T + n_write"""
    state = plugin.blank_state(2)
    content = torch.randn(2, 16, 64)

    hidden, mask = plugin.inject(state, content)

    # [Read(8) | Content(16) | Write(8)] = 32
    assert hidden.shape == (2, 8 + 16 + 8, 64)
    assert mask.shape == (1, 1, 32, 32)


def test_mask_is_causal(plugin):
    """mask 是标准因果: 位置 i 只能看 0..i"""
    state = plugin.blank_state(1)
    content = torch.randn(1, 6, 64)
    _, mask = plugin.inject(state, content)

    m = mask[0, 0]
    total = 8 + 6 + 8  # 22

    for i in range(total):
        for j in range(total):
            if j <= i:
                assert m[i, j] == 0, f"mask[{i},{j}] should be 0 (visible)"
            else:
                assert m[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf (masked)"


def test_no_leakage(plugin):
    """Read-State 看不到 Content，Content 看不到 Write-State"""
    state = plugin.blank_state(1)
    content = torch.randn(1, 6, 64)
    _, mask = plugin.inject(state, content)

    m = mask[0, 0]
    n = 8  # n_state

    # Read-State (pos 0..7) 看不到 Content (pos 8..13) 和 Write-State (pos 14..21)
    for i in range(n):
        for j in range(n, n + 6 + n):
            assert m[i, j] == float("-inf"), f"Read[{i}] should NOT see pos {j}"

    # Content (pos 8..13) 看不到 Write-State (pos 14..21)
    for i in range(n, n + 6):
        for j in range(n + 6, n + 6 + n):
            assert m[i, j] == float("-inf"), f"Content[{i}] should NOT see Write[{j}]"

    # Content CAN see Read-State
    for i in range(n, n + 6):
        for j in range(n):
            assert m[i, j] == 0, f"Content[{i}] SHOULD see Read[{j}]"

    # Write-State CAN see everything before it
    for i in range(n + 6, n + 6 + n):
        for j in range(i):
            assert m[i, j] == 0, f"Write[{i}] SHOULD see pos {j}"


def test_extract_and_update(plugin):
    """提取 + gate 更新，形状正确"""
    state_old = plugin.blank_state(2)
    # output 长度 = read(8) + content(16) + write(8) = 32
    output = torch.randn(2, 32, 64)

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


# --- 维度解耦测试 ---

def test_projection_shapes():
    """state_dim != hidden_size 时投影层工作正确"""
    plugin = StatePlugin(n_state=8, state_dim=32, hidden_size=64)
    state = plugin.blank_state(2)          # (2, 8, 32) — state_dim
    content = torch.randn(2, 16, 64)       # (2, 16, 64) — hidden_size

    hidden, mask = plugin.inject(state, content)
    assert hidden.shape == (2, 8 + 16 + 8, 64)  # 全部在 hidden_size 空间

    output = torch.randn(2, 8 + 16 + 8, 64)     # backbone 输出 hidden_size
    content_out, state_next = plugin.extract_and_update(output, state)
    assert content_out.shape == (2, 16, 64)      # hidden_size
    assert state_next.shape == (2, 8, 32)        # state_dim


def test_no_projection_when_equal():
    """state_dim == hidden_size 时无投影层"""
    plugin = StatePlugin(n_state=8, state_dim=64, hidden_size=64)
    assert plugin.proj_up is None
    assert plugin.proj_down is None


def test_projection_gradient_flow():
    """梯度能通过投影层流动"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=32)
    state = plugin.blank_state(1)
    content = torch.randn(1, 8, 32)

    hidden, _ = plugin.inject(state, content)
    # 模拟 backbone: 简单 identity
    output = hidden.clone()
    content_out, state_next = plugin.extract_and_update(output, state)

    loss = state_next.sum()
    loss.backward()

    assert plugin.proj_up.weight.grad is not None
    assert plugin.proj_down.weight.grad is not None


# --- 参数分类与迁移测试 ---

def test_core_parameters_classification():
    """core vs projection 参数分类正确"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=32)

    core_params = plugin.core_parameters()
    proj_params = plugin.projection_parameters()

    # core 包含所有核心参数
    core_names = {n for n, _ in plugin.named_parameters()
                  if any(n.startswith(p) for p in CORE_PARAM_PREFIXES)}
    assert len(core_params) == len(core_names)
    assert len(core_params) > 0

    # projection 包含 proj_up, proj_down
    proj_names = {n for n, _ in plugin.named_parameters()
                  if any(n.startswith(p) for p in PROJECTION_PARAM_PREFIXES)}
    assert len(proj_params) == len(proj_names)
    assert len(proj_params) == 2  # proj_up.weight + proj_down.weight

    # 无重叠
    core_ids = {id(p) for p in core_params}
    proj_ids = {id(p) for p in proj_params}
    assert core_ids.isdisjoint(proj_ids)

    # 覆盖所有参数
    all_ids = {id(p) for p in plugin.parameters()}
    assert core_ids | proj_ids == all_ids


def test_core_parameters_no_projection():
    """state_dim == hidden_size 时没有投影参数"""
    plugin = StatePlugin(n_state=4, state_dim=32, hidden_size=32)
    assert len(plugin.projection_parameters()) == 0
    assert len(plugin.core_parameters()) == len(list(plugin.parameters()))


def test_freeze_core():
    """freeze/unfreeze core 正确设置 requires_grad"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=32)

    plugin.freeze_core()
    for p in plugin.core_parameters():
        assert not p.requires_grad
    # projection 不受影响
    for p in plugin.projection_parameters():
        assert p.requires_grad

    plugin.unfreeze_core()
    for p in plugin.core_parameters():
        assert p.requires_grad


def test_core_state_dict():
    """core_state_dict 只返回核心参数"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=32)
    core = plugin.core_state_dict()

    for k in core:
        assert any(k.startswith(p) for p in CORE_PARAM_PREFIXES), f"非 core key: {k}"

    # 不含 proj
    for k in core:
        assert not any(k.startswith(p) for p in PROJECTION_PARAM_PREFIXES)


def test_core_state_dict_roundtrip():
    """从小 plugin 提取 core 加载到大 plugin (不同 hidden_size)"""
    # 源: state_dim=16, hidden_size=16 (无投影)
    src = StatePlugin(n_state=4, state_dim=16, hidden_size=16)
    # 修改 core 参数使其非零
    with torch.no_grad():
        src.gate_bias.fill_(1.23)

    core = src.core_state_dict()

    # 目标: state_dim=16, hidden_size=64 (有投影)
    dst = StatePlugin(n_state=4, state_dim=16, hidden_size=64)
    assert dst.proj_up is not None

    result = dst.load_state_dict(core, strict=False)
    # proj_up/proj_down 应在 missing_keys 中
    assert any("proj_up" in k for k in result.missing_keys)
    assert any("proj_down" in k for k in result.missing_keys)

    # core 参数应该已加载
    assert torch.allclose(dst.gate_bias, torch.full_like(dst.gate_bias, 1.23))
