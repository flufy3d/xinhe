"""
测试 StateInterface (v2 对称 cross-attention 架构)
"""
import torch
import pytest
from xinhe.model.state_plugin import (
    StateInterface, SlotAttn,
    CORE_PARAM_PREFIXES, PROJECTION_PARAM_PREFIXES, SLOT_ATTN_PARAM_PREFIXES,
)


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


# --- SlotAttn 测试 ---

def test_slot_attn_shape():
    """SlotAttn 保持输入形状"""
    module = SlotAttn(state_dim=64, n_heads=4)
    state = torch.randn(2, 8, 64)
    out = module(state)
    assert out.shape == state.shape


def test_slot_attn_identity_init():
    """SlotAttn 严格恒等初始化: 输出等于输入 (续训不冲击 checkpoint)"""
    torch.manual_seed(0)
    module = SlotAttn(state_dim=64, n_heads=4)
    module.eval()  # 关 dropout (虽然这里没用)

    state = torch.randn(2, 8, 64)
    out = module(state)

    # out_proj 和 mlp[-1] 初始全零 → 残差子层输出为 0 → SlotAttn(x) = x
    assert torch.allclose(out, state, atol=1e-6), \
        f"初始 SlotAttn 应为严格恒等, 最大差异={((out - state).abs().max().item())}"


def test_slot_attn_in_interface_identity():
    """StateInterface 内嵌的 slot_attn_read/write 初始恒等 → 与无 slot_attn 版本等价"""
    torch.manual_seed(0)
    iface = StateInterface(n_state=8, state_dim=32, hidden_size=32, n_layers=2, state_scale_init=0.0)

    # 手动调用 slot_attn_read 应为恒等
    state = iface.blank_state(2)
    state = state + torch.randn_like(state)  # 扰动一下
    out_read = iface.slot_attn_read(state)
    assert torch.allclose(out_read, state, atol=1e-6)

    # slot_attn_write 同理
    out_write = iface.slot_attn_write(state)
    assert torch.allclose(out_write, state, atol=1e-6)


def test_slot_attn_has_gradient():
    """SlotAttn 参数接收梯度, 可以离开恒等初始化"""
    module = SlotAttn(state_dim=32, n_heads=4)
    state = torch.randn(1, 4, 32, requires_grad=True)
    out = module(state)
    loss = out.sum()
    loss.backward()

    # out_proj.weight / mlp[-1].weight 初始为 0, 但 grad 应为非零
    assert module.self_attn.out_proj.weight.grad is not None
    assert module.self_attn.out_proj.weight.grad.abs().sum().item() > 0
    assert module.mlp[-1].weight.grad is not None
    assert module.mlp[-1].weight.grad.abs().sum().item() > 0


def test_slot_attn_in_core_params():
    """slot_attn 参数归入 core (backbone 无关, 可迁移)"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)

    core_names = [n for n, _ in iface.named_parameters()
                  if any(n.startswith(p) for p in CORE_PARAM_PREFIXES)]
    slot_attn_names = [n for n, _ in iface.named_parameters()
                       if any(n.startswith(p) for p in SLOT_ATTN_PARAM_PREFIXES)]

    # slot_attn 应是 core 的子集
    assert set(slot_attn_names).issubset(set(core_names))
    # slot_attn 非空
    assert len(slot_attn_names) > 0


def test_slot_attn_parameters_api():
    """slot_attn_parameters() 返回两个 SlotAttn 实例的所有参数"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    slot_attn_params = iface.slot_attn_parameters()
    slot_attn_ids = {id(p) for p in slot_attn_params}

    # 应包含 slot_attn_read 和 slot_attn_write 的全部参数
    expected_ids = set()
    for p in iface.slot_attn_read.parameters():
        expected_ids.add(id(p))
    for p in iface.slot_attn_write.parameters():
        expected_ids.add(id(p))
    assert slot_attn_ids == expected_ids


def test_interface_identity_at_init():
    """恒等初始化下, 含 slot_attn 的 write_from_content 与无 slot_attn 等价

    验证: slot_attn_write 恒等 → state_new 与不做 slot_attn 完全一致
    """
    torch.manual_seed(42)
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)

    state_old = torch.randn(1, 4, 16)
    content = torch.randn(1, 8, 32)

    # 有 slot_attn (当前实现)
    state_with = iface.write_from_content(state_old, content)

    # 手工复现无 slot_attn 版本 (绕过 slot_attn_write)
    Q = iface.write_q(state_old)
    K = content
    V = content
    d = Q.shape[-1]
    attn = torch.softmax(Q @ K.transpose(-2, -1) / (d ** 0.5), dim=-1)
    extracted = attn @ V
    state_new = iface.write_out(extracted)
    # 跳过 slot_attn_write
    gate = torch.sigmoid(iface.gate_proj(torch.cat([state_old, state_new], dim=-1)))
    state_without = gate * state_old + (1 - gate) * state_new

    assert torch.allclose(state_with, state_without, atol=1e-5), \
        f"恒等初始化下 slot_attn 应透传, 最大差异={((state_with - state_without).abs().max().item())}"
