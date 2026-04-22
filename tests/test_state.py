"""
测试 StateInterface (v5c: Delta Rule 联想记忆 W)
"""
import inspect
import torch
import pytest
import torch.nn.functional as F

from xinhe.model.state_plugin import StateInterface


@pytest.fixture
def interface():
    return StateInterface(hidden_size=64, n_heads=4, head_dim=16, n_layers=4)


def test_blank_state_shape(interface):
    """空白 W 形状正确: (B, H, d_v, d_k)"""
    W = interface.blank_state(batch_size=2)
    assert W.shape == (2, 4, 16, 16)
    assert (W == 0).all(), "零初始化: 空 W 读出恒为 0"


def test_blank_state_read_is_identity():
    """零 W 时 read_layer 输出 == 输入（scale*0 residual）"""
    iface = StateInterface(hidden_size=32, n_heads=4, head_dim=8, n_layers=2)
    W = iface.blank_state(1)
    hidden = torch.randn(1, 16, 32)
    out = iface.read_layer(hidden, W, layer_idx=0)
    assert torch.allclose(out, hidden, atol=1e-6), \
        "零 W 情况下读路径必须是恒等（避免空态污染）"


def test_read_layer_shape(interface):
    """read_layer 输出形状 (B,T,D) 与输入一致"""
    W = interface.blank_state(2)
    hidden = torch.randn(2, 16, 64)
    out = interface.read_layer(hidden, W, layer_idx=1)
    assert out.shape == (2, 16, 64)


def test_read_scale_init_near_zero():
    """初始 read_scale 接近 0（空态几乎无影响）"""
    iface = StateInterface(
        hidden_size=32, n_heads=4, head_dim=8, n_layers=2,
        read_scale_init=-5.0,
    )
    scale = torch.sigmoid(iface.read_scale).item()
    assert scale < 0.01, f"初始 read_scale 应接近 0，实际为 {scale}"


def test_write_from_content_shape(interface):
    """write_from_content 返回单个 W: (B,H,d_v,d_k)"""
    W = interface.blank_state(2)
    content = torch.randn(2, 16, 64)
    W_new = interface.write_from_content(W, content)
    assert W_new.shape == (2, 4, 16, 16)


def test_delta_rule_overwrite():
    """同 key 先写 v1 再写 v2，读回偏向 v2（覆写语义）"""
    iface = StateInterface(hidden_size=32, n_heads=1, head_dim=8, n_layers=1)
    # 手动构造: 2 token 序列，相同 key，不同 value
    # 绕开投影：直接操作 W 验证数学
    B, H, d_k, d_v = 1, 1, 8, 8
    W = torch.zeros(B, H, d_v, d_k)
    k = F.normalize(torch.randn(B, H, d_k), dim=-1)              # 单位 key
    v1 = torch.randn(B, H, d_v)
    v2 = torch.randn(B, H, d_v)
    beta = 0.8

    # 第一次写 v1
    v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
    W = W + beta * torch.einsum("bhv,bhd->bhvd", v1 - v_hat, k)
    # 第二次写 v2 (同 key)
    v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
    W = W + beta * torch.einsum("bhv,bhd->bhvd", v2 - v_hat, k)

    # 读出：W @ k 应更接近 v2 而非 v1
    read = torch.einsum("bhvd,bhd->bhv", W, k)
    dist_to_v2 = (read - v2).norm()
    dist_to_v1 = (read - v1).norm()
    assert dist_to_v2 < dist_to_v1, \
        f"覆写后应更接近 v2，但 |read-v1|={dist_to_v1:.3f} < |read-v2|={dist_to_v2:.3f}"


def test_delta_rule_subtract_interference():
    """正交 k1/k2 分别写 v1/v2，读 k1 应接近 v1 不受 k2 污染"""
    B, H, d_k, d_v = 1, 1, 8, 8
    W = torch.zeros(B, H, d_v, d_k)
    # 构造正交 key
    k1 = torch.zeros(B, H, d_k); k1[0, 0, 0] = 1.0
    k2 = torch.zeros(B, H, d_k); k2[0, 0, 1] = 1.0
    v1 = torch.randn(B, H, d_v)
    v2 = torch.randn(B, H, d_v)
    beta = 1.0

    for k, v in [(k1, v1), (k2, v2)]:
        v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
        W = W + beta * torch.einsum("bhv,bhd->bhvd", v - v_hat, k)

    # 读 k1 应精确复原 v1（因为 k1 与 k2 正交，干扰项 k1·k2=0）
    read = torch.einsum("bhvd,bhd->bhv", W, k1)
    assert torch.allclose(read, v1, atol=1e-5), \
        f"正交写入无干扰下 read(k1) 应 ≈ v1，实际 |diff|={((read-v1).norm()).item():.5f}"


def test_no_softmax_in_write():
    """静态断言 write_from_content 里没有 F.softmax / torch.softmax"""
    src = inspect.getsource(StateInterface.write_from_content)
    assert "softmax" not in src.lower(), \
        "Delta Rule 写路径绝对不允许 softmax（零 Softmax 约束）"


def test_gradient_flow():
    """梯度能通过 read + write 流回各个投影"""
    iface = StateInterface(hidden_size=32, n_heads=2, head_dim=8, n_layers=2)
    W = iface.blank_state(1)
    hidden = torch.randn(1, 8, 32)
    out = iface.read_layer(hidden, W, layer_idx=0)
    W_new = iface.write_from_content(W, out)
    loss = W_new.sum()
    loss.backward()

    assert iface.q_projs[0].weight.grad is not None
    assert iface.o_projs[0].weight.grad is not None
    assert iface.k_proj.weight.grad is not None
    assert iface.v_proj.weight.grad is not None
    assert iface.beta_proj.weight.grad is not None


def test_decoupled_dims():
    """n_heads * head_dim 可以不等于 hidden_size（内部投影桥接）"""
    iface = StateInterface(hidden_size=64, n_heads=8, head_dim=32, n_layers=2)
    W = iface.blank_state(2)
    assert W.shape == (2, 8, 32, 32)
    hidden = torch.randn(2, 16, 64)
    out = iface.read_layer(hidden, W, 0)
    assert out.shape == (2, 16, 64)
    content = torch.randn(2, 16, 64)
    W_new = iface.write_from_content(W, content)
    assert W_new.shape == (2, 8, 32, 32)


def test_v5c_minimal_surface():
    """v5c: 确认已删除 v5a/v5b 的 slot 架构字段和 API"""
    iface = StateInterface(hidden_size=32, n_heads=2, head_dim=8, n_layers=2)
    # 删除的字段
    for attr in [
        # v4 EKS 遗产
        "slot_keys", "key_proj", "temperature", "eks_alpha",
        "slot_attn_write", "slot_attn_read", "last_write_routing",
        # v5a/v5b 删除的
        "state_emb", "write_q", "write_out", "gate_proj", "value_head",
        "read_k_projs", "read_v_projs",
        # v5b 遗产配置
        "write_iterations", "n_state", "state_dim",
    ]:
        assert not hasattr(iface, attr), f"v5c 应已删除字段 {attr}"
    # 删除的 API
    for api in [
        "core_parameters", "projection_parameters", "slot_attn_parameters",
        "freeze_core", "unfreeze_core", "core_state_dict",
        "generate_read_kv",  # v5c 不再预生成 K/V
    ]:
        assert not hasattr(iface, api), f"v5c 应已删除 API {api}()"


def test_state_stats(interface):
    """get_state_stats 返回 v5c 字段"""
    W = interface.blank_state(1)
    stats = interface.get_state_stats(W)
    assert "read_scale" in stats
    assert "W_norm" in stats
    assert "W_effective_rank" in stats
    assert stats["W_norm"] == 0.0              # 零初始化
    # 空 W 的有效秩应该是 0（所有奇异值为 0，分布退化）
    assert stats["W_effective_rank"] >= 0.0


def test_param_count_budget():
    """v5c state plugin 参数量随 head_dim 缩放。
    hidden=1024, n_heads=16, n_layers=6:
      head_dim=64: q/o/k/v 投影都是 1024→1024，约 14.7M
      head_dim=128: 投影变 1024→2048，约 28M"""
    iface64 = StateInterface(
        hidden_size=1024, n_heads=16, head_dim=64, n_layers=6,
    )
    total64 = sum(p.numel() for p in iface64.parameters())
    assert 13_000_000 < total64 < 17_000_000, \
        f"head_dim=64 参数量超预算，实际 {total64:,}"

    iface128 = StateInterface(
        hidden_size=1024, n_heads=16, head_dim=128, n_layers=6,
    )
    total128 = sum(p.numel() for p in iface128.parameters())
    assert 25_000_000 < total128 < 32_000_000, \
        f"head_dim=128 参数量超预算，实际 {total128:,}"


def test_delta_parallel_matches_loop():
    """chunkwise 并行版与顺序循环数值等价（差异 < 1e-4，浮点误差范围内）"""
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, H, T, d_k, d_v = 2, 4, 32, 16, 16
    W0 = torch.zeros(B, H, d_v, d_k, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, d_k, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, d_v, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

    W_loop = StateInterface._delta_loop(W0.clone(), k, v, beta, T)
    W_par = StateInterface._delta_parallel(W0.clone(), k, v, beta)
    diff = (W_loop - W_par).abs().max().item()
    assert diff < 1e-8, f"parallel vs loop 数值不一致: max|diff|={diff}"


def test_delta_parallel_gradient_flow():
    """并行版梯度能回传到 k/v/beta/W_0"""
    B, H, T, d_k, d_v = 1, 2, 8, 4, 4
    W0 = torch.zeros(B, H, d_v, d_k, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, d_k, requires_grad=True), dim=-1)
    v = torch.randn(B, H, T, d_v, requires_grad=True)
    beta = torch.sigmoid(torch.randn(B, H, T, requires_grad=True))
    W_new = StateInterface._delta_parallel(W0, k, v, beta)
    W_new.sum().backward()
    assert W0.grad is not None


def test_write_then_read_end_to_end():
    """端到端：write 后用不同 content 触发 read，验证形状/非 NaN"""
    iface = StateInterface(hidden_size=32, n_heads=4, head_dim=8, n_layers=2)
    W = iface.blank_state(2)
    content = torch.randn(2, 16, 32)
    W_new = iface.write_from_content(W, content)
    assert not torch.isnan(W_new).any(), "写入后 W 不应含 NaN"

    hidden = torch.randn(2, 8, 32)
    out = iface.read_layer(hidden, W_new, layer_idx=0)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any(), "读后 hidden 不应含 NaN"
