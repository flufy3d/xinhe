"""
测试 Hippocampus (v7: 大一统短期记忆 W + per-head γ 内容 gating)
"""
import inspect
import torch
import pytest
import torch.nn.functional as F

from xinhe.model.hippocampus import Hippocampus


@pytest.fixture
def interface():
    return Hippocampus(hidden_size=64, n_heads=4, head_dim=16, n_layers=4)


# ═══════════════════════════════════════════════════════════════════
# 基础 Delta Rule 读写（v5c 继承，仍需成立）
# ═══════════════════════════════════════════════════════════════════


def test_blank_state_shape(interface):
    """空白 W 形状正确: (B, H, d_v, d_k)"""
    W = interface.blank_state(batch_size=2)
    assert W.shape == (2, 4, 16, 16)
    assert (W == 0).all(), "零初始化: 空 W 读出恒为 0"


def test_blank_state_read_is_identity():
    """零 W 时 read_layer 输出 == 输入"""
    iface = Hippocampus(hidden_size=32, n_heads=4, head_dim=8, n_layers=2)
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
    """初始 read_scale 接近 0"""
    iface = Hippocampus(
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
    """同 key 先写 v1 再写 v2，读回偏向 v2"""
    B, H, d_k, d_v = 1, 1, 8, 8
    W = torch.zeros(B, H, d_v, d_k)
    k = F.normalize(torch.randn(B, H, d_k), dim=-1)
    v1 = torch.randn(B, H, d_v)
    v2 = torch.randn(B, H, d_v)
    beta = 0.8

    v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
    W = W + beta * torch.einsum("bhv,bhd->bhvd", v1 - v_hat, k)
    v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
    W = W + beta * torch.einsum("bhv,bhd->bhvd", v2 - v_hat, k)

    read = torch.einsum("bhvd,bhd->bhv", W, k)
    dist_to_v2 = (read - v2).norm()
    dist_to_v1 = (read - v1).norm()
    assert dist_to_v2 < dist_to_v1


def test_delta_rule_subtract_interference():
    """正交 k1/k2 分别写 v1/v2，读 k1 应接近 v1"""
    B, H, d_k, d_v = 1, 1, 8, 8
    W = torch.zeros(B, H, d_v, d_k)
    k1 = torch.zeros(B, H, d_k); k1[0, 0, 0] = 1.0
    k2 = torch.zeros(B, H, d_k); k2[0, 0, 1] = 1.0
    v1 = torch.randn(B, H, d_v)
    v2 = torch.randn(B, H, d_v)
    beta = 1.0

    for k, v in [(k1, v1), (k2, v2)]:
        v_hat = torch.einsum("bhvd,bhd->bhv", W, k)
        W = W + beta * torch.einsum("bhv,bhd->bhvd", v - v_hat, k)

    read = torch.einsum("bhvd,bhd->bhv", W, k1)
    assert torch.allclose(read, v1, atol=1e-5)


def test_no_softmax_in_write():
    """静态断言 write_from_content 里没有 softmax（Delta Rule 零 softmax 约束）"""
    src = inspect.getsource(Hippocampus.write_from_content)
    assert "softmax" not in src.lower()


def test_gradient_flow():
    """梯度能通过 read + write 流回各个投影"""
    iface = Hippocampus(hidden_size=32, n_heads=2, head_dim=8, n_layers=2)
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
    """n_heads * head_dim 可以不等于 hidden_size"""
    iface = Hippocampus(hidden_size=64, n_heads=8, head_dim=32, n_layers=2)
    W = iface.blank_state(2)
    assert W.shape == (2, 8, 32, 32)
    hidden = torch.randn(2, 16, 64)
    out = iface.read_layer(hidden, W, 0)
    assert out.shape == (2, 16, 64)
    content = torch.randn(2, 16, 64)
    W_new = iface.write_from_content(W, content)
    assert W_new.shape == (2, 8, 32, 32)


def test_state_stats(interface):
    """get_state_stats 返回新字段(γ 已删)"""
    W = interface.blank_state(1)
    stats = interface.get_state_stats(W)
    assert "read_scale" in stats
    assert "W_norm" in stats
    assert "W_effective_rank" in stats
    assert stats["W_norm"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# 纯 Delta Rule (无 γ)
# ═══════════════════════════════════════════════════════════════════


def test_delta_parallel_matches_sequential_no_gamma():
    """chunkwise 并行版与顺序循环在无 γ 衰减下数值等价"""
    torch.manual_seed(0)
    B, H, T, d_k, d_v = 2, 4, 16, 8, 8
    W0 = torch.zeros(B, H, d_v, d_k, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, d_k, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, d_v, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

    # 手写原始 Delta Rule (无 γ) 作参考
    W_ref = W0.clone()
    for t in range(T):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        b_t = beta[:, :, t, None, None]
        v_hat = torch.einsum("bhvd,bhd->bhv", W_ref, k_t)
        W_ref = W_ref + b_t * torch.einsum("bhv,bhd->bhvd", v_t - v_hat, k_t)

    W_par = Hippocampus._delta_parallel(W0.clone(), k, v, beta)
    diff = (W_ref - W_par).abs().max().item()
    assert diff < 1e-8, f"parallel vs sequential 不一致: max|diff|={diff}"


def test_delta_parallel_gradient_flow():
    """并行版梯度能回传到 k/v/beta"""
    B, H, T, d_k, d_v = 1, 2, 6, 4, 4
    W0 = torch.zeros(B, H, d_v, d_k, requires_grad=True)
    k = F.normalize(torch.randn(B, H, T, d_k, requires_grad=True), dim=-1)
    v = torch.randn(B, H, T, d_v, requires_grad=True)
    beta = torch.sigmoid(torch.randn(B, H, T, requires_grad=True))
    W_new = Hippocampus._delta_parallel(W0, k, v, beta)
    W_new.sum().backward()
    assert W0.grad is not None


def test_write_then_read_end_to_end():
    """端到端：write 后用不同 content 触发 read"""
    iface = Hippocampus(hidden_size=32, n_heads=4, head_dim=8, n_layers=2)
    W = iface.blank_state(2)
    content = torch.randn(2, 16, 32)
    W_new = iface.write_from_content(W, content)
    assert not torch.isnan(W_new).any()

    hidden = torch.randn(2, 8, 32)
    out = iface.read_layer(hidden, W_new, layer_idx=0)
    assert out.shape == (2, 8, 32)
    assert not torch.isnan(out).any()


def test_param_count_budget():
    """Hippocampus 参数量(γ 已删)"""
    iface = Hippocampus(
        hidden_size=1024, n_heads=16, head_dim=64, n_layers=6,
    )
    total = sum(p.numel() for p in iface.parameters())
    assert 13_000_000 < total < 17_000_000, \
        f"参数量超预算,实际 {total:,}"
