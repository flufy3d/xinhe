"""NeuralMemoryPair v9 单元测试。"""
import math

import pytest
import torch

from xinhe.model.neural_memory_pair import (
    LayerMemState,
    NeuralMemoryPair,
    XinheMemoryState,
    _logit,
)


def _build_pair(phase: str = "P-cap", **kwargs):
    """常用尺寸的小型 Pair,B=2 friendly。"""
    defaults = dict(
        d_total=32,
        n_heads=2,
        d_head=16,
        chunk_size=4,
        phase=phase,
    )
    defaults.update(kwargs)
    return NeuralMemoryPair(**defaults)


# ── 构造 / state shape ────────────────────────────────────────────


def test_pair_constructs_with_defaults():
    pair = _build_pair()
    assert pair.d_total == 32
    assert pair.n_heads == 2
    assert pair.d_head == 16


def test_phase_pcap_neo_plastic_default():
    pair = _build_pair(phase="P-cap")
    assert pair.daytime_plastic_hippo is True
    assert pair.daytime_plastic_neo is True


def test_phase_operational_neo_frozen_default():
    pair = _build_pair(phase="Operational")
    assert pair.daytime_plastic_hippo is True
    assert pair.daytime_plastic_neo is False


def test_xinhe_memory_state_init():
    state = XinheMemoryState.init(layer_indices=[3, 7, 11])
    assert set(state.keys()) == {3, 7, 11}
    for s in state.values():
        assert s.hippo is None
        assert s.neo is None


def test_xinhe_memory_state_detach_passthrough():
    """空 state 的 detach 不报错(各层都是 None)。"""
    state = XinheMemoryState.init([3, 7])
    detached = state.detach()
    assert set(detached.keys()) == {3, 7}


# ── forward 形状 ──────────────────────────────────────────────────


def test_forward_shape_preserves_residual():
    pair = _build_pair()
    pair.eval()
    x = torch.randn(2, 32, 32)
    x_out, new_state, aux = pair(x)
    assert x_out.shape == x.shape
    assert new_state.hippo is not None
    # Neo 是无状态普通 MLP,layer state 里 neo 字段恒为 None
    assert new_state.neo is None
    assert "gate_entropy_reg_loss" in aux


def test_forward_with_explicit_layer_state():
    """两次 forward 后 fast weights 演化(看 state.states[0] = past_last_update)。
    NeuralMemory 的 NeuralMemState.weights 是入口 weights 副本,不演化;
    真正演化在 .states (past_last_update, past_last_momentum)。"""
    pair = _build_pair()
    pair.eval()
    x = torch.randn(2, 32, 32)
    _, state1, _ = pair(x)
    x2 = torch.randn(2, 32, 32)
    _, state2, _ = pair(x2, layer_state=state1)
    assert state2.hippo is not None
    s1_lu = next(iter(state1.hippo.states[0].values()))
    s2_lu = next(iter(state2.hippo.states[0].values()))
    assert not torch.allclose(s1_lu, s2_lu)


# ── alpha override / mem_alpha=0 干净路径 ─────────────────────────


def test_mem_alpha_override_zero_yields_pure_residual():
    pair = _build_pair()
    pair.eval()
    x = torch.randn(2, 16, 32)
    x_out, _, _ = pair(x, mem_alpha_override=0.0)
    assert torch.allclose(x_out, x, atol=1e-6)


def test_mem_alpha_override_one_full_memory():
    """alpha=1 时 x_out = x + mem_out,差值 = mem_out 不为 0。"""
    pair = _build_pair()
    pair.eval()
    x = torch.randn(2, 16, 32)
    x_out, _, _ = pair(x, mem_alpha_override=1.0)
    diff = (x_out - x).abs().mean().item()
    assert diff > 1e-4   # mem_out 应有非平凡贡献


# ── daytime_plastic 切换 ─────────────────────────────────────────


def test_daytime_plastic_off_freezes_state():
    """Hippo plastic=False 时,跨多次 forward state 不演化(等于入口 state)。
    Neo 无状态(走标准 backprop),其字段恒为 None。"""
    pair = _build_pair(phase="Operational")
    pair.set_daytime_plastic(hippo=False, neo=False)
    pair.eval()
    x = torch.randn(2, 16, 32)

    state0 = LayerMemState(None, None)
    _, state1, _ = pair(x, layer_state=state0)
    assert state1.hippo is None
    assert state1.neo is None


def test_daytime_plastic_on_updates_state():
    """plastic=True 时 fast weights 演化(看 past_last_update,见 test_forward_with_explicit_layer_state)。"""
    pair = _build_pair()
    pair.set_daytime_plastic(hippo=True, neo=True)
    pair.eval()
    x1 = torch.randn(2, 16, 32)
    x2 = torch.randn(2, 16, 32)

    _, state1, _ = pair(x1)
    _, state2, _ = pair(x2, layer_state=state1)

    s1_lu = next(iter(state1.hippo.states[0].values()))
    s2_lu = next(iter(state2.hippo.states[0].values()))
    assert not torch.allclose(s1_lu, s2_lu)


# ── gate entropy 正则 ────────────────────────────────────────────


def test_gate_entropy_reg_loss_is_nonpositive():
    """λ * (-H) ≤ 0 (因为 H ≥ 0,λ > 0)"""
    pair = _build_pair(gate_entropy_lambda=0.05)
    pair.eval()
    x = torch.randn(2, 16, 32)
    _, _, aux = pair(x)
    assert aux["gate_entropy_reg_loss"].item() <= 1e-9


def test_gate_entropy_in_valid_range():
    """gate 熵 in [0, ln 2] (二选一 softmax)"""
    pair = _build_pair()
    pair.eval()
    x = torch.randn(2, 16, 32)
    _, _, aux = pair(x)
    H = aux["gate_entropy"].item()
    assert 0.0 <= H <= math.log(2) + 1e-6


# ── 梯度流 ──────────────────────────────────────────────────────


def test_gradient_flow_through_pair():
    pair = _build_pair()
    pair.train()
    x = torch.randn(2, 16, 32, requires_grad=True)
    x_out, _, aux = pair(x)
    loss = x_out.pow(2).mean() + aux["gate_entropy_reg_loss"]
    loss.backward()

    # Pair 顶层 gate / alpha
    assert pair.gate_q.weight.grad is not None
    assert pair.alpha_logit.grad is not None
    # Hippo NeuralMemory 内部静态参数(meta-params,gate_q-like)
    assert any(p.grad is not None for p in pair.hippocampus.parameters() if p.requires_grad)
    # Neo 是普通 MLP,所有 weight 都该有 grad(走 backbone backprop)
    for w in pair.neocortex.weights:
        assert w.grad is not None


def test_neocortex_is_static_mlp():
    """Neo 走普通 backprop:
       - 每个 weight 是 (heads, dim_in, dim_out) 形状的 nn.Parameter
       - 跨 forward 不演化(weights identity 一致)
       - 不持有 NeuralMemState
    """
    pair = _build_pair()
    pair.eval()
    # 形状校验:depth=4 exp=4.0(默认)→ dim_head=16 → hidden=64 → 4 个 weight,
    # 中间 3 个 (heads=2, 16/64, 16/64) 这种结构。
    for w in pair.neocortex.weights:
        assert w.ndim == 3
        assert w.shape[0] == pair.n_heads

    x = torch.randn(2, 16, 32)
    w_before = [w.detach().clone() for w in pair.neocortex.weights]
    _ = pair(x)
    _ = pair(x)
    # 多次 forward 后 Neo weights 不应改变(只有 outer optimizer.step() 才会改)
    for w_after, w0 in zip(pair.neocortex.weights, w_before):
        assert torch.equal(w_after.detach(), w0)


def test_hippocampus_to_decay_factor_frozen():
    """Hippo retention 静态化 → to_decay_factor 不被 backprop 学走。"""
    pair = _build_pair()
    for p in pair.hippocampus.to_decay_factor.parameters():
        assert not p.requires_grad


# ── retention bias 验证 ──────────────────────────────────────────


def test_hippo_retention_init_decay_bias_set():
    """Hippo init_decay_bias = logit(1 - retention) → sigmoid(bias) = 1 - retention。
    Neo 无 retention(普通 MLP)。"""
    pair = _build_pair(hippo_retention=0.99)
    h_bias = pair.hippocampus.to_decay_factor[0].bias.detach()
    expected = _logit(0.01)
    assert torch.allclose(h_bias, torch.full_like(h_bias, expected), atol=1e-5)


# ── helpers ─────────────────────────────────────────────────────


def test_logit_clamps_extremes():
    assert _logit(0.0) < -10  # 应该是 logit(eps) 不报错
    assert _logit(1.0) > 10
    assert abs(_logit(0.5) - 0.0) < 1e-6
