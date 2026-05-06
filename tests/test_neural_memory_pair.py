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
    assert new_state.neo is not None
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
    """plastic=False 时,跨多次 forward state 的 weights 完全不变(等于入口 state)。"""
    pair = _build_pair(phase="Operational")
    pair.set_daytime_plastic(hippo=False, neo=False)
    pair.eval()
    x = torch.randn(2, 16, 32)

    state0 = LayerMemState(None, None)
    _, state1, _ = pair(x, layer_state=state0)
    # 入口 state 是 None 的话 forward 内部 lazy init,但因 plastic=False,丢弃 new state
    # → state1.hippo / state1.neo 应该都是 None(等于入口)
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

    # 关键参数都应有 grad
    assert pair.gate_q.weight.grad is not None
    assert pair.alpha_logit.grad is not None
    # NeuralMemory 内部静态参数(随便取一个)
    assert any(p.grad is not None for p in pair.hippocampus.parameters() if p.requires_grad)
    assert any(p.grad is not None for p in pair.neocortex.parameters() if p.requires_grad)


def test_to_decay_factor_frozen():
    """retention 静态化 → to_decay_factor 不被 backprop 学走。"""
    pair = _build_pair()
    for p in pair.hippocampus.to_decay_factor.parameters():
        assert not p.requires_grad
    for p in pair.neocortex.to_decay_factor.parameters():
        assert not p.requires_grad


# ── retention bias 验证 ──────────────────────────────────────────


def test_retention_init_decay_bias_set():
    """init_decay_bias = logit(1 - retention) → sigmoid(bias) = 1 - retention"""
    pair = _build_pair(hippo_retention=0.99, neo_retention=1.0)
    h_bias = pair.hippocampus.to_decay_factor[0].bias.detach()
    # 静态 retention=0.99 → decay = 0.01 → sigmoid(bias) = 0.01 → bias = logit(0.01)
    expected = _logit(0.01)
    assert torch.allclose(h_bias, torch.full_like(h_bias, expected), atol=1e-5)

    n_bias = pair.neocortex.to_decay_factor[0].bias.detach()
    # retention=1.0 → decay = 0.0 → sigmoid(bias) ≈ 0 → bias = logit(eps) ≈ -13.8
    expected_neo = _logit(0.0)  # 用 eps clamp
    assert torch.allclose(n_bias, torch.full_like(n_bias, expected_neo), atol=1e-5)


# ── helpers ─────────────────────────────────────────────────────


def test_logit_clamps_extremes():
    assert _logit(0.0) < -10  # 应该是 logit(eps) 不报错
    assert _logit(1.0) > 10
    assert abs(_logit(0.5) - 0.0) < 1e-6
