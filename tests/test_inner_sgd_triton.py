"""HippoInnerSGD 数值等价测试 — Triton fwd + PyTorch bwd 对照 vmap+grad ground truth。

公式 / 推导见 `docs/inner_sgd_triton_math.md`。
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap


# CUDA 必须;Triton 需要 GPU。CPU 无意义,单文件 skip。
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HippoInnerSGD 测试需要 CUDA"
)


# triton-windows 需要 TRITON_CACHE_DIR(避开 Windows 260 字符路径限);
# conftest 的 setdefault 也会兜底,这里再保守一次。
import os
os.environ.setdefault("TRITON_CACHE_DIR", "C:\\tc")


def _vmap_grad_reference(K, V, lr, gamma, W1, W2, eps=1e-5):
    """与 NeuralMemory.per_sample_grad_fn 等价的 ground truth。

    `forward_and_loss = mse(ResidualNorm(MemoryMLP)(K), V) * lr`,
    `vmap(grad)` 沿 BHN 0 维。
    """

    def forward_and_loss(params, inputs, loss_weights, target):
        gamma_p, W1_p, W2_p = params["gamma"], params["W1"], params["W2"]
        h_pre = inputs @ W1_p
        h = F.gelu(h_pre, approximate="none")
        raw = h @ W2_p
        mu = raw.mean(dim=-1, keepdim=True)
        var = ((raw - mu) ** 2).mean(dim=-1, keepdim=True)
        sigma = (var + eps).sqrt()
        ln = (raw - mu) / sigma
        norm_out = ln * (gamma_p + 1.0)
        pred = norm_out + inputs
        loss_per_token = (pred - target).pow(2).mean(dim=-1)
        weighted = loss_per_token * loss_weights
        return weighted.sum(), loss_per_token

    grad_fn = grad(forward_and_loss, has_aux=True)
    per_sample = vmap(grad_fn, in_dims=(0, 0, 0, 0))
    params = {"gamma": gamma, "W1": W1, "W2": W2}
    grads, L_c = per_sample(params, K, lr, V)
    return grads["gamma"], grads["W1"], grads["W2"], L_c


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    abs_err = diff.max().item()
    denom = b.float().abs().clamp_min(1e-8)
    rel_err = (diff / denom).max().item()
    return abs_err, rel_err


def _mk_inputs(BHN, C, D, DH, dtype, device, requires_grad=False, seed=42):
    torch.manual_seed(seed)
    K = torch.randn(BHN, C, D, device=device, dtype=dtype, requires_grad=requires_grad)
    V = torch.randn(BHN, C, D, device=device, dtype=dtype, requires_grad=requires_grad)
    lr = (torch.rand(BHN, C, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    gamma = (torch.randn(BHN, D, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    W1 = (torch.randn(BHN, D, DH, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    W2 = (torch.randn(BHN, DH, D, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    return K, V, lr, gamma, W1, W2


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch reference (bmm chain) 数值等价 vmap+grad
# 容差:fp32 max_abs < 5e-7(reduction 顺序差 fp32 噪声底)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "BHN,C,D,DH",
    [
        (4, 16, 16, 32),     # 小 shape sanity
        (8, 32, 32, 64),     # 中等
        (96, 128, 64, 128),  # 真实 production 配置
    ],
)
def test_pytorch_ref_forward_matches_vmap_grad(BHN, C, D, DH):
    from xinhe.model.inner_sgd_triton import _pytorch_inner_sgd_fwd

    K, V, lr, gamma, W1, W2 = _mk_inputs(BHN, C, D, DH, torch.float32, "cuda")

    ref_gg, ref_gW1, ref_gW2, ref_Lc = _vmap_grad_reference(K, V, lr, gamma, W1, W2)
    gg, gW1, gW2, Lc, _ = _pytorch_inner_sgd_fwd(K, V, lr, gamma, W1, W2)

    for name, ref, ours in [
        ("g_gamma", ref_gg, gg),
        ("g_W1", ref_gW1, gW1),
        ("g_W2", ref_gW2, gW2),
        ("L_c", ref_Lc, Lc),
    ]:
        ae, _ = _max_abs_rel(ours, ref)
        assert ae < 5e-7, f"{name}: max_abs={ae:.2e} too large"


@pytest.mark.parametrize(
    "BHN,C,D,DH",
    [
        (4, 16, 16, 32),
        (8, 32, 32, 64),
        (96, 128, 64, 128),
    ],
)
def test_pytorch_ref_outer_grad_matches_autograd(BHN, C, D, DH):
    """Outer backward 一致性:dummy outer loss <output, cot> 的 dK/dV/.../dW2"""
    from xinhe.model.inner_sgd_triton import _pytorch_inner_sgd_fwd, _pytorch_inner_sgd_bwd

    K, V, lr, gamma, W1, W2 = _mk_inputs(BHN, C, D, DH, torch.float32, "cuda", requires_grad=True, seed=7)
    dgg = torch.randn(BHN, D, device="cuda", dtype=torch.float32) * 0.1
    dgW1 = torch.randn(BHN, D, DH, device="cuda", dtype=torch.float32) * 0.1
    dgW2 = torch.randn(BHN, DH, D, device="cuda", dtype=torch.float32) * 0.1
    dLc = torch.randn(BHN, C, device="cuda", dtype=torch.float32) * 0.1

    # 参考路径
    ref_gg, ref_gW1, ref_gW2, ref_Lc = _vmap_grad_reference(K, V, lr, gamma, W1, W2)
    L_outer = (
        (ref_gg * dgg).sum()
        + (ref_gW1 * dgW1).sum()
        + (ref_gW2 * dgW2).sum()
        + (ref_Lc * dLc).sum()
    )
    ref_dK, ref_dV, ref_dlr, ref_dG, ref_dW1, ref_dW2 = torch.autograd.grad(
        L_outer, [K, V, lr, gamma, W1, W2]
    )

    # PyTorch fwd + bwd
    K_d = K.detach().clone()
    V_d = V.detach().clone()
    lr_d = lr.detach().clone()
    gamma_d = gamma.detach().clone()
    W1_d = W1.detach().clone()
    W2_d = W2.detach().clone()
    _, _, _, _, saved = _pytorch_inner_sgd_fwd(K_d, V_d, lr_d, gamma_d, W1_d, W2_d)
    dK, dV, dlr, dG, dW1, dW2 = _pytorch_inner_sgd_bwd(saved, dgg, dgW1, dgW2, dLc)

    for name, ref, ours in [
        ("dK", ref_dK, dK),
        ("dV", ref_dV, dV),
        ("dlr", ref_dlr, dlr),
        ("dgamma", ref_dG, dG),
        ("dW1", ref_dW1, dW1),
        ("dW2", ref_dW2, dW2),
    ]:
        ae, _ = _max_abs_rel(ours, ref)
        # 二阶 backward 涉及更多 reduction,容差略松
        tol = 2e-6 if name in ("dlr",) else 5e-7
        assert ae < tol, f"{name}: max_abs={ae:.2e} > {tol:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Triton fwd 数值等价 vmap+grad
# ─────────────────────────────────────────────────────────────────────────────


pytest.importorskip("triton")


@pytest.mark.parametrize(
    "BHN,C,D,DH",
    [
        (8, 16, 16, 32),
        (96, 128, 64, 128),
    ],
)
def test_triton_forward_matches_vmap_grad(BHN, C, D, DH):
    from xinhe.model._triton_inner_sgd_kernels import triton_inner_sgd_fwd

    K, V, lr, gamma, W1, W2 = _mk_inputs(BHN, C, D, DH, torch.float32, "cuda")

    ref_gg, ref_gW1, ref_gW2, ref_Lc = _vmap_grad_reference(K, V, lr, gamma, W1, W2)
    gg, gW1, gW2, Lc, _ = triton_inner_sgd_fwd(K, V, lr, gamma, W1, W2)

    for name, ref, ours in [
        ("g_gamma", ref_gg, gg),
        ("g_W1", ref_gW1, gW1),
        ("g_W2", ref_gW2, gW2),
        ("L_c", ref_Lc, Lc),
    ]:
        ae, _ = _max_abs_rel(ours, ref)
        # Triton fp32 + IEEE 精度 + tile reduce → 略大 fp32 噪声
        assert ae < 5e-6, f"{name}: max_abs={ae:.2e} too large"


# ─────────────────────────────────────────────────────────────────────────────
# 端到端:HippoInnerSGD.apply (Triton fwd + PyTorch bwd) 全链路
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "BHN,C,D,DH",
    [
        (8, 16, 16, 32),
        (96, 128, 64, 128),
    ],
)
def test_hippo_inner_sgd_e2e(BHN, C, D, DH):
    """forward 走 Triton,backward 走 PyTorch bmm,通过 autograd.Function 串起来。"""
    from xinhe.model.inner_sgd_triton import hippo_inner_sgd

    K, V, lr, gamma, W1, W2 = _mk_inputs(BHN, C, D, DH, torch.float32, "cuda", requires_grad=True, seed=7)
    dgg = torch.randn(BHN, D, device="cuda", dtype=torch.float32) * 0.1
    dgW1 = torch.randn(BHN, D, DH, device="cuda", dtype=torch.float32) * 0.1
    dgW2 = torch.randn(BHN, DH, D, device="cuda", dtype=torch.float32) * 0.1
    dLc = torch.randn(BHN, C, device="cuda", dtype=torch.float32) * 0.1

    # 参考
    ref_gg, ref_gW1, ref_gW2, ref_Lc = _vmap_grad_reference(K, V, lr, gamma, W1, W2)
    L_outer = (
        (ref_gg * dgg).sum()
        + (ref_gW1 * dgW1).sum()
        + (ref_gW2 * dgW2).sum()
        + (ref_Lc * dLc).sum()
    )
    ref_dK, ref_dV, ref_dlr, ref_dG, ref_dW1, ref_dW2 = torch.autograd.grad(
        L_outer, [K, V, lr, gamma, W1, W2]
    )

    # 待测
    K2 = K.detach().clone().requires_grad_(True)
    V2 = V.detach().clone().requires_grad_(True)
    lr2 = lr.detach().clone().requires_grad_(True)
    g2 = gamma.detach().clone().requires_grad_(True)
    W12 = W1.detach().clone().requires_grad_(True)
    W22 = W2.detach().clone().requires_grad_(True)

    gg, gW1, gW2, Lc = hippo_inner_sgd(K2, V2, lr2, g2, W12, W22)
    L2 = (gg * dgg).sum() + (gW1 * dgW1).sum() + (gW2 * dgW2).sum() + (Lc * dLc).sum()
    dK, dV, dlr, dG, dW1, dW2 = torch.autograd.grad(L2, [K2, V2, lr2, g2, W12, W22])

    for name, ref, ours in [
        ("forward g_gamma", ref_gg, gg),
        ("forward g_W1", ref_gW1, gW1),
        ("forward g_W2", ref_gW2, gW2),
        ("forward L_c", ref_Lc, Lc),
        ("dK", ref_dK, dK),
        ("dV", ref_dV, dV),
        ("dlr", ref_dlr, dlr),
        ("dgamma", ref_dG, dG),
        ("dW1", ref_dW1, dW1),
        ("dW2", ref_dW2, dW2),
    ]:
        ae, _ = _max_abs_rel(ours, ref)
        tol = 5e-6 if name.startswith("forward") else 5e-6
        if name == "dlr":
            tol = 5e-6
        assert ae < tol, f"{name}: max_abs={ae:.2e} > {tol:.2e}"


def test_hippo_inner_sgd_force_pytorch():
    """`force_pytorch=True` 强制走 PyTorch fwd(测试 fallback 路径)。"""
    from xinhe.model.inner_sgd_triton import HippoInnerSGD

    BHN, C, D, DH = 4, 16, 16, 32
    K, V, lr, gamma, W1, W2 = _mk_inputs(BHN, C, D, DH, torch.float32, "cuda", requires_grad=True)
    gg, gW1, gW2, Lc = HippoInnerSGD.apply(K, V, lr, gamma, W1, W2, 1e-5, True)
    assert gg.shape == (BHN, D)
    assert gW1.shape == (BHN, D, DH)
    assert gW2.shape == (BHN, DH, D)
    assert Lc.shape == (BHN, C)
    # 反向跑通(force_pytorch 路径下 ctx.use_triton=False)
    L_outer = gg.sum() + gW1.sum() + gW2.sum() + Lc.sum()
    L_outer.backward()
    assert K.grad is not None and W1.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Fallback 路径(NeuralMemory shape guard):chunk_size < 16 → 退回 vmap+grad
# ─────────────────────────────────────────────────────────────────────────────


def test_neural_memory_fast_path_eligibility_flag():
    """ResidualNorm(MemoryMLP(depth=2)) + default MSE 时 flag=True。"""
    from xinhe.model.neural_memory import NeuralMemory

    nm = NeuralMemory(dim=32, dim_head=16, heads=2, chunk_size=4, batch_size=4)
    assert nm._fast_path_eligible is True


def test_neural_memory_fast_path_falls_back_on_small_chunk():
    """chunk_size=4 < 16 → store_memories 走 vmap+grad fallback,行为不变。"""
    from xinhe.model.neural_memory import NeuralMemory

    nm = NeuralMemory(dim=32, dim_head=16, heads=2, chunk_size=4, batch_size=4).cuda()
    seq = torch.randn(2, 16, 32, device="cuda")
    # 不该报错(走 vmap+grad fallback)
    retrieved, state = nm(seq)
    assert retrieved.shape == seq.shape
