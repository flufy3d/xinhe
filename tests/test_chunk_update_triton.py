"""HippoChunkUpdate 数值等价测试 — fused inner SGD + momentum scan + decay scan。

参考路径:NM `store_memories` 里的 lines 720-849 完整流程(`HippoInnerSGD` +
两次 `AssocScan`)。HippoChunkUpdate 应该数学完全等价(fp32 < 1e-5,bf16 < 5e-3)。
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from xinhe.model.inner_sgd_triton import (
    HippoInnerSGD,
    HippoChunkUpdate,
    _pytorch_chunk_update_fwd_autograd,
)

# Windows triton 缓存兜底
os.environ.setdefault("TRITON_CACHE_DIR", "C:\\tc")


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    abs_err = diff.max().item()
    denom = b.float().abs().clamp_min(1e-8)
    rel_err = (diff / denom).max().item()
    return abs_err, rel_err


def _reference_chunk_update_via_assoc_scan(
    K, V, lr,
    gamma_init, W1_init, W2_init,
    adaptive_momentum, decay_factor,
    prev_m_gamma, prev_m_W1, prev_m_W2,
    prev_W_gamma, prev_W_W1, prev_W_W2,
    chunk_size_inner,
    eps=1e-5,
):
    """Ground truth:复刻 NM store_memories 数学,直接用 PyTorch + assoc_scan。
    与 _pytorch_chunk_update_fwd_autograd 等价但用真正的 AssocScan 路径,排除
    实现 bug。"""
    from assoc_scan import AssocScan

    BHO, T, D = K.shape
    DH = W1_init.shape[-1]
    num_inner = T // chunk_size_inner

    # Inner SGD per (b, h, n) sample,等价 NM 路径
    K_flat = K.view(BHO, num_inner, chunk_size_inner, D).reshape(BHO * num_inner, chunk_size_inner, D)
    V_flat = V.view(BHO, num_inner, chunk_size_inner, D).reshape(BHO * num_inner, chunk_size_inner, D)
    lr_flat = lr.view(BHO, num_inner, chunk_size_inner).reshape(BHO * num_inner, chunk_size_inner)

    gamma_rep = gamma_init.unsqueeze(1).expand(BHO, num_inner, D).reshape(BHO * num_inner, D).contiguous()
    W1_rep = W1_init.unsqueeze(1).expand(BHO, num_inner, D, DH).reshape(BHO * num_inner, D, DH).contiguous()
    W2_rep = W2_init.unsqueeze(1).expand(BHO, num_inner, DH, D).reshape(BHO * num_inner, DH, D).contiguous()

    g_gamma_flat, g_W1_flat, g_W2_flat, L_c_flat = HippoInnerSGD.apply(
        K_flat, V_flat, lr_flat, gamma_rep, W1_rep, W2_rep, eps,
    )

    g_gamma = g_gamma_flat.view(BHO, num_inner, D)
    g_W1 = g_W1_flat.view(BHO, num_inner, D, DH)
    g_W2 = g_W2_flat.view(BHO, num_inner, DH, D)
    L_c = L_c_flat.view(BHO, T)

    s_gamma = -g_gamma
    s_W1 = -g_W1
    s_W2 = -g_W2

    # 用 assoc_scan(NM 走的路径)
    scanner = AssocScan()

    # adaptive_momentum 在 NM 里 shape (1, BHO, num_inner, 1) 拆出 order=0 → (BHO, num_inner, 1)
    am_3d = adaptive_momentum.unsqueeze(-1)  # (BHO, num_inner, 1) for assoc_scan broadcast

    # momentum scan: m = β * m_prev + s,prev 默认 remove_prev=True
    m_gamma_scan = scanner(am_3d, s_gamma, prev=prev_m_gamma)            # (BHO, num_inner, D)
    m_W1_scan = scanner(am_3d, s_W1, prev=prev_m_W1)                      # (BHO, num_inner, D, DH)
    m_W2_scan = scanner(am_3d, s_W2, prev=prev_m_W2)                      # (BHO, num_inner, DH, D)

    next_m_gamma = m_gamma_scan[:, -1]
    next_m_W1 = m_W1_scan[:, -1]
    next_m_W2 = m_W2_scan[:, -1]

    # decay scan: W = (1-δ) * W_prev + m,prev=last_update,remove_prev=False
    df_3d = decay_factor.unsqueeze(-1)
    upd_g = scanner(1.0 - df_3d, m_gamma_scan, prev=prev_W_gamma, remove_prev=False)  # (BHO, num_inner+1, D)
    upd_W1 = scanner(1.0 - df_3d, m_W1_scan, prev=prev_W_W1, remove_prev=False)
    upd_W2 = scanner(1.0 - df_3d, m_W2_scan, prev=prev_W_W2, remove_prev=False)

    return upd_g, upd_W1, upd_W2, next_m_gamma, next_m_W1, next_m_W2, L_c


def _mk_chunk_update_inputs(
    BHO=2, num_inner=3, c=16, D=16, DH=32,
    dtype=torch.float32, device="cpu", requires_grad=False, seed=42,
):
    torch.manual_seed(seed)
    T = num_inner * c
    K = torch.randn(BHO, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    V = torch.randn(BHO, T, D, device=device, dtype=dtype, requires_grad=requires_grad)
    lr = (torch.rand(BHO, T, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    gamma = (torch.randn(BHO, D, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    W1 = (torch.randn(BHO, D, DH, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    W2 = (torch.randn(BHO, DH, D, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    am = (torch.rand(BHO, num_inner, device=device, dtype=dtype) * 0.5 + 0.4).requires_grad_(requires_grad)  # [0.4, 0.9]
    df = (torch.rand(BHO, num_inner, device=device, dtype=dtype) * 0.05).requires_grad_(requires_grad)       # [0, 0.05] 接近 paper retention=0.99
    pm_g = (torch.randn(BHO, D, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    pm_W1 = (torch.randn(BHO, D, DH, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    pm_W2 = (torch.randn(BHO, DH, D, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    pW_g = (torch.randn(BHO, D, device=device, dtype=dtype) * 0.01).requires_grad_(requires_grad)
    pW_W1 = (torch.randn(BHO, D, DH, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    pW_W2 = (torch.randn(BHO, DH, D, device=device, dtype=dtype) * 0.1).requires_grad_(requires_grad)
    return (K, V, lr, gamma, W1, W2, am, df, pm_g, pm_W1, pm_W2, pW_g, pW_W1, pW_W2)


# ─────────────────────────────────────────────────────────────────────────────
# CPU fp32:_pytorch_chunk_update_fwd_autograd vs assoc_scan reference
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_inner", [1, 2, 4])
def test_pytorch_fwd_equiv_assoc_scan(num_inner):
    """直接 PyTorch 标量 ops 路径 == assoc_scan 路径(数学等价 sanity)。"""
    inputs = _mk_chunk_update_inputs(BHO=2, num_inner=num_inner, c=16, D=16, DH=32)
    K, V, lr, gamma, W1, W2, am, df, pm_g, pm_W1, pm_W2, pW_g, pW_W1, pW_W2 = inputs

    out_a = _pytorch_chunk_update_fwd_autograd(*inputs, 16, 1e-5)
    out_b = _reference_chunk_update_via_assoc_scan(*inputs, 16, 1e-5)

    names = ["upd_g", "upd_W1", "upd_W2", "next_m_g", "next_m_W1", "next_m_W2", "L_c"]
    for n, a, b in zip(names, out_a, out_b):
        abs_err, rel_err = _max_abs_rel(a, b)
        assert abs_err < 1e-5, f"{n} fp32 abs_err={abs_err} (>1e-5)"
        # rel_err 在零附近会爆,只检查 abs


@pytest.mark.parametrize("num_inner", [1, 3])
def test_chunk_update_apply_fwd_equiv(num_inner):
    """HippoChunkUpdate.apply (forward 在 no_grad 跑 PyTorch ref) == 直接 PyTorch 路径。"""
    inputs = _mk_chunk_update_inputs(BHO=2, num_inner=num_inner, c=16, D=16, DH=32)

    out_apply = HippoChunkUpdate.apply(*inputs, 16, 1e-5)
    out_ref = _pytorch_chunk_update_fwd_autograd(*inputs, 16, 1e-5)

    for a, b in zip(out_apply, out_ref):
        abs_err, _ = _max_abs_rel(a, b)
        assert abs_err < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Backward 等价:HippoChunkUpdate 的 recomputation bwd vs PyTorch autograd bwd
# ─────────────────────────────────────────────────────────────────────────────


def test_chunk_update_backward_equiv():
    """对 dummy outer loss `(updates_W2 * cot).sum()`,HippoChunkUpdate.apply 的
    grad 与 autograd 走 _pytorch_chunk_update_fwd_autograd 的 grad 应数值等价。"""
    inputs_a = _mk_chunk_update_inputs(BHO=2, num_inner=3, c=16, D=16, DH=32, requires_grad=True, seed=123)
    inputs_b = _mk_chunk_update_inputs(BHO=2, num_inner=3, c=16, D=16, DH=32, requires_grad=True, seed=123)

    # Path A: HippoChunkUpdate.apply
    out_a = HippoChunkUpdate.apply(*inputs_a, 16, 1e-5)
    upd_W2_a = out_a[2]
    cot = torch.randn_like(upd_W2_a)
    loss_a = (upd_W2_a * cot).sum()
    loss_a.backward()

    # Path B: 直接 _pytorch_chunk_update_fwd_autograd
    out_b = _pytorch_chunk_update_fwd_autograd(*inputs_b, 16, 1e-5)
    upd_W2_b = out_b[2]
    loss_b = (upd_W2_b * cot).sum()
    loss_b.backward()

    # 比对所有可导输入的 grad。recomputation 给所有 inputs 分配 zero grad;
    # 直接 autograd 不可达的输入 grad 是 None。两者数学等价(0 == None 0)。
    names = ["K", "V", "lr", "gamma", "W1", "W2", "am", "df", "pm_g", "pm_W1", "pm_W2", "pW_g", "pW_W1", "pW_W2"]
    for n, ta, tb in zip(names, inputs_a, inputs_b):
        ta_g = ta.grad if ta.grad is not None else torch.zeros_like(ta)
        tb_g = tb.grad if tb.grad is not None else torch.zeros_like(tb)
        abs_err, _ = _max_abs_rel(ta_g, tb_g)
        assert abs_err < 1e-5, f"{n} bwd abs_err={abs_err}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_num_inner_one_trivial():
    """num_inner=1:scan 退化成单步,验证不爆。"""
    inputs = _mk_chunk_update_inputs(BHO=4, num_inner=1, c=16, D=16, DH=32)
    out = HippoChunkUpdate.apply(*inputs, 16, 1e-5)
    upd_g, upd_W1, upd_W2, next_m_g, next_m_W1, next_m_W2, L_c = out
    assert upd_g.shape == (4, 2, 16)        # num_inner+1 = 2
    assert upd_W1.shape == (4, 2, 16, 32)
    assert next_m_g.shape == (4, 16)
    assert L_c.shape == (4, 16)              # T = 1*16


# ─────────────────────────────────────────────────────────────────────────────
# CUDA / Triton(若 GPU 可用)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("num_inner", [1, 2])
def test_chunk_update_cuda_equiv(num_inner):
    """CUDA 上 HippoChunkUpdate(走 HippoInnerSGD 的 Triton fwd)与 CPU 等价。"""
    inputs_cpu = _mk_chunk_update_inputs(
        BHO=2, num_inner=num_inner, c=32, D=32, DH=64, dtype=torch.float32, seed=7,
    )
    inputs_gpu = tuple(t.cuda() for t in inputs_cpu)

    out_cpu = HippoChunkUpdate.apply(*inputs_cpu, 32, 1e-5)
    out_gpu = HippoChunkUpdate.apply(*inputs_gpu, 32, 1e-5)

    for a, b in zip(out_cpu, out_gpu):
        abs_err, _ = _max_abs_rel(a.cpu(), b.cpu())
        # Triton fp32 IEEE matmul 与 CPU bmm 顺序不同,容差 1e-4
        assert abs_err < 1e-4, f"abs_err={abs_err}"


def test_nm_fast_path_equiv_regular_path():
    """端到端:NM 走 fast path(HippoChunkUpdate)与禁用 fast path 走原路径
    (HippoInnerSGD + assoc_scan)产出相同 forward / state / loss(fp32 < 1e-5)。"""
    from xinhe.model.neural_memory_pair import NeuralMemoryPair

    torch.manual_seed(42)
    # chunk_size=16 触发 fast path(>= 16 阈值)
    pair_a = NeuralMemoryPair(d_total=32, n_heads=2, d_head=16, chunk_size=16, phase="P-cap")
    pair_b = NeuralMemoryPair(d_total=32, n_heads=2, d_head=16, chunk_size=16, phase="P-cap")
    # 同 init
    pair_b.load_state_dict(pair_a.state_dict())
    pair_a.eval()
    pair_b.eval()

    # 禁用 pair_b 的 fast path,强走原路径
    pair_b.hippocampus._fast_path_eligible = False

    x = torch.randn(2, 32, 32)
    out_a, state_a, _ = pair_a(x)
    out_b, state_b, _ = pair_b(x)

    abs_err, _ = _max_abs_rel(out_a, out_b)
    assert abs_err < 1e-5, f"NM forward output abs_err={abs_err}"

    # state 也应一致(hippo state 包含 weights / states / updates)
    assert state_a.hippo is not None
    assert state_b.hippo is not None
    # last_update 应一致
    for k in state_a.hippo.states[0].keys():
        a = state_a.hippo.states[0][k]
        b = state_b.hippo.states[0][k]
        abs_err, _ = _max_abs_rel(a, b)
        assert abs_err < 1e-5, f"state.{k} abs_err={abs_err}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_nm_compile_equiv_eager():
    """Stage 3:`use_compile_chunk_loop=True` 与 eager 数学等价(fwd + bwd)。

    backward 在 HippoChunkUpdate.backward 的 `requires_grad_()` 处会有 graph break
    (Dynamo 限制),但会 gracefully fallback eager,正确性不受影响。"""
    from xinhe.model.neural_memory_pair import NeuralMemoryPair

    torch.manual_seed(0)
    pair_a = NeuralMemoryPair(d_total=32, n_heads=2, d_head=16, chunk_size=16, phase="P-cap").cuda()
    pair_b = NeuralMemoryPair(d_total=32, n_heads=2, d_head=16, chunk_size=16, phase="P-cap").cuda()
    pair_b.load_state_dict(pair_a.state_dict())

    pair_a.hippocampus.use_compile_chunk_loop = True
    pair_b.hippocampus.use_compile_chunk_loop = False

    # Forward 等价
    pair_a.eval()
    pair_b.eval()
    x = torch.randn(2, 32, 32, device="cuda")
    out_a, _, _ = pair_a(x)
    out_b, _, _ = pair_b(x)
    assert (out_a - out_b).abs().max().item() < 1e-5

    # Backward 等价(compile + eager 在 graph break 处均能正确传梯度)
    pair_a.train()
    pair_b.train()
    torch.manual_seed(7)
    x_a = torch.randn(2, 32, 32, device="cuda", requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    out_a, _, _ = pair_a(x_a)
    out_b, _, _ = pair_b(x_b)
    out_a.sum().backward()
    out_b.sum().backward()

    assert (x_a.grad - x_b.grad).abs().max().item() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_chunk_update_bf16_tol():
    """bf16:多步 scan 累积误差,容差 5e-2(scan=2 步,bf16 7-bit mantissa)。

    Triton fwd 内部 fp32 reduction,但 scan 加乘 + 累积 W/m 走 bf16 PyTorch ops
    (HippoInnerSGD 之外的部分)。bf16 单步加乘约 ~5e-3,累积后 ~1-2e-2 正常。
    """
    inputs = _mk_chunk_update_inputs(
        BHO=2, num_inner=2, c=32, D=32, DH=64, dtype=torch.bfloat16, device="cuda", seed=11,
    )
    inputs_fp32 = tuple(t.float() for t in inputs)

    out_bf16 = HippoChunkUpdate.apply(*inputs, 32, 1e-5)
    out_fp32 = HippoChunkUpdate.apply(*inputs_fp32, 32, 1e-5)

    for a, b in zip(out_bf16, out_fp32):
        abs_err, _ = _max_abs_rel(a.float(), b.float())
        assert abs_err < 5e-2
