"""Hippo NM inner SGD operator -- fused forward + backward.

Replaces `xinhe/model/neural_memory.py` `per_sample_grad_fn = vmap(grad(...))`.
At config BHN=96, C=128, D=64, DH=128 vmap+grad expands into ~1000+ small CUDA
kernels with GPU util ~14%; this implementation collapses inner SGD into a
single Triton kernel.

See docs/inner_sgd_triton_math.md for math equivalence and the forward /
inner-grad / outer-bwd derivations. The file provides two paths, both routed
through `HippoInnerSGD.apply`:

  * PyTorch reference (`_pytorch_inner_sgd_fwd / _pytorch_inner_sgd_bwd`):
    bmm/einsum along BHN expressing forward and 2nd-order backward. Used as
    ground-truth in tests and as a fallback when Triton is not available.

  * Triton kernel (`_triton_inner_sgd_fwd / _triton_inner_sgd_bwd`):
    1 program per (b, h, n) sample, activations SRAM-resident, fp32 reduction.
    Production path; ~5-10x faster than vmap+grad.

`HippoInnerSGD` wraps both via `torch.autograd.Function`; outer Adam reads
`(dK, dV, dlr, dgamma, dW1, dW2)` through its backward, exactly like the
vmap+grad path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ----------------------------------------------------------------------------
# GeLU exact (F.gelu(approximate='none')) closed-form derivatives
# ----------------------------------------------------------------------------

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _gelu_phi(x: Tensor) -> Tensor:
    """Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2))


def _gelu_pdf(x: Tensor) -> Tensor:
    """phi(x) = exp(-x^2 / 2) / sqrt(2 * pi)"""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _gelu_prime(x: Tensor) -> Tensor:
    """G'(x) = Phi(x) + x * phi(x), where G(x) = x * Phi(x).

    Matches F.gelu(approximate='none') and torch.autograd backward exactly.
    """
    return _gelu_phi(x) + x * _gelu_pdf(x)


def _gelu_pprime(x: Tensor) -> Tensor:
    """G''(x) = (2 - x^2) * phi(x)"""
    return (2.0 - x * x) * _gelu_pdf(x)


# ----------------------------------------------------------------------------
# PyTorch reference: bmm/einsum along BHN expressing fwd + 2nd-order bwd.
#
# - `_pytorch_inner_sgd_fwd`: returns (grad_gamma, grad_W1, grad_W2, L_c) +
#   saved tensors.
# - `_pytorch_inner_sgd_bwd`: returns (dK, dV, dlr, dgamma, dW1, dW2) given
#   saved tensors + cotangents.
#
# 2nd-order backward derivations: docs/inner_sgd_triton_math.md section 3.
# Each R# comment refers to a node in the reverse chain in that doc.
# ----------------------------------------------------------------------------


def _pytorch_inner_sgd_fwd(
    K: Tensor,       # (BHN, C, D)
    V: Tensor,       # (BHN, C, D)
    lr: Tensor,      # (BHN, C)
    gamma: Tensor,   # (BHN, D)
    W1: Tensor,      # (BHN, D, DH)
    W2: Tensor,      # (BHN, DH, D)
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    """Inner SGD per-sample gradient for ResidualNorm(MemoryMLP(depth=2)).

    All reductions accumulate in fp32 (safe for bf16/fp16 callers).
    """
    # Run reduction chain in fp32; inputs/outputs follow caller dtype (may be bf16).
    out_dtype = K.dtype
    K_f = K.float()
    V_f = V.float()
    lr_f = lr.float()
    g_f = gamma.float()
    W1_f = W1.float()
    W2_f = W2.float()

    BHN, C, D = K.shape
    DH = W1.shape[-1]

    # Forward
    h_pre = torch.bmm(K_f, W1_f)                         # (BHN, C, DH)
    h = F.gelu(h_pre, approximate="none")                # (BHN, C, DH)
    raw = torch.bmm(h, W2_f)                             # (BHN, C, D)
    mu = raw.mean(dim=-1, keepdim=True)                  # (BHN, C, 1)
    var = ((raw - mu) ** 2).mean(dim=-1, keepdim=True)   # (BHN, C, 1)
    sigma = (var + eps).sqrt()                           # (BHN, C, 1)
    ln = (raw - mu) / sigma                              # (BHN, C, D)
    pred = ln * (g_f.unsqueeze(1) + 1.0) + K_f           # (BHN, C, D)
    r = pred - V_f                                       # (BHN, C, D)
    L_c = (r * r).mean(dim=-1)                           # (BHN, C)  unweighted MSE per token

    # Inner gradients (forward outputs)
    err = (2.0 / D) * lr_f.unsqueeze(-1) * r             # (BHN, C, D)
    g_gamma = (err * ln).sum(dim=1)                       # (BHN, D)
    g_ln = err * (g_f.unsqueeze(1) + 1.0)                # (BHN, C, D)
    m_g = g_ln.mean(dim=-1, keepdim=True)                # (BHN, C, 1)
    m_gln = (g_ln * ln).mean(dim=-1, keepdim=True)       # (BHN, C, 1)
    g_raw = (g_ln - m_g - ln * m_gln) / sigma            # (BHN, C, D)
    g_W2 = torch.bmm(h.transpose(-2, -1), g_raw)         # (BHN, DH, D)
    g_h = torch.bmm(g_raw, W2_f.transpose(-2, -1))       # (BHN, C, DH)
    g_h_pre = g_h * _gelu_prime(h_pre)                   # (BHN, C, DH)
    g_W1 = torch.bmm(K_f.transpose(-2, -1), g_h_pre)     # (BHN, D, DH)

    # Saved for backward (minimum set; rest is recomputed in bwd)
    saved = {
        "K": K_f, "V": V_f, "lr": lr_f, "gamma": g_f, "W1": W1_f, "W2": W2_f,
        "h_pre": h_pre, "ln": ln, "sigma": sigma,
        "eps": eps,
    }

    return (
        g_gamma.to(out_dtype),
        g_W1.to(out_dtype),
        g_W2.to(out_dtype),
        L_c.to(out_dtype),
        saved,
    )


def _pytorch_inner_sgd_bwd(
    saved: dict,
    d_g_gamma: Tensor,   # (BHN, D)
    d_g_W1: Tensor,       # (BHN, D, DH)
    d_g_W2: Tensor,       # (BHN, DH, D)
    d_L_c: Tensor,        # (BHN, C)
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Outer backward -- 2nd-order VJP. Output order matches HippoInnerSGD.forward
    inputs: (dK, dV, dlr, dgamma, dW1, dW2)."""
    K = saved["K"]
    V = saved["V"]
    lr = saved["lr"]
    gamma = saved["gamma"]
    W1 = saved["W1"]
    W2 = saved["W2"]
    h_pre = saved["h_pre"]
    ln = saved["ln"]
    sigma = saved["sigma"]

    BHN, C, D = K.shape
    DH = W1.shape[-1]
    invD = 1.0 / D

    # All-fp32 (saved tensors already .float() in fwd)
    d_g_gamma_f = d_g_gamma.float()
    d_g_W1_f = d_g_W1.float()
    d_g_W2_f = d_g_W2.float()
    d_L_c_f = d_L_c.float()

    # Recompute forward intermediates (cheap, code is simpler this way)
    h = F.gelu(h_pre, approximate="none")                 # (BHN, C, DH)
    pred = ln * (gamma.unsqueeze(1) + 1.0) + K            # (BHN, C, D)
    r = pred - V                                           # (BHN, C, D)
    err = (2.0 * invD) * lr.unsqueeze(-1) * r              # (BHN, C, D)
    g_ln = err * (gamma.unsqueeze(1) + 1.0)                # (BHN, C, D)
    m_g = g_ln.mean(dim=-1, keepdim=True)                  # (BHN, C, 1)
    m_gln = (g_ln * ln).mean(dim=-1, keepdim=True)         # (BHN, C, 1)
    g_raw = (g_ln - m_g - ln * m_gln) / sigma              # (BHN, C, D)
    g_h = torch.bmm(g_raw, W2.transpose(-2, -1))           # (BHN, C, DH)
    G_p = _gelu_prime(h_pre)                               # (BHN, C, DH)
    G_pp = _gelu_pprime(h_pre)                             # (BHN, C, DH)
    g_h_pre = g_h * G_p                                    # (BHN, C, DH)

    # R1: grad_gamma -> adj_ln, adj_err
    d_g_g_unsq = d_g_gamma_f.unsqueeze(1)                  # (BHN, 1, D)
    adj_ln = d_g_g_unsq * err                              # (BHN, C, D)
    adj_err = d_g_g_unsq * ln                              # (BHN, C, D)

    # R2: grad_W1 -> adj_K_part1, adj_g_h_pre
    # grad_W1 = K.T @ g_h_pre, dgrad_W1 shape (BHN, D, DH)
    # adj_K_part1 = g_h_pre @ dgrad_W1.T -> (BHN, C, D)
    adj_K_part1 = torch.bmm(g_h_pre, d_g_W1_f.transpose(-2, -1))   # (BHN, C, D)
    adj_g_h_pre = torch.bmm(K, d_g_W1_f)                            # (BHN, C, DH)

    # R3: grad_W2 -> adj_h_part1, adj_g_raw_part1
    adj_h_part1 = torch.bmm(g_raw, d_g_W2_f.transpose(-2, -1))      # (BHN, C, DH)
    adj_g_raw_p1 = torch.bmm(h, d_g_W2_f)                            # (BHN, C, D)

    # R4: L_c -> adj_r partial
    adj_r_part_L = d_L_c_f.unsqueeze(-1) * (2.0 * invD) * r          # (BHN, C, D)

    # R5: g_h_pre = g_h * G'(h_pre)
    adj_g_h = adj_g_h_pre * G_p                                       # (BHN, C, DH)
    adj_h_pre = adj_g_h_pre * g_h * G_pp                              # (BHN, C, DH) - partial

    # R6: g_h = g_raw @ W2.T
    adj_g_raw = adj_g_raw_p1 + torch.bmm(adj_g_h, W2)                 # (BHN, C, D)
    adj_W2_part1 = torch.bmm(adj_g_h.transpose(-2, -1), g_raw)         # (BHN, DH, D)

    # R7: g_raw = (g_ln - m_g - ln * m_gln) / sigma
    adj_g_ln_R7 = adj_g_raw / sigma                                   # (BHN, C, D)
    adj_m_g = -(adj_g_raw.sum(dim=-1, keepdim=True)) / sigma          # (BHN, C, 1)
    adj_ln = adj_ln + (-adj_g_raw * m_gln / sigma)                    # accumulate
    adj_m_gln = -((adj_g_raw * ln).sum(dim=-1, keepdim=True)) / sigma # (BHN, C, 1)
    adj_sigma_p1 = -((adj_g_raw * g_raw).sum(dim=-1, keepdim=True)) / sigma  # (BHN, C, 1)

    # R8: m_gln = mean_d(g_ln * ln)
    adj_g_ln_R8 = adj_m_gln * ln * invD                                # (BHN, C, D)
    adj_ln = adj_ln + adj_m_gln * g_ln * invD                          # accumulate

    # R9: m_g = mean_d(g_ln)
    adj_g_ln_R9 = adj_m_g * invD                                       # (BHN, C, 1) -> broadcast to (C, D)

    adj_g_ln = adj_g_ln_R7 + adj_g_ln_R8 + adj_g_ln_R9                  # (BHN, C, D)

    # R10: g_ln = err * (gamma + 1)
    adj_err = adj_err + adj_g_ln * (gamma.unsqueeze(1) + 1.0)           # accumulate
    adj_gamma_R10 = (adj_g_ln * err).sum(dim=1)                          # (BHN, D)

    # R11: err = (2/D) * lr * r
    adj_lr = (2.0 * invD) * (adj_err * r).sum(dim=-1)                    # (BHN, C)
    adj_r_R11 = (2.0 * invD) * lr.unsqueeze(-1) * adj_err                # (BHN, C, D)

    adj_r = adj_r_part_L + adj_r_R11                                      # (BHN, C, D)

    # R12: r = pred - V
    adj_pred = adj_r                                                      # (BHN, C, D)
    adj_V = -adj_r                                                         # (BHN, C, D)  -> dV

    # R13: pred = ln * (gamma + 1) + K
    adj_ln = adj_ln + adj_pred * (gamma.unsqueeze(1) + 1.0)                # accumulate
    adj_gamma_R13 = (adj_pred * ln).sum(dim=1)                              # (BHN, D)
    adj_K_R13 = adj_pred                                                    # (BHN, C, D)

    adj_gamma = adj_gamma_R10 + adj_gamma_R13                                # (BHN, D)  -> dgamma

    # R14: ln = (raw - mu) / sigma
    adj_raw = adj_ln / sigma                                                 # (BHN, C, D)
    adj_mu = -(adj_ln.sum(dim=-1, keepdim=True)) / sigma                     # (BHN, C, 1)
    adj_sigma_p2 = -((adj_ln * ln).sum(dim=-1, keepdim=True)) / sigma         # (BHN, C, 1)

    adj_sigma = adj_sigma_p1 + adj_sigma_p2                                   # (BHN, C, 1)

    # R15: sigma = sqrt(var + eps)
    adj_var = adj_sigma / (2.0 * sigma)                                       # (BHN, C, 1)

    # R16: var = mean_d((raw - mu)^2);  d var / d mu = 0
    adj_raw = adj_raw + adj_var * (2.0 * invD) * sigma * ln                   # broadcast

    # R17: mu = mean_d(raw)
    adj_raw = adj_raw + adj_mu * invD                                          # broadcast

    # R18: raw = h @ W2
    adj_h_R18 = torch.bmm(adj_raw, W2.transpose(-2, -1))                       # (BHN, C, DH)
    adj_W2_R18 = torch.bmm(h.transpose(-2, -1), adj_raw)                       # (BHN, DH, D)

    adj_h = adj_h_part1 + adj_h_R18                                             # (BHN, C, DH)
    adj_W2 = adj_W2_part1 + adj_W2_R18                                          # (BHN, DH, D)  -> dW2

    # R19: h = gelu(h_pre)
    adj_h_pre = adj_h_pre + adj_h * G_p                                         # accumulate

    # R20: h_pre = K @ W1
    adj_K_R20 = torch.bmm(adj_h_pre, W1.transpose(-2, -1))                       # (BHN, C, D)
    adj_W1 = torch.bmm(K.transpose(-2, -1), adj_h_pre)                           # (BHN, D, DH)  -> dW1

    # Aggregate dK
    dK = adj_K_part1 + adj_K_R13 + adj_K_R20                                     # (BHN, C, D)

    return (
        dK.to(K.dtype),
        adj_V.to(V.dtype),
        adj_lr.to(lr.dtype),
        adj_gamma.to(gamma.dtype),
        adj_W1.to(W1.dtype),
        adj_W2.to(W2.dtype),
    )


# ----------------------------------------------------------------------------
# Triton kernel (SRAM-resident, 1 program / sample) is in
# _triton_inner_sgd_kernels.py. Falls back to PyTorch ref if Triton unavailable.
# ----------------------------------------------------------------------------

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _triton_available() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


# Placeholder; the real kernel lives in _triton_inner_sgd_kernels.py.
_triton_inner_sgd_fwd = None  # type: ignore[assignment]


def _maybe_load_triton_kernels() -> bool:
    """Lazy-import Triton kernels; returns False if unavailable / compile fails.

    Note: only loads the fwd kernel. Backward stays on the PyTorch bmm chain
    (`_pytorch_inner_sgd_bwd`) -- the 30+ adjoint tensors per tile do not fit
    in sm_120's 100KB SRAM even after tiling and trimming persistent state to
    768B (still ~118KB live). The bmm fallback is ~10 kernel launches vs ~1000
    for vmap+grad, so we still get the bulk of the speedup.
    """
    global _triton_inner_sgd_fwd
    if _triton_inner_sgd_fwd is not None:
        return True
    if not _triton_available():
        return False
    try:
        from . import _triton_inner_sgd_kernels as _kernels
        _triton_inner_sgd_fwd = _kernels.triton_inner_sgd_fwd
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# autograd.Function entry point
# ----------------------------------------------------------------------------


class HippoInnerSGD(torch.autograd.Function):
    """Inner SGD per-sample gradient operator for ResidualNorm(MemoryMLP(depth=2)).

    Inputs (all leading dim BHN):
        K     (BHN, C, D)
        V     (BHN, C, D)
        lr    (BHN, C)
        gamma (BHN, D)
        W1    (BHN, D, DH)
        W2    (BHN, DH, D)

    Outputs:
        g_gamma (BHN, D)
        g_W1    (BHN, D, DH)
        g_W2    (BHN, DH, D)
        L_c     (BHN, C)   unweighted per-token MSE -- equals the vmap+grad aux loss.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        K: Tensor,
        V: Tensor,
        lr: Tensor,
        gamma: Tensor,
        W1: Tensor,
        W2: Tensor,
        eps: float = 1e-5,
        force_pytorch: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        use_triton = (not force_pytorch) and K.is_cuda and _maybe_load_triton_kernels()
        if use_triton:
            g_gamma, g_W1, g_W2, L_c, saved = _triton_inner_sgd_fwd(
                K, V, lr, gamma, W1, W2, eps,
            )
            ctx.use_triton = True
        else:
            g_gamma, g_W1, g_W2, L_c, saved = _pytorch_inner_sgd_fwd(
                K, V, lr, gamma, W1, W2, eps,
            )
            ctx.use_triton = False

        # ctx.save_for_backward only accepts Tensors; the scalar (eps) is stored separately.
        ctx.eps = eps
        ctx.saved_keys = list(saved.keys())
        ctx.save_for_backward(*[saved[k] for k in ctx.saved_keys if torch.is_tensor(saved[k])])
        # Cast bwd outputs to the original input dtype.
        ctx.in_dtypes = (K.dtype, V.dtype, lr.dtype, gamma.dtype, W1.dtype, W2.dtype)
        return g_gamma, g_W1, g_W2, L_c

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        d_g_gamma: Tensor,
        d_g_W1: Tensor,
        d_g_W2: Tensor,
        d_L_c: Tensor,
    ) -> tuple:
        # Reconstruct saved dict.
        saved_tensors = ctx.saved_tensors
        keys = [k for k in ctx.saved_keys if k != "eps"]
        saved = dict(zip(keys, saved_tensors))
        saved["eps"] = ctx.eps

        # Backward always runs the PyTorch bmm chain: Triton bwd doesn't fit
        # sm_120 SRAM (see _maybe_load_triton_kernels docstring). Forward already
        # cast saved tensors to fp32 so the bmm chain consumes them directly.
        # ~10 kernel launches vs ~1000 for vmap+grad.
        dK, dV, dlr, dgamma, dW1, dW2 = _pytorch_inner_sgd_bwd(
            saved, d_g_gamma, d_g_W1, d_g_W2, d_L_c,
        )

        # forward signature: (K, V, lr, gamma, W1, W2, eps, force_pytorch).
        # backward must match its arity (eps/force_pytorch are non-differentiable -> None).
        return dK, dV, dlr, dgamma, dW1, dW2, None, None


def hippo_inner_sgd(
    K: Tensor,
    V: Tensor,
    lr: Tensor,
    gamma: Tensor,
    W1: Tensor,
    W2: Tensor,
    eps: float = 1e-5,
    force_pytorch: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convenience wrapper, equivalent to `HippoInnerSGD.apply(...)`."""
    return HippoInnerSGD.apply(K, V, lr, gamma, W1, W2, eps, force_pytorch)


# ----------------------------------------------------------------------------
# Stage 2: Fused chunk-update -- inner SGD + momentum scan + decay scan.
#
# Replaces the PyTorch path at NM `store_memories` lines 806-849 (two AssocScan
# calls per param; even at n=1, AssocScan internals expand into ~5-10 ops).
#
# Math (per param, sequential along inner-chunk dim n):
#   surprise[n] = -grad_W L(K[n], V[n], lr[n], W_init, gamma_init)  # HippoInnerSGD
#   m[n]        = adaptive_momentum[n] * m[n-1] + surprise[n]
#   W[n+1]      = (1 - decay_factor[n]) * W[n] + m[n]              # remove_prev=False:
#                                                                  # entry 0 = prev_W (last_update),
#                                                                  # entry n+1 = scan_output[n]
#
# Note: inner SGD references W_init / gamma_init (NM input weights), not the
# accumulated W. This is the paper-faithful TTT design -- all surprises in a
# batch share the same init, then momentum + decay accumulate the update.
#
# Implementation strategy:
#   - forward uses direct PyTorch ops (replaces assoc_scan, saves ~6 ops *
#     3 params * 2 scans).
#   - Calls HippoInnerSGD.apply internally; Triton fwd available, bwd handled
#     via recomputation autograd.
#   - HippoChunkUpdate.forward runs the PyTorch ref under no_grad (skips the
#     fwd autograd-graph build); backward recomputes the forward with autograd
#     enabled and runs autograd.backward (recomputation pattern).
#   - SRAM does not fit a fully fused Triton kernel for production config
#     (D=64, DH=128) -- m + WD + W_init weights are ~192KB, > sm_120 100KB.
#     We rely on HippoInnerSGD's Triton fwd + PyTorch elementwise ops.
# ----------------------------------------------------------------------------


def _pytorch_chunk_update_fwd_autograd(
    K: Tensor,                  # (BHO, T, D)  T = num_inner * c_inner; BHO = B*H
    V: Tensor,                  # (BHO, T, D)
    lr: Tensor,                 # (BHO, T)
    gamma_init: Tensor,         # (BHO, D)
    W1_init: Tensor,            # (BHO, D, DH)
    W2_init: Tensor,            # (BHO, DH, D)
    adaptive_momentum: Tensor,  # (BHO, num_inner)  per-inner-chunk momentum factor
    decay_factor: Tensor,       # (BHO, num_inner)  per-inner-chunk weight decay
    prev_m_gamma: Tensor,       # (BHO, D)
    prev_m_W1: Tensor,          # (BHO, D, DH)
    prev_m_W2: Tensor,          # (BHO, DH, D)
    prev_W_gamma: Tensor,       # (BHO, D)  past_last_update gamma (prev for weight decay scan)
    prev_W_W1: Tensor,          # (BHO, D, DH)
    prev_W_W2: Tensor,          # (BHO, DH, D)
    chunk_size_inner: int,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fully-autograd PyTorch implementation: HippoInnerSGD.apply + direct PyTorch ops.

    Outputs:
        updates_gamma : (BHO, num_inner+1, D)
        updates_W1    : (BHO, num_inner+1, D, DH)
        updates_W2    : (BHO, num_inner+1, DH, D)
        next_m_gamma  : (BHO, D)
        next_m_W1     : (BHO, D, DH)
        next_m_W2     : (BHO, DH, D)
        L_c           : (BHO, T)  unweighted MSE per token
    """
    BHO, T, D = K.shape
    DH = W1_init.shape[-1]
    num_inner = T // chunk_size_inner
    assert num_inner * chunk_size_inner == T, (
        f"T={T} must be divisible by chunk_size_inner={chunk_size_inner}"
    )

    # Reshape to (BHO * num_inner, c_inner, *) for HippoInnerSGD.apply.
    K_flat = K.view(BHO, num_inner, chunk_size_inner, D).reshape(BHO * num_inner, chunk_size_inner, D)
    V_flat = V.view(BHO, num_inner, chunk_size_inner, D).reshape(BHO * num_inner, chunk_size_inner, D)
    lr_flat = lr.view(BHO, num_inner, chunk_size_inner).reshape(BHO * num_inner, chunk_size_inner)

    # Repeat init weights: (BHO, *) -> (BHO * num_inner, *). .expand uses stride trick (no copy).
    gamma_rep = gamma_init.unsqueeze(1).expand(BHO, num_inner, D).reshape(BHO * num_inner, D).contiguous()
    W1_rep = W1_init.unsqueeze(1).expand(BHO, num_inner, D, DH).reshape(BHO * num_inner, D, DH).contiguous()
    W2_rep = W2_init.unsqueeze(1).expand(BHO, num_inner, DH, D).reshape(BHO * num_inner, DH, D).contiguous()

    # Inner SGD: returns per-param grads + unweighted MSE per token.
    g_gamma_flat, g_W1_flat, g_W2_flat, L_c_flat = HippoInnerSGD.apply(
        K_flat, V_flat, lr_flat, gamma_rep, W1_rep, W2_rep, eps,
    )

    # Reshape back to (BHO, num_inner, *param)
    g_gamma = g_gamma_flat.view(BHO, num_inner, D)
    g_W1 = g_W1_flat.view(BHO, num_inner, D, DH)
    g_W2 = g_W2_flat.view(BHO, num_inner, DH, D)
    L_c = L_c_flat.view(BHO, T)

    # surprise = -grad
    s_gamma = -g_gamma
    s_W1 = -g_W1
    s_W2 = -g_W2

    # Momentum scan (sequential along num_inner; single step at n=1).
    m_gamma_cur = prev_m_gamma
    m_W1_cur = prev_m_W1
    m_W2_cur = prev_m_W2
    m_gamma_list = []
    m_W1_list = []
    m_W2_list = []
    for n in range(num_inner):
        am_n = adaptive_momentum[:, n:n+1]                          # (BHO, 1)
        am_n_3d = am_n.unsqueeze(-1)                                 # (BHO, 1, 1)
        m_gamma_cur = am_n * m_gamma_cur + s_gamma[:, n]
        m_W1_cur = am_n_3d * m_W1_cur + s_W1[:, n]
        m_W2_cur = am_n_3d * m_W2_cur + s_W2[:, n]
        m_gamma_list.append(m_gamma_cur)
        m_W1_list.append(m_W1_cur)
        m_W2_list.append(m_W2_cur)

    next_m_gamma = m_gamma_cur
    next_m_W1 = m_W1_cur
    next_m_W2 = m_W2_cur

    # Weight-decay scan with remove_prev=False: entry 0 = prev_W, then scan outputs.
    WD_gamma_cur = prev_W_gamma
    WD_W1_cur = prev_W_W1
    WD_W2_cur = prev_W_W2
    upd_gamma_list = [WD_gamma_cur]
    upd_W1_list = [WD_W1_cur]
    upd_W2_list = [WD_W2_cur]
    for n in range(num_inner):
        gate_n = 1.0 - decay_factor[:, n:n+1]                       # (BHO, 1)
        gate_n_3d = gate_n.unsqueeze(-1)                             # (BHO, 1, 1)
        WD_gamma_cur = gate_n * WD_gamma_cur + m_gamma_list[n]
        WD_W1_cur = gate_n_3d * WD_W1_cur + m_W1_list[n]
        WD_W2_cur = gate_n_3d * WD_W2_cur + m_W2_list[n]
        upd_gamma_list.append(WD_gamma_cur)
        upd_W1_list.append(WD_W1_cur)
        upd_W2_list.append(WD_W2_cur)

    updates_gamma = torch.stack(upd_gamma_list, dim=1)               # (BHO, num_inner+1, D)
    updates_W1 = torch.stack(upd_W1_list, dim=1)
    updates_W2 = torch.stack(upd_W2_list, dim=1)

    return updates_gamma, updates_W1, updates_W2, next_m_gamma, next_m_W1, next_m_W2, L_c


class HippoChunkUpdate(torch.autograd.Function):
    """Hippo NM single outer-chunk fused chunk-update operator.

    Wraps inner SGD + momentum scan + weight-decay scan to replace the
    per-param assoc_scan double pass at NM `store_memories` lines 806-849.

    Forward runs the PyTorch ref under no_grad to skip the redundant
    autograd-graph build. Backward re-runs the forward on a grad-required
    copy (recomputation pattern) so the inner HippoInnerSGD's manual bwd
    chains naturally with autograd.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        K: Tensor,
        V: Tensor,
        lr: Tensor,
        gamma: Tensor,
        W1: Tensor,
        W2: Tensor,
        adaptive_momentum: Tensor,
        decay_factor: Tensor,
        prev_m_gamma: Tensor,
        prev_m_W1: Tensor,
        prev_m_W2: Tensor,
        prev_W_gamma: Tensor,
        prev_W_W1: Tensor,
        prev_W_W2: Tensor,
        chunk_size_inner: int,
        eps: float = 1e-5,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            outputs = _pytorch_chunk_update_fwd_autograd(
                K, V, lr, gamma, W1, W2,
                adaptive_momentum, decay_factor,
                prev_m_gamma, prev_m_W1, prev_m_W2,
                prev_W_gamma, prev_W_W1, prev_W_W2,
                chunk_size_inner, eps,
            )

        ctx.save_for_backward(
            K, V, lr, gamma, W1, W2,
            adaptive_momentum, decay_factor,
            prev_m_gamma, prev_m_W1, prev_m_W2,
            prev_W_gamma, prev_W_W1, prev_W_W2,
        )
        ctx.chunk_size_inner = chunk_size_inner
        ctx.eps = eps
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs) -> tuple:  # type: ignore[override]
        saved = ctx.saved_tensors
        chunk_size_inner = ctx.chunk_size_inner
        eps = ctx.eps

        # Re-create grad-required copies (detach + clone + requires_grad_).
        inputs_grad = [t.detach().clone().requires_grad_(t.is_floating_point()) for t in saved]

        # autograd.Function.backward defaults to grad disabled; recomputation
        # needs enable_grad so HippoInnerSGD.apply records its own backward.
        with torch.enable_grad():
            outputs = _pytorch_chunk_update_fwd_autograd(
                *inputs_grad, chunk_size_inner, eps,
            )
            torch.autograd.backward(outputs, grad_outputs)

        # Collect grads in input order; non-floating inputs (scalars) -> None.
        grads = tuple(t.grad for t in inputs_grad)

        # forward signature:
        #   K, V, lr, gamma, W1, W2,
        #   adaptive_momentum, decay_factor,
        #   prev_m_gamma, prev_m_W1, prev_m_W2,
        #   prev_W_gamma, prev_W_W1, prev_W_W2,
        #   chunk_size_inner, eps
        return (*grads, None, None)


def hippo_chunk_update(
    K: Tensor,
    V: Tensor,
    lr: Tensor,
    gamma: Tensor,
    W1: Tensor,
    W2: Tensor,
    adaptive_momentum: Tensor,
    decay_factor: Tensor,
    prev_m_gamma: Tensor,
    prev_m_W1: Tensor,
    prev_m_W2: Tensor,
    prev_W_gamma: Tensor,
    prev_W_W1: Tensor,
    prev_W_W2: Tensor,
    chunk_size_inner: int,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convenience wrapper, equivalent to `HippoChunkUpdate.apply(...)`."""
    return HippoChunkUpdate.apply(
        K, V, lr, gamma, W1, W2,
        adaptive_momentum, decay_factor,
        prev_m_gamma, prev_m_W1, prev_m_W2,
        prev_W_gamma, prev_W_W1, prev_W_W2,
        chunk_size_inner, eps,
    )
