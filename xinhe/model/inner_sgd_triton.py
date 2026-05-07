"""Hippo NM inner SGD 算子化 — fused forward + backward。

替换 `xinhe/model/neural_memory.py` 的 `per_sample_grad_fn = vmap(grad(forward_and_loss))`
路径(配置 BHN≈96, C=128, D=64, DH=128 时,vmap 展开成 ~1000+ small kernel,GPU util 14%)。

数学等价性、forward/inner-grad/outer-bwd 公式见 `docs/inner_sgd_triton_math.md`。
本文件提供两条等价路径,由 caller 通过 `HippoInnerSGD.apply` 透明选择:

  * **PyTorch 参考实现**(`_pytorch_inner_sgd_fwd / _pytorch_inner_sgd_bwd`):
    用 bmm/einsum 沿 BHN batch dim 表达完整 forward + 二阶 backward。**等价于 vmap+grad,
    但仅作为 ground truth 测试参考 + Triton 不可用时的 fallback**(慢但正确)。

  * **Triton kernel**(`_triton_inner_sgd_fwd / _triton_inner_sgd_bwd`):
    1 program / 1 (b,h,n) sample,activation SRAM 内驻留,fp32 累加。**生产路径**,
    速度 ≈ vmap+grad 的 5-10×。

`HippoInnerSGD` 封装为 `torch.autograd.Function`,outer Adam 通过它的 backward 拿到
`dK / dV / dlr / dγ / dW₁ / dW₂`,与 vmap+grad 路径完全等价。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# GeLU exact (F.gelu(approximate='none')) 闭式导数
# ─────────────────────────────────────────────────────────────────────────────

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _gelu_phi(x: Tensor) -> Tensor:
    """Φ(x) = 0.5·(1 + erf(x/√2))"""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2))


def _gelu_pdf(x: Tensor) -> Tensor:
    """φ(x) = exp(-x²/2)/√(2π)"""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _gelu_prime(x: Tensor) -> Tensor:
    """G'(x) = Φ(x) + x·φ(x),其中 G(x) = x·Φ(x)。

    与 F.gelu(approximate='none') 严格一致,与 torch.autograd 反向一致。
    """
    return _gelu_phi(x) + x * _gelu_pdf(x)


def _gelu_pprime(x: Tensor) -> Tensor:
    """G''(x) = (2 - x²)·φ(x)"""
    return (2.0 - x * x) * _gelu_pdf(x)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch 参考实现:用 bmm/einsum 沿 BHN batch dim 表达完整 forward + 二阶 backward
#
# - `_pytorch_inner_sgd_fwd`:返回 (∇γ, ∇W₁, ∇W₂, L_c) + saved tensors
# - `_pytorch_inner_sgd_bwd`:从 saved tensors + cotangents 返还
#                           (dK, dV, dlr, dγ, dW₁, dW₂)
#
# 二阶 backward 公式见 docs/inner_sgd_triton_math.md 第 3 节。每条 R# 注释
# 对应 doc 中的反向链节点。
# ─────────────────────────────────────────────────────────────────────────────


def _pytorch_inner_sgd_fwd(
    K: Tensor,       # (BHN, C, D)
    V: Tensor,       # (BHN, C, D)
    lr: Tensor,      # (BHN, C)
    gamma: Tensor,   # (BHN, D)
    W1: Tensor,      # (BHN, D, DH)
    W2: Tensor,      # (BHN, DH, D)
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    """ResidualNorm(MemoryMLP(depth=2)) 的 inner SGD per-sample 梯度。

    所有 reduction 用 fp32 累加(bf16/fp16 caller 安全)。
    """
    # 用 fp32 跑 reduction 链;输入输出 dtype 跟 caller(可能 bf16)
    out_dtype = K.dtype
    K_f = K.float()
    V_f = V.float()
    lr_f = lr.float()
    g_f = gamma.float()
    W1_f = W1.float()
    W2_f = W2.float()

    BHN, C, D = K.shape
    DH = W1.shape[-1]

    # ── Forward
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

    # ── Inner gradients(forward 输出)
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

    # ── Saved for backward(尽量少存,其余在 bwd 重算)
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
    """outer 反传 — 二阶 VJP。输出顺序与 HippoInnerSGD.forward 入参对齐:
    `(dK, dV, dlr, dγ, dW₁, dW₂)`。
    """
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

    # 全程 fp32(saved 张量在 fwd 已经 .float())
    d_g_gamma_f = d_g_gamma.float()
    d_g_W1_f = d_g_W1.float()
    d_g_W2_f = d_g_W2.float()
    d_L_c_f = d_L_c.float()

    # ── 重算 inner forward 的中间量(成本可控,代码简单)
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

    # ── R1: ∇γ → adj_ln, adj_err
    d_g_g_unsq = d_g_gamma_f.unsqueeze(1)                  # (BHN, 1, D)
    adj_ln = d_g_g_unsq * err                              # (BHN, C, D)
    adj_err = d_g_g_unsq * ln                              # (BHN, C, D)

    # ── R2: ∇W₁ → adj_K_part1, adj_g_h_pre
    # ∇W₁ = K.T @ g_h_pre,d∇W₁ shape (BHN, D, DH)
    # adj_K_part1 = g_h_pre @ d∇W₁.T  → (BHN, C, D)
    adj_K_part1 = torch.bmm(g_h_pre, d_g_W1_f.transpose(-2, -1))   # (BHN, C, D)
    adj_g_h_pre = torch.bmm(K, d_g_W1_f)                            # (BHN, C, DH)

    # ── R3: ∇W₂ → adj_h_part1, adj_g_raw_part1
    adj_h_part1 = torch.bmm(g_raw, d_g_W2_f.transpose(-2, -1))      # (BHN, C, DH)
    adj_g_raw_p1 = torch.bmm(h, d_g_W2_f)                            # (BHN, C, D)

    # ── R4: L_c → adj_r partial
    adj_r_part_L = d_L_c_f.unsqueeze(-1) * (2.0 * invD) * r          # (BHN, C, D)

    # ── R5: g_h_pre = g_h · G'(h_pre)
    adj_g_h = adj_g_h_pre * G_p                                       # (BHN, C, DH)
    adj_h_pre = adj_g_h_pre * g_h * G_pp                              # (BHN, C, DH) - partial

    # ── R6: g_h = g_raw @ W₂.T
    adj_g_raw = adj_g_raw_p1 + torch.bmm(adj_g_h, W2)                 # (BHN, C, D)
    adj_W2_part1 = torch.bmm(adj_g_h.transpose(-2, -1), g_raw)         # (BHN, DH, D)

    # ── R7: g_raw = (g_ln - m_g - ln · m_gln) / σ
    adj_g_ln_R7 = adj_g_raw / sigma                                   # (BHN, C, D)
    adj_m_g = -(adj_g_raw.sum(dim=-1, keepdim=True)) / sigma          # (BHN, C, 1)
    adj_ln = adj_ln + (-adj_g_raw * m_gln / sigma)                    # accumulate
    adj_m_gln = -((adj_g_raw * ln).sum(dim=-1, keepdim=True)) / sigma # (BHN, C, 1)
    adj_sigma_p1 = -((adj_g_raw * g_raw).sum(dim=-1, keepdim=True)) / sigma  # (BHN, C, 1)

    # ── R8: m_gln = mean_d(g_ln · ln)
    adj_g_ln_R8 = adj_m_gln * ln * invD                                # (BHN, C, D)
    adj_ln = adj_ln + adj_m_gln * g_ln * invD                          # accumulate

    # ── R9: m_g = mean_d(g_ln)
    adj_g_ln_R9 = adj_m_g * invD                                       # (BHN, C, 1) → broadcast to (C, D)

    adj_g_ln = adj_g_ln_R7 + adj_g_ln_R8 + adj_g_ln_R9                  # (BHN, C, D)

    # ── R10: g_ln = err · (γ + 1)
    adj_err = adj_err + adj_g_ln * (gamma.unsqueeze(1) + 1.0)           # accumulate
    adj_gamma_R10 = (adj_g_ln * err).sum(dim=1)                          # (BHN, D)

    # ── R11: err = (2/D) · lr · r
    adj_lr = (2.0 * invD) * (adj_err * r).sum(dim=-1)                    # (BHN, C)
    adj_r_R11 = (2.0 * invD) * lr.unsqueeze(-1) * adj_err                # (BHN, C, D)

    adj_r = adj_r_part_L + adj_r_R11                                      # (BHN, C, D)

    # ── R12: r = pred - V
    adj_pred = adj_r                                                      # (BHN, C, D)
    adj_V = -adj_r                                                         # (BHN, C, D)  → dV

    # ── R13: pred = ln · (γ + 1) + K
    adj_ln = adj_ln + adj_pred * (gamma.unsqueeze(1) + 1.0)                # accumulate
    adj_gamma_R13 = (adj_pred * ln).sum(dim=1)                              # (BHN, D)
    adj_K_R13 = adj_pred                                                    # (BHN, C, D)

    adj_gamma = adj_gamma_R10 + adj_gamma_R13                                # (BHN, D)  → dγ

    # ── R14: ln = (raw - μ) / σ
    adj_raw = adj_ln / sigma                                                 # (BHN, C, D)
    adj_mu = -(adj_ln.sum(dim=-1, keepdim=True)) / sigma                     # (BHN, C, 1)
    adj_sigma_p2 = -((adj_ln * ln).sum(dim=-1, keepdim=True)) / sigma         # (BHN, C, 1)

    adj_sigma = adj_sigma_p1 + adj_sigma_p2                                   # (BHN, C, 1)

    # ── R15: σ = √(var + eps)
    adj_var = adj_sigma / (2.0 * sigma)                                       # (BHN, C, 1)

    # ── R16: var = mean_d((raw - μ)²),∂var/∂μ = 0
    adj_raw = adj_raw + adj_var * (2.0 * invD) * sigma * ln                   # broadcast

    # ── R17: μ = mean_d(raw)
    adj_raw = adj_raw + adj_mu * invD                                          # broadcast

    # ── R18: raw = h @ W₂
    adj_h_R18 = torch.bmm(adj_raw, W2.transpose(-2, -1))                       # (BHN, C, DH)
    adj_W2_R18 = torch.bmm(h.transpose(-2, -1), adj_raw)                       # (BHN, DH, D)

    adj_h = adj_h_part1 + adj_h_R18                                             # (BHN, C, DH)
    adj_W2 = adj_W2_part1 + adj_W2_R18                                          # (BHN, DH, D)  → dW₂

    # ── R19: h = gelu(h_pre)
    adj_h_pre = adj_h_pre + adj_h * G_p                                         # accumulate

    # ── R20: h_pre = K @ W₁
    adj_K_R20 = torch.bmm(adj_h_pre, W1.transpose(-2, -1))                       # (BHN, C, D)
    adj_W1 = torch.bmm(K.transpose(-2, -1), adj_h_pre)                           # (BHN, D, DH)  → dW₁

    # ── 汇总 dK
    dK = adj_K_part1 + adj_K_R13 + adj_K_R20                                     # (BHN, C, D)

    return (
        dK.to(K.dtype),
        adj_V.to(V.dtype),
        adj_lr.to(lr.dtype),
        adj_gamma.to(gamma.dtype),
        adj_W1.to(W1.dtype),
        adj_W2.to(W2.dtype),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel(SRAM-resident,1 program / sample)— 由 _triton_inner_sgd.py
# 提供;若 triton 不可用则透明 fallback PyTorch 参考。
# ─────────────────────────────────────────────────────────────────────────────

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _triton_available() -> bool:
    return _HAS_TRITON and torch.cuda.is_available()


# 占位:实际 Triton 实现在 _triton_inner_sgd_kernels.py
_triton_inner_sgd_fwd = None  # type: ignore[assignment]


def _maybe_load_triton_kernels() -> bool:
    """lazy import triton kernels;若失败(triton unavailable / 编译失败)返回 False。

    注:只导入 fwd kernel。bwd 保持 PyTorch bmm 链(`_pytorch_inner_sgd_bwd`)—
    bwd 的 30+ adjoint 张量在 sm_120 100KB SRAM 限里装不进单 program,即便 tile
    化 + 持久量降到 768B 仍 118KB 超限。bwd 用 bmm 链是 ~10 kernel,数学等价,
    比 vmap+grad 的 ~1000 kernel 仍大幅减少。
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


# ─────────────────────────────────────────────────────────────────────────────
# autograd.Function 入口
# ─────────────────────────────────────────────────────────────────────────────


class HippoInnerSGD(torch.autograd.Function):
    """ResidualNorm(MemoryMLP(depth=2)) 的 inner SGD per-sample 梯度算子。

    输入(全部 leading dim BHN):
        K     (BHN, C, D)
        V     (BHN, C, D)
        lr    (BHN, C)
        gamma (BHN, D)
        W1    (BHN, D, DH)
        W2    (BHN, DH, D)

    输出:
        g_gamma (BHN, D)
        g_W1    (BHN, D, DH)
        g_W2    (BHN, DH, D)
        L_c     (BHN, C)   unweighted per-token MSE,等价于 vmap+grad 的 aux loss
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

        # ctx.save_for_backward 只接受 Tensor;dict 里的标量(eps)单独存
        ctx.eps = eps
        ctx.saved_keys = list(saved.keys())
        ctx.save_for_backward(*[saved[k] for k in ctx.saved_keys if torch.is_tensor(saved[k])])
        # K/V/lr/gamma/W1/W2 的 dtype 用于 bwd 输出 cast
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
        # 重建 saved dict
        saved_tensors = ctx.saved_tensors
        keys = [k for k in ctx.saved_keys if k != "eps"]
        saved = dict(zip(keys, saved_tensors))
        saved["eps"] = ctx.eps

        # bwd 永走 PyTorch bmm 链:Triton bwd 装不下 sm_120 SRAM(见
        # _maybe_load_triton_kernels docstring)。fwd 已经把 saved 全转 fp32,
        # bmm 链直接消费。这里 ~10 kernel 启动,远好于 vmap+grad 的 ~1000。
        dK, dV, dlr, dgamma, dW1, dW2 = _pytorch_inner_sgd_bwd(
            saved, d_g_gamma, d_g_W1, d_g_W2, d_L_c,
        )

        # forward 接受 (K, V, lr, gamma, W1, W2, eps, force_pytorch);
        # backward 必须返回与之相同长度的 tuple(eps/force_pytorch 不可导 → None)
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
    """便利 wrapper,等价 `HippoInnerSGD.apply(...)`。"""
    return HippoInnerSGD.apply(K, V, lr, gamma, W1, W2, eps, force_pytorch)
