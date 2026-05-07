"""Triton kernel:Hippo NM inner SGD per-sample 梯度。

公式 / 反向链推导见 docs/inner_sgd_triton_math.md。
本文件提供两个 `@triton.jit` kernel + Python wrapper:

    triton_inner_sgd_fwd(K, V, lr, gamma, W1, W2, eps)
        → (g_gamma, g_W1, g_W2, L_c, saved_dict)

    triton_inner_sgd_bwd(saved_dict, d_g_gamma, d_g_W1, d_g_W2, d_L_c)
        → (dK, dV, dlr, dgamma, dW1, dW2)

设计:
    grid = (BHN,),1 program / 1 (b,h,n) sample
    沿 C 维 tile(BLOCK_C 默认 32),累加器(g_gamma / g_W1 / g_W2)在 SRAM 持久,
    W1/W2/γ 也持久(per-sample 输入)。其余每个 tile 独立。
    内部全 fp32 + `input_precision="ieee"`(关掉 TF32,精度对齐 vmap+grad)
"""
from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# device 函数:GeLU exact + 一阶 / 二阶导
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _gelu_phi(x):
    """Φ(x) = 0.5·(1 + erf(x/√2))"""
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865475))


@triton.jit
def _gelu_pdf(x):
    """φ(x) = exp(-x²/2)/√(2π)"""
    return 0.3989422804014327 * tl.exp(-0.5 * x * x)


@triton.jit
def _gelu_exact(x):
    """G(x) = x · Φ(x)"""
    return x * _gelu_phi(x)


@triton.jit
def _gelu_prime(x):
    """G'(x) = Φ(x) + x · φ(x)"""
    return _gelu_phi(x) + x * _gelu_pdf(x)


@triton.jit
def _gelu_pprime(x):
    """G''(x) = (2 - x²) · φ(x)"""
    return (2.0 - x * x) * _gelu_pdf(x)


# ─────────────────────────────────────────────────────────────────────────────
# Forward kernel
# ─────────────────────────────────────────────────────────────────────────────


@triton.jit
def _hippo_inner_sgd_fwd_kernel(
    K_ptr, V_ptr, lr_ptr, gamma_ptr, W1_ptr, W2_ptr,
    g_gamma_ptr, g_W1_ptr, g_W2_ptr, L_c_ptr,
    h_pre_save_ptr, ln_save_ptr, sigma_save_ptr,
    C: tl.constexpr,
    D: tl.constexpr,
    DH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_D = 1.0 / D

    offs_d = tl.arange(0, D)
    offs_dh = tl.arange(0, DH)
    offs_bc = tl.arange(0, BLOCK_C)

    # per-sample bases
    K_base     = K_ptr     + pid * C * D
    V_base     = V_ptr     + pid * C * D
    lr_base    = lr_ptr    + pid * C
    gamma_base = gamma_ptr + pid * D
    W1_base    = W1_ptr    + pid * D * DH
    W2_base    = W2_ptr    + pid * DH * D
    g_gamma_base    = g_gamma_ptr    + pid * D
    g_W1_base       = g_W1_ptr       + pid * D * DH
    g_W2_base       = g_W2_ptr       + pid * DH * D
    L_c_base        = L_c_ptr        + pid * C
    h_pre_save_base = h_pre_save_ptr + pid * C * DH
    ln_save_base    = ln_save_ptr    + pid * C * D
    sigma_save_base = sigma_save_ptr + pid * C

    # Persistent loads
    gamma = tl.load(gamma_base + offs_d)                                              # (D,)
    W1 = tl.load(W1_base + offs_d[:, None] * DH + offs_dh[None, :])                   # (D, DH)
    W2 = tl.load(W2_base + offs_dh[:, None] * D + offs_d[None, :])                    # (DH, D)
    gp1 = gamma + 1.0                                                                  # (D,)

    # Accumulators
    acc_g_gamma = tl.zeros((D,), dtype=tl.float32)
    acc_g_W1 = tl.zeros((D, DH), dtype=tl.float32)
    acc_g_W2 = tl.zeros((DH, D), dtype=tl.float32)

    # Tile along C
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + offs_bc                                                            # (BLOCK_C,)
        # Loads
        K_t = tl.load(K_base + c_idx[:, None] * D + offs_d[None, :])                    # (BLOCK_C, D)
        V_t = tl.load(V_base + c_idx[:, None] * D + offs_d[None, :])
        lr_t = tl.load(lr_base + c_idx)                                                  # (BLOCK_C,)

        # Forward
        h_pre_t = tl.dot(K_t, W1, out_dtype=tl.float32, input_precision="ieee")         # (BLOCK_C, DH)
        tl.store(h_pre_save_base + c_idx[:, None] * DH + offs_dh[None, :], h_pre_t)
        h_t = _gelu_exact(h_pre_t)
        raw_t = tl.dot(h_t, W2, out_dtype=tl.float32, input_precision="ieee")            # (BLOCK_C, D)

        mu_t = tl.sum(raw_t, axis=1) * inv_D                                              # (BLOCK_C,)
        raw_c = raw_t - mu_t[:, None]                                                      # (BLOCK_C, D)
        var_t = tl.sum(raw_c * raw_c, axis=1) * inv_D                                      # (BLOCK_C,)
        sigma_t = tl.sqrt(var_t + EPS)                                                      # (BLOCK_C,)
        inv_sigma_t = 1.0 / sigma_t                                                         # (BLOCK_C,)
        ln_t = raw_c * inv_sigma_t[:, None]                                                # (BLOCK_C, D)

        tl.store(ln_save_base + c_idx[:, None] * D + offs_d[None, :], ln_t)
        tl.store(sigma_save_base + c_idx, sigma_t)

        pred_t = ln_t * gp1[None, :] + K_t                                                 # (BLOCK_C, D)
        r_t = pred_t - V_t                                                                  # (BLOCK_C, D)
        L_c_t = tl.sum(r_t * r_t, axis=1) * inv_D                                          # (BLOCK_C,)
        tl.store(L_c_base + c_idx, L_c_t)

        err_t = (2.0 * inv_D) * lr_t[:, None] * r_t                                        # (BLOCK_C, D)

        # ∇γ contribution
        acc_g_gamma += tl.sum(err_t * ln_t, axis=0)                                        # (D,)

        # g_raw
        g_ln_t = err_t * gp1[None, :]                                                       # (BLOCK_C, D)
        m_g_t = tl.sum(g_ln_t, axis=1) * inv_D                                              # (BLOCK_C,)
        m_gln_t = tl.sum(g_ln_t * ln_t, axis=1) * inv_D                                     # (BLOCK_C,)
        g_raw_t = (g_ln_t - m_g_t[:, None] - ln_t * m_gln_t[:, None]) * inv_sigma_t[:, None]   # (BLOCK_C, D)

        # ∇W2 += h_t.T @ g_raw_t
        acc_g_W2 += tl.dot(tl.trans(h_t), g_raw_t, out_dtype=tl.float32, input_precision="ieee")

        # g_h = g_raw_t @ W2.T;g_h_pre = g_h * G'(h_pre)
        g_h_t = tl.dot(g_raw_t, tl.trans(W2), out_dtype=tl.float32, input_precision="ieee")    # (BLOCK_C, DH)
        G_p_t = _gelu_prime(h_pre_t)                                                              # (BLOCK_C, DH)
        g_h_pre_t = g_h_t * G_p_t                                                                  # (BLOCK_C, DH)

        # ∇W1 += K_t.T @ g_h_pre_t
        acc_g_W1 += tl.dot(tl.trans(K_t), g_h_pre_t, out_dtype=tl.float32, input_precision="ieee")

    # Final stores
    tl.store(g_gamma_base + offs_d, acc_g_gamma)
    tl.store(g_W1_base + offs_d[:, None] * DH + offs_dh[None, :], acc_g_W1)
    tl.store(g_W2_base + offs_dh[:, None] * D + offs_d[None, :], acc_g_W2)


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel — **当前不在 production 路径**
#
# 状态密度:30+ adjoint tensors per tile,加 W2/d_g_W1/d_g_W2 即使 tile 内载入,
# sm_120 100KB SRAM 仍装不下(实测 production 大小 BHN=96/C=128/D=64/DH=128 需
# ~118KB)。已经把 dW1/dW2/dK_R20 移到 Python 外用 bmm 算,但仍超限。
#
# `HippoInnerSGD.backward` 永走 PyTorch bmm 链(`_pytorch_inner_sgd_bwd`)—
# ~10 kernel 启动,vs vmap+grad 的 ~1000 仍是大幅减少;forward Triton 已捕获
# launch overhead 大头(实测 7.78× 加速)。
#
# kernel 留着供未来分裂多 sub-kernel 重启;小 shape(C ≤ 16)compile OK,可作
# 数值正确性参考。
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _hippo_inner_sgd_bwd_kernel(
    K_ptr, V_ptr, lr_ptr, gamma_ptr, W2_ptr,
    h_pre_ptr, ln_ptr, sigma_ptr,
    d_g_gamma_ptr, d_g_W1_ptr, d_g_W2_ptr, d_L_c_ptr,
    # outputs
    dK_ptr, dV_ptr, dlr_ptr, dgamma_ptr,
    # save for outside-bmm
    adj_h_pre_save_ptr,   # (BHN, C, DH) — 用于 dW1 = K.T @ adj_h_pre
    adj_raw_save_ptr,     # (BHN, C, D)  — 用于 dW2 R18 部分 = h.T @ adj_raw
    adj_g_h_save_ptr,     # (BHN, C, DH) — 用于 dW2 R6 部分  = adj_g_h.T @ g_raw
    g_raw_save_ptr,       # (BHN, C, D)  — 同上
    h_save_ptr,           # (BHN, C, DH) — 同上
    C: tl.constexpr,
    D: tl.constexpr,
    DH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    inv_D = 1.0 / D

    offs_d = tl.arange(0, D)
    offs_dh = tl.arange(0, DH)
    offs_bc = tl.arange(0, BLOCK_C)

    # bases
    K_base = K_ptr + pid * C * D
    V_base = V_ptr + pid * C * D
    lr_base = lr_ptr + pid * C
    gamma_base = gamma_ptr + pid * D
    W2_base = W2_ptr + pid * DH * D
    h_pre_base = h_pre_ptr + pid * C * DH
    ln_base = ln_ptr + pid * C * D
    sigma_base = sigma_ptr + pid * C
    d_g_gamma_base = d_g_gamma_ptr + pid * D
    d_g_W1_base    = d_g_W1_ptr    + pid * D * DH
    d_g_W2_base    = d_g_W2_ptr    + pid * DH * D
    d_L_c_base     = d_L_c_ptr     + pid * C
    dK_base     = dK_ptr     + pid * C * D
    dV_base     = dV_ptr     + pid * C * D
    dlr_base    = dlr_ptr    + pid * C
    dgamma_base = dgamma_ptr + pid * D
    adj_h_pre_save_base = adj_h_pre_save_ptr + pid * C * DH
    adj_raw_save_base   = adj_raw_save_ptr   + pid * C * D
    adj_g_h_save_base   = adj_g_h_save_ptr   + pid * C * DH
    g_raw_save_base     = g_raw_save_ptr     + pid * C * D
    h_save_base         = h_save_ptr         + pid * C * DH

    # 持久量极简:仅 gamma / d_g_gamma / acc_dgamma(总计 < 1KB)
    # W2 / d_g_W1 / d_g_W2 全部移到 tile 内加载 — 每 tile 重复 HBM load,
    # 但 sm_120 SRAM 100KB 装不下大于 64KB 的持久 + 60KB 的 per-tile 工作集
    gamma = tl.load(gamma_base + offs_d)
    gp1 = gamma + 1.0
    d_g_gamma = tl.load(d_g_gamma_base + offs_d)

    acc_dgamma = tl.zeros((D,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + offs_bc

        K_t = tl.load(K_base + c_idx[:, None] * D + offs_d[None, :])
        V_t = tl.load(V_base + c_idx[:, None] * D + offs_d[None, :])
        lr_t = tl.load(lr_base + c_idx)
        h_pre_t = tl.load(h_pre_base + c_idx[:, None] * DH + offs_dh[None, :])
        ln_t = tl.load(ln_base + c_idx[:, None] * D + offs_d[None, :])
        sigma_t = tl.load(sigma_base + c_idx)
        d_L_c_t = tl.load(d_L_c_base + c_idx)
        inv_sigma_t = 1.0 / sigma_t

        # 大 tensor 移到 tile 内 load(节省 persistent SRAM)
        W2 = tl.load(W2_base + offs_dh[:, None] * D + offs_d[None, :])                    # (DH, D)
        d_g_W1 = tl.load(d_g_W1_base + offs_d[:, None] * DH + offs_dh[None, :])           # (D, DH)
        d_g_W2 = tl.load(d_g_W2_base + offs_dh[:, None] * D + offs_d[None, :])            # (DH, D)

        # ── 重算可推量
        h_t = _gelu_exact(h_pre_t)
        G_p_t = _gelu_prime(h_pre_t)
        G_pp_t = _gelu_pprime(h_pre_t)
        pred_t = ln_t * gp1[None, :] + K_t
        r_t = pred_t - V_t
        err_t = (2.0 * inv_D) * lr_t[:, None] * r_t
        g_ln_t = err_t * gp1[None, :]
        m_g_t = tl.sum(g_ln_t, axis=1) * inv_D
        m_gln_t = tl.sum(g_ln_t * ln_t, axis=1) * inv_D
        g_raw_t = (g_ln_t - m_g_t[:, None] - ln_t * m_gln_t[:, None]) * inv_sigma_t[:, None]
        g_h_t = tl.dot(g_raw_t, tl.trans(W2), out_dtype=tl.float32, input_precision="ieee")
        g_h_pre_t = g_h_t * G_p_t

        # ── R1
        adj_ln = d_g_gamma[None, :] * err_t
        adj_err = d_g_gamma[None, :] * ln_t

        # ── R2
        adj_K_p1 = tl.dot(g_h_pre_t, tl.trans(d_g_W1), out_dtype=tl.float32, input_precision="ieee")
        adj_g_h_pre = tl.dot(K_t, d_g_W1, out_dtype=tl.float32, input_precision="ieee")

        # ── R3
        adj_h_p1 = tl.dot(g_raw_t, tl.trans(d_g_W2), out_dtype=tl.float32, input_precision="ieee")
        adj_g_raw_p1 = tl.dot(h_t, d_g_W2, out_dtype=tl.float32, input_precision="ieee")

        # ── R4
        adj_r_part_L = d_L_c_t[:, None] * (2.0 * inv_D) * r_t

        # ── R5
        adj_g_h = adj_g_h_pre * G_p_t
        adj_h_pre = adj_g_h_pre * g_h_t * G_pp_t

        # ── R6
        adj_g_raw = adj_g_raw_p1 + tl.dot(adj_g_h, W2, out_dtype=tl.float32, input_precision="ieee")
        # 把 adj_g_h, g_raw_t 写到 HBM,Python 算 dW2 R6 部分
        tl.store(adj_g_h_save_base + c_idx[:, None] * DH + offs_dh[None, :], adj_g_h)
        tl.store(g_raw_save_base + c_idx[:, None] * D + offs_d[None, :], g_raw_t)

        # ── R7
        adj_g_ln_R7 = adj_g_raw * inv_sigma_t[:, None]
        adj_m_g = -tl.sum(adj_g_raw, axis=1) * inv_sigma_t
        adj_ln += -adj_g_raw * m_gln_t[:, None] * inv_sigma_t[:, None]
        adj_m_gln = -tl.sum(adj_g_raw * ln_t, axis=1) * inv_sigma_t
        adj_sigma_p1 = -tl.sum(adj_g_raw * g_raw_t, axis=1) * inv_sigma_t

        # ── R8, R9
        adj_g_ln_R8 = adj_m_gln[:, None] * ln_t * inv_D
        adj_ln += adj_m_gln[:, None] * g_ln_t * inv_D
        adj_g_ln_R9 = adj_m_g[:, None] * inv_D + 0.0 * ln_t
        adj_g_ln = adj_g_ln_R7 + adj_g_ln_R8 + adj_g_ln_R9

        # ── R10
        adj_err += adj_g_ln * gp1[None, :]
        acc_dgamma += tl.sum(adj_g_ln * err_t, axis=0)

        # ── R11
        adj_lr_t = (2.0 * inv_D) * tl.sum(adj_err * r_t, axis=1)
        adj_r_R11 = (2.0 * inv_D) * lr_t[:, None] * adj_err
        adj_r = adj_r_part_L + adj_r_R11

        # ── R12
        tl.store(dV_base + c_idx[:, None] * D + offs_d[None, :], -adj_r)
        tl.store(dlr_base + c_idx, adj_lr_t)
        adj_pred = adj_r

        # ── R13
        adj_ln += adj_pred * gp1[None, :]
        acc_dgamma += tl.sum(adj_pred * ln_t, axis=0)
        adj_K_R13 = adj_pred

        # ── R14
        adj_raw = adj_ln * inv_sigma_t[:, None]
        adj_mu = -tl.sum(adj_ln, axis=1) * inv_sigma_t
        adj_sigma_p2 = -tl.sum(adj_ln * ln_t, axis=1) * inv_sigma_t

        adj_sigma = adj_sigma_p1 + adj_sigma_p2

        # ── R15
        adj_var = adj_sigma * 0.5 * inv_sigma_t

        # ── R16, R17
        adj_raw += adj_var[:, None] * (2.0 * inv_D) * sigma_t[:, None] * ln_t
        adj_raw += adj_mu[:, None] * inv_D + 0.0 * ln_t

        # 把 adj_raw, h_t 写出 HBM,Python 算 dW2 R18 部分
        tl.store(adj_raw_save_base + c_idx[:, None] * D + offs_d[None, :], adj_raw)
        tl.store(h_save_base + c_idx[:, None] * DH + offs_dh[None, :], h_t)

        # ── R18
        adj_h_R18 = tl.dot(adj_raw, tl.trans(W2), out_dtype=tl.float32, input_precision="ieee")
        adj_h = adj_h_p1 + adj_h_R18

        # ── R19
        adj_h_pre += adj_h * G_p_t

        # 把 adj_h_pre 写到 HBM,Python 算 dW1 = K.T @ adj_h_pre
        tl.store(adj_h_pre_save_base + c_idx[:, None] * DH + offs_dh[None, :], adj_h_pre)

        # ── R20:仅算 dK 的 R20 贡献(无需 W1!直接靠 adj_h_pre 在 Python 里 bmm)
        # 但 dK 自己需要 R20:dK += adj_h_pre @ W1.T
        # 我们这里 SRAM 装不下 W1。把 adj_h_pre 已经写出去了,Python 端算 dK_R20 = bmm(adj_h_pre, W1.T) 然后 add。
        # → dK 在 Python 里汇总 R2(adj_K_p1)+R13(adj_K_R13)+R20(从 adj_h_pre 和 W1 算)
        # 因此这里只写 dK_part = adj_K_p1 + adj_K_R13;Python add R20 部分

        dK_partial = adj_K_p1 + adj_K_R13
        tl.store(dK_base + c_idx[:, None] * D + offs_d[None, :], dK_partial)

    # final
    tl.store(dgamma_base + offs_d, acc_dgamma)


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_contig_fp32(t: Tensor) -> Tensor:
    return t.contiguous().float()


def _pick_block_c(C: int, kernel: str = "fwd") -> int:
    """根据 C 选 BLOCK_C。fwd/bwd 区分:bwd persistent 多半,需更小 tile。"""
    if kernel == "bwd":
        # bwd 状态密集,即使 W1/W2/d_g_W1/d_g_W2 都移到 tile 内,
        # 30+ 个 8KB adjoint 张量同时 live,SRAM 极紧 → BLOCK_C=16
        if C >= 16:
            return 16
        return C
    # fwd 持久状态多 (W1/W2/acc 共 128KB),per-tile 简单,BLOCK_C=32 OK
    if C >= 32:
        return 32
    if C >= 16:
        return 16
    return C


def triton_inner_sgd_fwd(
    K: Tensor,
    V: Tensor,
    lr: Tensor,
    gamma: Tensor,
    W1: Tensor,
    W2: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict]:
    assert K.is_cuda, "Triton kernel requires CUDA tensors"
    BHN, C, D = K.shape
    DH = W1.shape[-1]
    assert V.shape == (BHN, C, D)
    assert lr.shape == (BHN, C)
    assert gamma.shape == (BHN, D)
    assert W1.shape == (BHN, D, DH)
    assert W2.shape == (BHN, DH, D)

    out_dtype = K.dtype

    K_f = _ensure_contig_fp32(K)
    V_f = _ensure_contig_fp32(V)
    lr_f = _ensure_contig_fp32(lr)
    gamma_f = _ensure_contig_fp32(gamma)
    W1_f = _ensure_contig_fp32(W1)
    W2_f = _ensure_contig_fp32(W2)

    device = K.device
    g_gamma = torch.empty((BHN, D), device=device, dtype=torch.float32)
    g_W1 = torch.empty((BHN, D, DH), device=device, dtype=torch.float32)
    g_W2 = torch.empty((BHN, DH, D), device=device, dtype=torch.float32)
    L_c = torch.empty((BHN, C), device=device, dtype=torch.float32)
    h_pre_save = torch.empty((BHN, C, DH), device=device, dtype=torch.float32)
    ln_save = torch.empty((BHN, C, D), device=device, dtype=torch.float32)
    sigma_save = torch.empty((BHN, C), device=device, dtype=torch.float32)

    block_c = _pick_block_c(C, kernel="fwd")
    grid = (BHN,)
    _hippo_inner_sgd_fwd_kernel[grid](
        K_f, V_f, lr_f, gamma_f, W1_f, W2_f,
        g_gamma, g_W1, g_W2, L_c,
        h_pre_save, ln_save, sigma_save,
        C=C, D=D, DH=DH, EPS=float(eps), BLOCK_C=block_c,
        num_warps=4, num_stages=1,
    )

    saved = {
        "K": K_f, "V": V_f, "lr": lr_f, "gamma": gamma_f, "W1": W1_f, "W2": W2_f,
        "h_pre": h_pre_save,
        "ln": ln_save,
        # 与 _pytorch_inner_sgd_fwd 形状对齐:bwd 期待 (BHN, C, 1) 用于广播
        "sigma": sigma_save.unsqueeze(-1),
        "eps": eps,
    }

    return (
        g_gamma.to(out_dtype),
        g_W1.to(out_dtype),
        g_W2.to(out_dtype),
        L_c.to(out_dtype),
        saved,
    )


def triton_inner_sgd_bwd(
    saved: dict,
    d_g_gamma: Tensor,
    d_g_W1: Tensor,
    d_g_W2: Tensor,
    d_L_c: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    K = saved["K"]
    V = saved["V"]
    lr = saved["lr"]
    gamma = saved["gamma"]
    W1 = saved["W1"]
    W2 = saved["W2"]
    h_pre = saved["h_pre"]
    ln = saved["ln"]
    sigma = saved["sigma"]
    eps = saved["eps"]

    BHN, C, D = K.shape
    DH = W1.shape[-1]
    device = K.device

    d_g_gamma_f = d_g_gamma.contiguous().float()
    d_g_W1_f = d_g_W1.contiguous().float()
    d_g_W2_f = d_g_W2.contiguous().float()
    d_L_c_f = d_L_c.contiguous().float()

    dK = torch.empty((BHN, C, D), device=device, dtype=torch.float32)
    dV = torch.empty((BHN, C, D), device=device, dtype=torch.float32)
    dlr = torch.empty((BHN, C), device=device, dtype=torch.float32)
    dgamma = torch.empty((BHN, D), device=device, dtype=torch.float32)

    # Triton kernel 内部 SRAM 装不下 W1 + d_g_W1 + d_g_W2 + acc_dW1 + acc_dW2
    # → 把 dW1 / dW2 / dK_R20 三块 bmm 移到 Python 外算
    adj_h_pre_save = torch.empty((BHN, C, DH), device=device, dtype=torch.float32)
    adj_raw_save = torch.empty((BHN, C, D), device=device, dtype=torch.float32)
    adj_g_h_save = torch.empty((BHN, C, DH), device=device, dtype=torch.float32)
    g_raw_save = torch.empty((BHN, C, D), device=device, dtype=torch.float32)
    h_save = torch.empty((BHN, C, DH), device=device, dtype=torch.float32)

    block_c = _pick_block_c(C, kernel="bwd")
    grid = (BHN,)
    _hippo_inner_sgd_bwd_kernel[grid](
        K, V, lr, gamma, W2,
        h_pre, ln, sigma,
        d_g_gamma_f, d_g_W1_f, d_g_W2_f, d_L_c_f,
        dK, dV, dlr, dgamma,
        adj_h_pre_save, adj_raw_save, adj_g_h_save, g_raw_save, h_save,
        C=C, D=D, DH=DH, EPS=float(eps), BLOCK_C=block_c,
        num_warps=4, num_stages=1,
    )

    # ── Python bmm 收尾(已 fp32):
    #   dW1 = K.T @ adj_h_pre                            (D, DH)
    #   dW2 = adj_g_h.T @ g_raw + h.T @ adj_raw           (DH, D)
    #   dK 还差 R20 部分:adj_h_pre @ W1.T  → 加到 dK
    dW1 = torch.bmm(K.transpose(-2, -1), adj_h_pre_save)                                   # (BHN, D, DH)
    dW2 = (
        torch.bmm(adj_g_h_save.transpose(-2, -1), g_raw_save)
        + torch.bmm(h_save.transpose(-2, -1), adj_raw_save)
    )                                                                                         # (BHN, DH, D)
    dK_R20 = torch.bmm(adj_h_pre_save, W1.transpose(-2, -1))                                  # (BHN, C, D)
    dK.add_(dK_R20)

    return dK, dV, dlr, dgamma, dW1, dW2
