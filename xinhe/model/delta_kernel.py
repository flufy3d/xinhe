"""
Delta Rule 写 kernel 后端派发：FLA Triton (Linux) 自动探测 + PyTorch 全平台 fallback。

后端契约（与 Hippocampus._delta_parallel 对齐）：
    delta_rule_write(W, k, v, beta, gamma, *, backend="auto") -> Tensor

    W: (B, H, d_v, d_k)
    k: (B, H, T, d_k)   已 L2 归一
    v: (B, H, T, d_v)
    beta:  (B, H, T) ∈ (0, 1)
    gamma: (B, H, T) ∈ (0, 1)   per-head per-token decay
    返回: W_new 同 W 形状

import 时一次性探测 _FLA_AVAILABLE，结果固定不变。运行时报错直接抛出，不静默降级。
"""
from __future__ import annotations
import sys
import torch

# 注：torch 后端实现在 Hippocampus._delta_parallel 静态方法上；调用时延迟导入避免循环依赖
# （hippocampus.py 在顶层 import 本模块以拿 delta_rule_write）。

# ── 一次性探测 ──
_FLA_AVAILABLE: bool = False
_chunk_gated_delta_rule = None  # type: ignore[assignment]

if sys.platform == "linux":
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _chunk_gated_delta_rule
        _FLA_AVAILABLE = True
    except ImportError:
        _FLA_AVAILABLE = False

_LOGGED: bool = False


def _resolve_backend(backend: str, W: torch.Tensor) -> str:
    if backend == "auto":
        return "fla" if (_FLA_AVAILABLE and W.is_cuda) else "torch"
    if backend == "fla":
        if not _FLA_AVAILABLE:
            raise RuntimeError(
                "delta_backend='fla' 但 flash-linear-attention 不可用 "
                f"(sys.platform={sys.platform!r})。Linux 装 fla 或改 backend='auto'/'torch'。"
            )
        if not W.is_cuda:
            raise RuntimeError("delta_backend='fla' 需要 CUDA tensor")
        return "fla"
    if backend == "torch":
        return "torch"
    raise ValueError(f"未知 backend={backend!r}，应为 auto/fla/torch")


def _maybe_log(chosen: str) -> None:
    global _LOGGED
    if _LOGGED:
        return
    _LOGGED = True
    print(f"[delta_kernel] backend={chosen} (fla_available={_FLA_AVAILABLE}, platform={sys.platform})")


def delta_rule_write(
    W: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """带 γ 衰减的 Delta Rule chunkwise 写。

    后端可选：auto（自动）/ fla（Triton, Linux）/ torch（_delta_parallel 全平台）。
    """
    chosen = _resolve_backend(backend, W)
    _maybe_log(chosen)
    if chosen == "fla":
        return _fla_write(W, k, v, beta, gamma)
    from .hippocampus import Hippocampus  # 延迟导入避循环
    return Hippocampus._delta_parallel(W, k, v, beta, gamma)


def _fla_write(
    W: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """FLA chunk_gated_delta_rule 适配器。

    形状契约转换：
      内部 (B,H,T,D)         → FLA (B,T,H,D)
      内部 W: (B,H,d_v,d_k)  → FLA initial_state: (B,H,d_k,d_v)（K 在前 V 在后）
      gamma → g = log(gamma) 传入 FLA（log-decay 约定）

    dtype 处理：
      FLA 0.4.2 `chunk_gated_delta_rule` 用 @custom_fwd 装饰。在外层 autocast bf16
      上下文里，custom_fwd 会把 q/k/v/beta 下转 bf16，但 kernel 内部 A 矩阵（gating
      cumsum）保持 fp32，导致 wy_fast.py 内 `tl.dot(b_A, b_vb)` fp32 vs bf16 dtype
      mismatch（已知坑）。因此显式 `autocast(enabled=False)` 关掉 + 全部走 fp32，
      让 kernel 全程 fp32 路径走通；牺牲 bf16 张量核加速但保留 chunkwise SRAM tiling
      核心收益（不物化 T×T 中间矩阵到 HBM）。

    TODO（首次 Linux 部署 verify）：
      若 test_fla_matches_torch_when_available 等价误差 > 1e-2，把
      `g_fla = torch.log(gamma...)` 改为 `g_fla = gamma...`（FLA `g` log/raw 约定可能跨版本变化）。

    我们只取 final_state（写侧不需要 attention 形式的输出），q==k 是合法占位。
    """
    assert _chunk_gated_delta_rule is not None  # _resolve_backend 已守过

    orig_dtype = W.dtype

    # (B,H,T,D) -> (B,T,H,D)，统一 fp32
    k_fla = k.transpose(1, 2).contiguous().float()
    v_fla = v.transpose(1, 2).contiguous().float()
    q_fla = k_fla  # write-only 路径，q 占位用 k
    # (B,H,T) -> (B,T,H)，fp32
    beta_fla = beta.transpose(1, 2).contiguous().float()
    g_fla = torch.log(gamma.clamp_min(1e-8)).transpose(1, 2).contiguous().float()

    # W: (B,H,d_v,d_k) -> initial_state: (B,H,d_k,d_v)，fp32
    init_state = W.transpose(-1, -2).contiguous().float()

    device_type = W.device.type if W.is_cuda else "cpu"
    # head_first 参数在 FLA 0.4.2+ 已 deprecated（默认即 False，传入会刷 UserWarning），不再显式传
    with torch.amp.autocast(device_type=device_type, enabled=False):
        out, final_state = _chunk_gated_delta_rule(
            q=q_fla,
            k=k_fla,
            v=v_fla,
            g=g_fla,
            beta=beta_fla,
            initial_state=init_state,
            output_final_state=True,
        )
    # final_state: (B,H,d_k,d_v) -> (B,H,d_v,d_k)
    W_new = final_state.transpose(-1, -2).contiguous().to(orig_dtype)
    return W_new
