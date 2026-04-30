"""
Delta Rule 写 kernel 后端派发(无 γ 衰减):FLA Triton(Linux)+ PyTorch fallback。

后端契约:
    delta_rule_write(W, k, v, beta, *, backend="auto") -> Tensor

    W: (B, H, d_v, d_k)
    k: (B, H, T, d_k)   已 L2 归一
    v: (B, H, T, d_v)
    beta:  (B, H, T) ∈ (0, 1)
    返回: W_new 同 W 形状

policy:
  - 训练 (model.training=True):Hippocampus.write_from_content 强制 backend="torch"。
    FLA Triton backward 在 bf16 累加上引入 5-25% 梯度幅度误差(实测 read_scale
    甚至 25%+),长序列下累积导致优化器收敛到读不出 W 的最优。
  - 推理 (model.eval()):按用户配置 (auto/fla/torch),auto 模式 Linux+CUDA 优先
    FLA。FLA forward 与 torch 差 < 0.5%(实测 cos≈0.99999),可放心用以加速。

import 时一次性探测 _FLA_AVAILABLE,结果固定不变。运行时报错直接抛出,不静默降级。
"""
from __future__ import annotations
import sys
import torch

_FLA_AVAILABLE: bool = False
_chunk_delta_rule = None  # type: ignore[assignment]

if sys.platform == "linux":
    try:
        from fla.ops.delta_rule import chunk_delta_rule as _chunk_delta_rule
        _FLA_AVAILABLE = True
    except ImportError:
        _FLA_AVAILABLE = False

_LOGGED_BACKENDS: set = set()


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
    raise ValueError(f"未知 backend={backend!r},应为 auto/fla/torch")


def _maybe_log(chosen: str) -> None:
    if chosen in _LOGGED_BACKENDS:
        return
    _LOGGED_BACKENDS.add(chosen)
    print(f"[delta_kernel] backend={chosen} (fla_available={_FLA_AVAILABLE}, platform={sys.platform})")


def delta_rule_write(
    W: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """纯 Delta Rule chunkwise 写,无 γ:
      W_t = W_{t-1} + β_t · (v_t - W_{t-1}·k_t) ⊗ k_t^T
    """
    chosen = _resolve_backend(backend, W)
    _maybe_log(chosen)
    if chosen == "fla":
        return _fla_write(W, k, v, beta)
    from .hippocampus import Hippocampus
    return Hippocampus._delta_parallel(W, k, v, beta)


def suppress_log(backend: str) -> None:
    """标记 backend 已 logged,后续首次调用不再打印。

    用途:trainer 在 __init__ 时调用 suppress_log("fla"),抑制训练进程里
    验证段的 `[delta_kernel] backend=fla` 噪音。独立运行 evaluate.py /
    chat.py / visualize_state.py 等脚本不调用本函数,首次写时正常打印。
    """
    _LOGGED_BACKENDS.add(backend)


def _fla_write(
    W: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """FLA chunk_delta_rule(无 γ)适配器。
    chunk_delta_rule 强制要求 bfloat16 (q/k/v/beta);state 仍 fp32。
    """
    assert _chunk_delta_rule is not None
    orig_dtype = W.dtype
    bf = torch.bfloat16

    k_fla = k.transpose(1, 2).contiguous().to(bf)
    v_fla = v.transpose(1, 2).contiguous().to(bf)
    q_fla = k_fla  # write-only,q 占位用 k
    beta_fla = beta.transpose(1, 2).contiguous().to(bf)
    init_state = W.transpose(-1, -2).contiguous().float()

    out, final_state = _chunk_delta_rule(
        q=q_fla, k=k_fla, v=v_fla, beta=beta_fla,
        initial_state=init_state, output_final_state=True,
    )
    return final_state.transpose(-1, -2).contiguous().to(orig_dtype)
