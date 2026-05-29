"""Gated DeltaNet 写 kernel 后端派发(Phase 3;resurrect 自 v8 `dc597dc^:delta_kernel.py`)。

boxed rule(蓝图;g=1 即纯 delta,**已含删旧关联项 (I-βkkᵀ)**,对治 NIH distract 累加):
    M_t = M_{t-1}(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ
gated 增强(g_t<1,额外遗忘门;gated_delta=True 时启用):
    M_t = M_{t-1}(g_t I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ

后端策略(v8 实测,见 [[feedback_delta_train_torch_infer_fla]]):
  - 训练 (model.training=True):强制 backend="torch"。FLA Triton backward 在 bf16 累加
    引入 5-25% 梯度幅度误差,长序列累积 → 优化器收敛到读不出 M 的最优。
  - 推理 (model.eval()):auto 模式 Linux+CUDA 优先 FLA(forward 与 torch cos≈0.99999)。
import 时一次探测 _FLA_AVAILABLE;运行时报错直接抛出,不静默降级。

约定形状:M(B,H,d_v,d_k) k(B,H,T,d_k,已 L2 归一) v(B,H,T,d_v) beta(B,H,T)∈(0,1)。
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
                f"(sys.platform={sys.platform!r})。Linux 装 fla 或改 'auto'/'torch'。"
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


def suppress_log(backend: str) -> None:
    """标记 backend 已 logged(trainer __init__ 抑制验证段噪音)。"""
    _LOGGED_BACKENDS.add(backend)


def delta_rule_write(
    W: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """delta rule chunkwise 写。g=None → plain(蓝图 boxed g=1);g 给定 → gated(走 recurrent)。
    训练务必 backend="torch"(调用方按 model.training 决定)。返回 M_new 同 W 形状。"""
    if g is not None:
        # gated(g<1)暂走 token-recurrent(正确优先;chunked-gated 待 GPU 验证后再上)
        return torch_delta_recurrent(W, k, v, beta, g=g)
    chosen = _resolve_backend(backend, W)
    _maybe_log(chosen)
    if chosen == "fla":
        return _fla_write(W, k, v, beta)
    return torch_delta_chunk(W, k, v, beta)


def torch_delta_chunk(W, k, v, beta) -> torch.Tensor:
    """Chunkwise 并行 plain Delta Rule(Yang et al. 2024;搬自 v8 `_delta_parallel`):
      W_T = W_0 + Σ_i β_i v'_i k_iᵀ,  v'_i = v_i - W_0 k_i - Σ_{l<i} β_l (k_l·k_i) v'_l
    三角系统 (I + A_tril) V' = V_rhs 一次 solve_triangular。fp32 求解(bf16 输入自动 upcast)。"""
    B, H, T, d_k = k.shape
    d_v = v.shape[-1]
    orig_dtype = W.dtype
    solve_dtype = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype

    W_f = W.to(solve_dtype)
    k_f = k.to(solve_dtype)
    v_f = v.to(solve_dtype)
    beta_f = beta.to(solve_dtype)

    W0k = torch.einsum("bhvd,bhtd->bhtv", W_f, k_f)        # W_0 k_i
    V_rhs = v_f - W0k

    KK = torch.einsum("bhid,bhjd->bhij", k_f, k_f)         # (B,H,T,T)
    A = beta_f.unsqueeze(-2) * KK                          # A[i,l] = β_l (k_i·k_l)
    tril_mask = torch.tril(torch.ones(T, T, device=k.device, dtype=torch.bool), diagonal=-1)
    A_tril = A * tril_mask.to(A.dtype)

    eye_T = torch.eye(T, device=k.device, dtype=solve_dtype)
    L = eye_T + A_tril
    Vp = torch.linalg.solve_triangular(
        L.reshape(B * H, T, T), V_rhs.reshape(B * H, T, d_v),
        upper=False, unitriangular=True,
    ).reshape(B, H, T, d_v)

    weighted = beta_f.unsqueeze(-1) * Vp                   # β_i v'_i
    W_new = W_f + torch.einsum("bhtv,bhtd->bhvd", weighted, k_f)
    return W_new.to(orig_dtype)


def torch_delta_recurrent(W, k, v, beta, g: torch.Tensor | None = None) -> torch.Tensor:
    """Token-recurrent 参考实现(单测 ground truth;gated g 也走这条)。O(T) 串行,仅小尺寸/验证用。
      M_t = (g_t·)M_{t-1} + β_t (v_t - M_{t-1} k_t) k_tᵀ
    g=None 即 plain(展开 = M(I-βkkᵀ)+βvkᵀ,蓝图 boxed)。"""
    B, H, T, d_k = k.shape
    orig_dtype = W.dtype
    # bf16/fp16 upcast fp32 数值稳定;fp32/fp64 原样(单测 fp64 要能精确对上 chunk)
    work = torch.float32 if orig_dtype in (torch.bfloat16, torch.float16) else orig_dtype
    M = W.to(work).clone()
    k_f, v_f, beta_f = k.to(work), v.to(work), beta.to(work)
    g_f = g.to(work) if g is not None else None
    for t in range(T):
        kt = k_f[:, :, t]                                  # (B,H,d_k)
        vt = v_f[:, :, t]                                  # (B,H,d_v)
        bt = beta_f[:, :, t].unsqueeze(-1).unsqueeze(-1)   # (B,H,1,1)
        Mk = torch.einsum("bhvd,bhd->bhv", M, kt)          # (B,H,d_v)
        upd = bt * torch.einsum("bhv,bhd->bhvd", vt - Mk, kt)
        if g_f is not None:
            gt = g_f[:, :, t].unsqueeze(-1).unsqueeze(-1)
            M = gt * M + upd
        else:
            M = M + upd
    return M.to(orig_dtype)


def _fla_write(W, k, v, beta) -> torch.Tensor:
    """FLA chunk_delta_rule(plain)适配器:推理前向加速用。q/k/v/beta bf16,state fp32。"""
    assert _chunk_delta_rule is not None
    orig_dtype = W.dtype
    bf = torch.bfloat16
    k_fla = k.transpose(1, 2).contiguous().to(bf)
    v_fla = v.transpose(1, 2).contiguous().to(bf)
    beta_fla = beta.transpose(1, 2).contiguous().to(bf)
    init_state = W.transpose(-1, -2).contiguous().float()
    _out, final_state = _chunk_delta_rule(
        q=k_fla, k=k_fla, v=v_fla, beta=beta_fla,
        initial_state=init_state, output_final_state=True,
    )
    return final_state.transpose(-1, -2).contiguous().to(orig_dtype)
