# 心核 (Xinhe) — 统一状态涌现实验

# 关 TorchScript / TensorExpr jit fusion。NM `batch_size=chunk_size` 路径触发的
# fusion 在 NVRTC 上有 bf16 codegen bug(`__nv_bfloat16 undefined`,nvfuser 残留)。
# 关掉 TensorExpr 的 GPU/CPU fusion 已足够规避;NVFuser 在 PyTorch 2.5+ 已被上游移除,
# 不再调 _jit_set_nvfuser_enabled(deprecated 会打 nvfuser warning)。
# NM inner SGD 的 vmap(grad) 不需要 jit fuser 加速,任何入口 import xinhe 时都先关掉。
# compile_backbone_layers 用的是 torch.compile,与 jit fuser 不是同一套机制,不冲突。
import os as _os
from pathlib import Path as _Path
import torch as _torch

_torch._C._jit_set_profiling_executor(False)
_torch._C._jit_set_profiling_mode(False)
_torch._C._jit_override_can_fuse_on_gpu(False)
_torch._C._jit_override_can_fuse_on_cpu(False)

# Triton kernel cache:工程根的 .cache/triton(与 .cache/episodes / HF cache 同档)。
# Windows 默认 cache 路径有 260 字符限会炸 triton-windows;统一指到工程目录避雷。
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent
_TRITON_CACHE = _PROJECT_ROOT / ".cache" / "triton"
_TRITON_CACHE.mkdir(parents=True, exist_ok=True)
_os.environ.setdefault("TRITON_CACHE_DIR", str(_TRITON_CACHE))

# CUDA prime:在 fla import 之前 prime CUDA(touch 一个 tensor),让 triton 检测到 cuda。
if _torch.cuda.is_available():
    try:
        _torch.cuda.init()
        _ = _torch.zeros(1, device="cuda")
    except Exception:  # pragma: no cover
        pass

# fla 0.5.0 兼容 monkey-patch:
# fla.utils 在 import time 通过 triton runtime 检测 device,若 triton driver 还没 active
# (Linux + 新 torch 常见)会 fallback 到 'cpu',然后把 device_torch_lib 设成 torch.cpu,
# 后续在 ChunkGatedDeltaRuleFunction.apply 等地方调 torch.cpu.device(idx) → torch 2.11+
# 已移除该 API,抛 AttributeError。
# 修法:让 torch.cpu.device 成为 no-op context(cpu 没有 device index 概念,nullcontext 安全)。
# 仅当 fla / torch 版本对齐(fla 自己检测对 cuda)时 device_torch_lib 才指向 torch.cuda,
# 此 patch 不会被调用,纯防御。
import contextlib as _ctx
if not hasattr(_torch.cpu, "device"):
    _torch.cpu.device = lambda index=0: _ctx.nullcontext()
