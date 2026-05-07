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
