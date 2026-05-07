# 心核 (Xinhe) — 统一状态涌现实验

# 关 TorchScript / TensorExpr / NVFuser jit fusion。NM `batch_size=chunk_size` 路径
# 触发的 fusion 在 NVRTC 上有 bf16 codegen bug(`__nv_bfloat16 undefined`)。
# NM inner SGD 的 vmap(grad) 不需要 jit fuser 加速,任何入口(train / evaluate / tests)
# import xinhe 时都先关掉。compile_backbone_layers 用的是 torch.compile,与 jit fuser
# 不是同一套机制,不冲突。
import torch as _torch
_torch._C._jit_set_profiling_executor(False)
_torch._C._jit_set_profiling_mode(False)
_torch._C._jit_override_can_fuse_on_gpu(False)
_torch._C._jit_override_can_fuse_on_cpu(False)
try:
    _torch._C._jit_set_nvfuser_enabled(False)
except Exception:
    pass
