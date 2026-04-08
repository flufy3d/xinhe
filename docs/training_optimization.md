# 训练优化指南

## 总览

以 Qwen3.5-4B backbone 为例，优化前后对比：

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| GPU 功耗 | 100-200W | 300W+ |
| GPU 利用率 | 低（大量调度等待） | 高（计算密集） |
| 训练速度 | 基准 | ~1.5-2x |

## 优化项

### 1. torch.compile (Linux, 效果最大)

将 Python 级别的逐 op 执行编译为融合的 CUDA kernel，减少 kernel launch 开销和显存读写。

- 自动启用：Linux + Triton 可用时自动开启（见 `trainer.py`）
- Windows 不支持（无 Triton），自动跳过
- 首次 forward 时 JIT 编译（1-2 分钟），之后全程加速
- 加速幅度：20-50%

注意事项：
- forward 中避免 `.item()`，会导致 graph break（已修复）
- 动态 shape 会触发重编译，xinhe 的 segment_length 固定所以没问题

### 2. flash-linear-attention (Linux)

加速 attention 计算，通过优化的 CUDA kernel 减少显存占用和计算时间。

```toml
# pyproject.toml 中已配置，Linux 自动安装
"flash-linear-attention ; sys_platform == 'linux'"
```

### 3. causal-conv1d (Linux)

flash-linear-attention 的 fast path 依赖。没有它，fla 退回慢的 torch 实现。

需要 `--no-build-isolation` 编译（确保用 venv 的 torch 而非隔离环境的）：
```bash
uv pip install causal-conv1d --no-build-isolation
```

deploy 脚本已自动处理，首次编译约 5-10 分钟。

要求：系统 CUDA toolkit 版本与 PyTorch CUDA 版本一致（如都是 13.0）。

### 4. TF32 精度

```python
torch.set_float32_matmul_precision('high')
```

允许 float32 矩阵乘法使用 TensorFloat32（19-bit），加速少量 float32 运算。主要计算已经是 bfloat16，影响较小但没坏处。

### 5. gradient_checkpointing

用重算激活值换显存。4B 模型必须开启，否则 OOM。

在 backbone yaml 中配置：
```yaml
training:
  gradient_checkpointing: true
```

## 环境要求

| 项目 | 要求 |
|------|------|
| OS | Linux（torch.compile 需要 Triton） |
| CUDA toolkit | 与 PyTorch CUDA 版本一致 |
| GPU | Ampere+ (A100/H100/RTX 4090 等) |

## deploy 自动化

`scripts/remote.py deploy` 自动处理：
- uv sync 安装依赖（flash-linear-attention 等）
- 首次安装 causal-conv1d（--no-build-isolation）
- 设置 CUDA_HOME、UV_CACHE_DIR 等环境变量
- uv 缓存持久化到 shared-storage，换 VM 不重下载
