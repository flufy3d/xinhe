"""
delta_kernel 后端派发 + per-segment 外层 checkpoint 测试。

覆盖：
- torch 后端通过 dispatcher 仍然与 _delta_loop fp64 等价（兼容性守门）
- _FLA_AVAILABLE flag 与平台相符
- FLA vs torch 后端 bf16 等价（仅 Linux + FLA 装好时跑）
- per_segment_checkpoint=True/False 梯度数值一致（外层 ckpt 不改训练轨迹）
"""
import sys
import torch
import torch.nn.functional as F
import pytest

from xinhe.model.delta_kernel import delta_rule_write, _FLA_AVAILABLE
from xinhe.model.hippocampus import Hippocampus
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from tests.test_model import MockBackbone


def test_dispatch_torch_matches_loop_fp64():
    """torch 后端经 dispatcher 调用，与 _delta_loop fp64 等价（守 wrapper 不引入 dtype/contiguity bug）。"""
    torch.manual_seed(0)
    B, H, T, d_k, d_v = 2, 4, 16, 8, 8
    W0 = torch.zeros(B, H, d_v, d_k, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, d_k, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, d_v, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    gamma = 0.3 + 0.7 * torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

    W_loop = Hippocampus._delta_loop(W0.clone(), k, v, beta, gamma, T)
    W_dispatch = delta_rule_write(W0.clone(), k, v, beta, gamma, backend="torch")
    diff = (W_loop - W_dispatch).abs().max().item()
    assert diff < 1e-8, f"dispatch[torch] vs loop 数值不一致: max|diff|={diff}"


def test_fla_available_flag_matches_platform():
    """非 Linux → _FLA_AVAILABLE 必须 False；Linux 上视环境装否而定（不强制 True）。"""
    if sys.platform != "linux":
        assert _FLA_AVAILABLE is False, \
            f"非 Linux ({sys.platform}) 不应能用 FLA"


def test_explicit_fla_raises_when_unavailable():
    """显式 backend='fla' 在 FLA 不可用时应抛 RuntimeError，不静默降级。"""
    if _FLA_AVAILABLE:
        pytest.skip("FLA 可用时此测试不适用")
    B, H, T, d_k, d_v = 1, 2, 4, 4, 4
    W = torch.zeros(B, H, d_v, d_k)
    k = F.normalize(torch.randn(B, H, T, d_k), dim=-1)
    v = torch.randn(B, H, T, d_v)
    beta = torch.sigmoid(torch.randn(B, H, T))
    gamma = torch.sigmoid(torch.randn(B, H, T))
    with pytest.raises(RuntimeError):
        delta_rule_write(W, k, v, beta, gamma, backend="fla")


@pytest.mark.skipif(not _FLA_AVAILABLE, reason="FLA 未装")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA Triton kernel 需要 CUDA")
def test_fla_matches_torch_when_available():
    """Linux + FLA + CUDA 下 fla 与 torch 后端 bf16 输出在 fp32 比对应 max|diff| < 1e-2。

    若失败 > 1e-2，第一嫌犯是 _fla_write 里 `g = log(γ)` 与 FLA 期望约定不符；
    把那行改成 `g = γ` 重测即可。
    """
    torch.manual_seed(0)
    B, H, T, d_k, d_v = 2, 4, 64, 32, 32
    device = "cuda"
    dtype = torch.bfloat16
    W = torch.zeros(B, H, d_v, d_k, device=device, dtype=dtype)
    k = F.normalize(torch.randn(B, H, T, d_k, device=device, dtype=dtype), dim=-1)
    v = torch.randn(B, H, T, d_v, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))
    gamma = 0.5 + 0.5 * torch.sigmoid(torch.randn(B, H, T, device=device, dtype=dtype))

    W_torch = delta_rule_write(W.clone(), k, v, beta, gamma, backend="torch").float()
    W_fla = delta_rule_write(W.clone(), k, v, beta, gamma, backend="fla").float()
    diff = (W_torch - W_fla).abs().max().item()
    assert diff < 1e-2, f"FLA vs torch bf16 等价误差超阈值: max|diff|={diff:.4f}"


def _make_test_model(seed: int = 42) -> XinheModel:
    torch.manual_seed(seed)
    config = XinheConfig(
        hidden_size=64,
        n_heads=4,
        head_dim=16,
        read_scale_init=0.0,  # 让 read 真的影响输出，制造非平凡 grad
        lora_rank=0,
        freeze_backbone=False,
        delta_backend="torch",  # 测试与 FLA 无关，用 torch 后端确保平台无关
    )
    backbone = MockBackbone(hidden_size=64, vocab_size=100)
    return XinheModel(config, backbone=backbone)


def test_per_segment_checkpoint_grad_match():
    """per_segment_checkpoint True/False 下两次 segment + 一次 backward 的参数梯度应数值一致。

    验证外层 torch.utils.checkpoint 不改变训练轨迹（只换显存/计算的 tradeoff）。
    """
    torch.manual_seed(0)
    B, T = 2, 8
    seg1 = torch.randint(0, 100, (B, T))
    seg2 = torch.randint(0, 100, (B, T))
    labels2 = torch.randint(0, 100, (B, T))

    def run(per_segment_checkpoint: bool):
        model = _make_test_model(seed=42)
        model.train()
        model.config.per_segment_checkpoint = per_segment_checkpoint

        state = model.init_state(B)
        r1 = model(seg1, state)
        r2 = model(seg2, r1["state_next"], labels=labels2)
        r2["loss"].backward()
        grads = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters() if p.grad is not None}
        return grads

    g_off = run(per_segment_checkpoint=False)
    g_on = run(per_segment_checkpoint=True)

    assert g_off.keys() == g_on.keys(), \
        f"参数集不一致: off-on={g_off.keys() - g_on.keys()}, on-off={g_on.keys() - g_off.keys()}"

    max_diff = 0.0
    worst_name = ""
    for name in g_off:
        d = (g_off[name] - g_on[name]).abs().max().item()
        if d > max_diff:
            max_diff = d
            worst_name = name
    # checkpoint(use_reentrant=False) 在 fp32 下应当 bit-exact 或极接近；放 1e-5 容忍数值噪声
    assert max_diff < 1e-5, f"checkpoint on/off 梯度不一致 (worst={worst_name}): {max_diff}"
