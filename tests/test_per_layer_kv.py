"""测试 per-layer K/V persistent memory(paper Titans MAC 严格形态)。

验证 PerLayerPersistentKV 模块本身的形状/独立性/grad,以及
XinheQwen3FullAttention wrapper 正确拼接到 K/V 计算图。
"""
import pytest
import torch
import torch.nn as nn

from xinhe.model.per_layer_kv import PerLayerPersistentKV


def test_per_layer_kv_shape():
    """K_pers / V_pers 形状 (N_p, n_kv_heads, head_dim)。"""
    m = PerLayerPersistentKV(n_persistent=8, n_kv_heads=2, head_dim=64)
    assert m.K_pers.shape == (8, 2, 64)
    assert m.V_pers.shape == (8, 2, 64)


def test_per_layer_kv_expand_for_attention():
    """expand_for_attention 返回 (B, n_kv, N_p, d) 形状(transpose 后,匹配 attention)。"""
    m = PerLayerPersistentKV(n_persistent=4, n_kv_heads=2, head_dim=8)
    K, V = m.expand_for_attention(B=3, dtype=torch.float32, device="cpu")
    assert K.shape == (3, 2, 4, 8)
    assert V.shape == (3, 2, 4, 8)


def test_per_layer_kv_trainable():
    """K_pers / V_pers 是 nn.Parameter 且 requires_grad=True。"""
    m = PerLayerPersistentKV(n_persistent=4, n_kv_heads=2, head_dim=8)
    assert isinstance(m.K_pers, nn.Parameter)
    assert isinstance(m.V_pers, nn.Parameter)
    assert m.K_pers.requires_grad
    assert m.V_pers.requires_grad


def test_per_layer_kv_grad_flow():
    """expand_for_attention + 简单 loss → backward 后 K_pers/V_pers grad 非零。"""
    torch.manual_seed(42)
    m = PerLayerPersistentKV(n_persistent=4, n_kv_heads=2, head_dim=8)
    K, V = m.expand_for_attention(B=2, dtype=torch.float32, device="cpu")
    # 简单 dot product loss
    loss = (K * V).sum()
    loss.backward()
    assert m.K_pers.grad is not None
    assert m.V_pers.grad is not None
    assert m.K_pers.grad.abs().sum() > 0
    assert m.V_pers.grad.abs().sum() > 0


def test_per_layer_kv_independent_per_instance():
    """两个独立实例的 K_pers 不共享(每层独立)。"""
    m1 = PerLayerPersistentKV(n_persistent=4, n_kv_heads=2, head_dim=8)
    m2 = PerLayerPersistentKV(n_persistent=4, n_kv_heads=2, head_dim=8)
    # 不同 seed 初始化,不会全等
    assert not torch.equal(m1.K_pers, m2.K_pers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Qwen3 backbone")
def test_xinhe_qwen3_wrapper_full_integration():
    """端到端集成:LoRA + per-layer K/V + XinheModel forward + backward。"""
    from xinhe.model.config import XinheConfig
    from xinhe.model.xinhe_model import XinheModel

    cfg = XinheConfig(
        backbone_model_path="./models/qwen3.5-0.8b",
        hidden_size=1024, n_heads=8, head_dim=256,
        n_persistent_per_layer=4, n_mem_tokens=4,
        mac_inject_logit_init=10.0,
        lora_rank=4, lora_alpha=8,
        freeze_backbone=True, per_segment_checkpoint=False,
        compile_backbone_layers=False, mem_chunk_size=4,
    )
    model = XinheModel(cfg).to("cuda", dtype=torch.bfloat16)

    B, T = 1, 8
    state = model.init_state(B)
    input_ids = torch.randint(0, 1000, (B, T), device="cuda")
    labels = torch.randint(0, 1000, (B, T), device="cuda")
    result = model(input_ids, state, labels=labels)

    assert result["logits"].shape == (B, T, 248320)
    result["loss"].backward()

    # K_pers grad 非零(per-layer K/V 在计算图)
    found_kpers_grad = False
    for n, p in model.backbone.named_parameters():
        if "K_pers" in n and p.grad is not None and p.grad.abs().sum() > 0:
            found_kpers_grad = True
            break
    assert found_kpers_grad, "K_pers 没收到梯度,attention 计算图断了"

    # LoRA grad 非零(注:启动期 lora_B=0 → lora_A 的 grad 经 lora_B=0 中断 = 0,
    # 所以应当检查 lora_B 而非 lora_A;LoRA 训练几步后 lora_A 才有 grad)
    found_lora_grad = False
    for n, p in model.backbone.named_parameters():
        if "lora_B" in n and p.grad is not None and p.grad.abs().sum() > 0:
            found_lora_grad = True
            break
    assert found_lora_grad, "LoRA(lora_B)没收到梯度"
