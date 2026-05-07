"""测试 LoRA 注入与零初始化语义(v9.5 恢复)。"""
import torch
import torch.nn as nn

from xinhe.model.lora import LoRALinear, inject_lora, get_lora_params


class MiniBackbone(nn.Module):
    """小 backbone,内含 q_proj/k_proj/v_proj/o_proj 模仿 transformer attention。"""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp_gate_proj = nn.Linear(hidden_size, hidden_size)  # 不是 target,验证不被注入

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = self.o_proj(q + k + v)
        return out + self.mlp_gate_proj(x)


def test_inject_lora_finds_qkvo():
    """inject_lora 找到 q/k/v/o_proj,不动 mlp_gate_proj。"""
    model = MiniBackbone()
    replaced = inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=4, alpha=8,
    )
    assert len(replaced) == 4
    assert isinstance(model.q_proj, LoRALinear)
    assert isinstance(model.k_proj, LoRALinear)
    assert isinstance(model.v_proj, LoRALinear)
    assert isinstance(model.o_proj, LoRALinear)
    # mlp_gate_proj 不是 target,保持原样
    assert not isinstance(model.mlp_gate_proj, LoRALinear)


def test_lora_zero_init_no_change():
    """LoRA 注入后,因为 lora_B = 0,启动期 forward 输出与未注入完全一致。"""
    torch.manual_seed(42)
    model = MiniBackbone()
    x = torch.randn(2, 4, 32)
    out_before = model(x).clone()

    inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=4, alpha=8,
    )
    out_after = model(x)
    # lora_B = 0 → 增量 0 → 完全一致(数值精度内)
    assert torch.allclose(out_before, out_after, atol=1e-5)


def test_lora_grad_flow():
    """backward 后 lora_A.grad / lora_B.grad 非零(LoRA 在学)。"""
    torch.manual_seed(42)
    model = MiniBackbone()
    inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=4, alpha=8,
    )
    # 给 lora_B 一个非零扰动,否则 backward 时 lora_A 的梯度依赖于 lora_B 也无法非零
    # 实际训练中第一个 step 后 lora_B 就非零了,这里用扰动模拟
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.add_(torch.randn_like(m.lora_B) * 0.01)

    x = torch.randn(2, 4, 32)
    target = torch.randn(2, 4, 32)
    out = model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()

    # 至少一个 LoRA Linear 的 lora_A / lora_B 有 grad
    found_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.lora_A.grad is not None and m.lora_A.grad.abs().sum() > 0:
                found_grad = True
            if m.lora_B.grad is not None and m.lora_B.grad.abs().sum() > 0:
                found_grad = True
    assert found_grad, "LoRA 参数 grad 都是 0,LoRA 未参与计算图"


def test_get_lora_params():
    """get_lora_params 收集所有 LoRA Linear 的 lora_A + lora_B。"""
    model = MiniBackbone()
    inject_lora(
        model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        rank=4, alpha=8,
    )
    params = get_lora_params(model)
    # 4 个 LoRALinear × 2 (A + B) = 8
    assert len(params) == 8


def test_lora_original_frozen():
    """LoRA 注入后,原 Linear 的 weight/bias requires_grad=False。"""
    model = MiniBackbone()
    inject_lora(
        model,
        target_modules=["q_proj"],
        rank=4, alpha=8,
    )
    assert isinstance(model.q_proj, LoRALinear)
    assert not model.q_proj.original.weight.requires_grad
    if model.q_proj.original.bias is not None:
        assert not model.q_proj.original.bias.requires_grad
    # lora_A / lora_B 应该 trainable
    assert model.q_proj.lora_A.requires_grad
    assert model.q_proj.lora_B.requires_grad
