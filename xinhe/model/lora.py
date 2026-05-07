"""
LoRA (Low-Rank Adaptation) 实现 — v9.5 恢复版(从 git 7c42936)

注入到 backbone 的 attention 层(q/k/v/o),让冻结的 backbone 学会处理 v9 MAC 的
OOD 输入(per-layer persistent K/V + fresh_mem soft prompt)。
LoRA 与 MAC 是 producer/consumer 协同:MAC 放 prefix,LoRA 学怎么读。
零初始化 → 启动时增量 0,不破坏 frozen backbone 行为。
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    在原始 Linear 层旁边添加低秩旁路: output = W @ x + (B @ A) @ x * (alpha/rank)

    初始化时 B=0,所以初始输出增量为零 → 不破坏预训练模型。
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # 低秩矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 冻结原始权重(双重保险:backbone freeze 已做,这里显式 enforce)
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        result = self.original(x.to(self.original.weight.dtype))
        lora_out = (
            self.dropout(x)
            @ self.lora_A.to(device=x.device, dtype=x.dtype).T
            @ self.lora_B.to(device=x.device, dtype=x.dtype).T
            * self.scaling
        )
        return (result + lora_out).to(orig_dtype)


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
) -> list[str]:
    """
    在模型中查找指定名称的 Linear 层,替换为 LoRALinear。

    返回被替换的模块路径列表。
    """
    replaced = []

    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                replaced.append(name)

    return replaced


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """收集所有 LoRA 可训练参数"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
