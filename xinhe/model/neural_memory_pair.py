"""State 容器 + autocast 友好 RMSNorm(单全局 QueryHead/HippoDelta 架构遗留)。

老的 NeuralMemoryPair(Hippo TTT + Neo MLP + gate_q)已删,只保留:
  - `LayerMemState` / `XinheMemoryState`:跨 turn state 容器,被 xinhe_model.py 用
  - `AdaptiveRMSNorm`:autocast 友好的 RMSNorm,HippoDelta/QueryHead 用
  - `_logit`:辅助函数,留作可能的旁路 init 使用
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(p: float, eps: float = 1e-6) -> float:
    """logit(p) = log(p / (1-p)),有 eps 防溢出。"""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class AdaptiveRMSNorm(nn.RMSNorm):
    """nn.RMSNorm 的 autocast 友好版:γ 始终保持 fp32(Adam 高精),forward 时
    按 input dtype 即时 cast。

    PyTorch 默认 nn.RMSNorm + autocast(bf16) 时,weight 是 fp32 / input 是 bf16,
    `torch.rms_norm` 的 fused 内核拒绝混合 dtype → 退到非融合路径 + 警告。本类把
    weight 临时 cast 到 input dtype 再调 fused 内核,既无警告也走 fused 快路径,
    且 fp32 master 不变。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if w is not None and w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, w, self.eps)


@dataclass
class LayerMemState:
    """单层状态容器。

    单全局架构下,只有 `_global_write_idx` 一层会持有非空 hippo(HippoDeltaState);
    其它 hook 层始终 None。neo 字段已废弃(老 NeuralMemoryPair 的位),保留位置以兼容
    旧 ckpt 字段名(load 时会被覆盖为 None)。鸭子类型:hippo 是 HippoDeltaState,
    自带 .detach() / .to(),无需 pytree walk。
    """
    hippo: Optional[object] = None
    neo: Optional[object] = None

    def detach(self) -> "LayerMemState":
        h, n = self.hippo, self.neo
        return LayerMemState(
            hippo=(h.detach() if h is not None and hasattr(h, "detach") else h),
            neo=(n.detach() if n is not None and hasattr(n, "detach") else n),
        )

    def to(self, device) -> "LayerMemState":
        h, n = self.hippo, self.neo
        return LayerMemState(
            hippo=(h.to(device) if h is not None and hasattr(h, "to") else h),
            neo=(n.to(device) if n is not None and hasattr(n, "to") else n),
        )


class XinheMemoryState:
    """模型级 state 容器,per-layer LayerMemState 字典。

    单全局架构下只有 _global_write_idx 那一层真的会被填进 HippoDeltaState,
    其它 hook 层永远是 LayerMemState(None, None) — 容器形态保留是为了 trainer
    和 evaluate 的 .detach() / .to() 通用循环代码不用因为架构改造而改 API。
    """

    def __init__(
        self,
        layers: Optional[dict[int, LayerMemState]] = None,
    ):
        self.layers: dict[int, LayerMemState] = layers if layers is not None else {}

    @classmethod
    def init(cls, layer_indices: list[int]) -> "XinheMemoryState":
        return cls({l: LayerMemState(None, None) for l in layer_indices})

    def detach(self) -> "XinheMemoryState":
        return XinheMemoryState(
            {l: s.detach() for l, s in self.layers.items()},
        )

    def to(self, device) -> "XinheMemoryState":
        return XinheMemoryState(
            {l: s.to(device) for l, s in self.layers.items()},
        )

    def __getitem__(self, l: int) -> LayerMemState:
        return self.layers[l]

    def __setitem__(self, l: int, s: LayerMemState):
        self.layers[l] = s

    def get(self, l: int, default=None) -> Optional[LayerMemState]:
        return self.layers.get(l, default)

    def items(self):
        return self.layers.items()

    def keys(self):
        return self.layers.keys()

    def values(self):
        return self.layers.values()
