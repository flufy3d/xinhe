"""NeuralMemoryPair (v9) — 双 NeuralMemory:Hippocampus 浅 MLP + Neocortex 深 MLP

每个 full-attn 层挂一个实例。基于 NeuralMemory(test-time SGD MLP fast-weights,
xinhe 内 patch 了 read_before_write=True 入口分支)。生物类比:
  - Hippocampus(海马):depth=2 浅 MLP,白天 test-time SGD,retention=0.99 自然遗忘
  - Neocortex(大脑皮层):depth=4 深 MLP,白天冻结(P-cap 例外),Sleep 标准 backprop

forward:
  1. retrieve(x, h_old_weights, read_before_write=True) → r_h
  2. retrieve(x, n_old_weights, read_before_write=True) → r_n
  3. store(x) 内嵌在 NeuralMemory.forward → next_state(daytime_plastic=False 时丢弃)
  4. q = gate_q(x); logit = <q, r>/√d; gate = softmax([h, n])     # 内容感知门控
  5. mem_out = gate * r_{h,n} 加权 → mem_rmsnorm
  6. alpha = sigmoid(alpha_logit).clamp(min);可被 mem_alpha_override 覆写
  7. return (x + alpha * mem_out, LayerMemState, aux)

read 在 write 之前由 NeuralMemory 的 `read_before_write=True` 保证(见
xinhe/model/neural_memory.py 的 forward 内入口 retrieve 分支)。
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from .neural_memory import NeuralMemory, NeuralMemState, mem_state_detach
from .memory_models import MemoryMLP


def _mem_state_to(state: NeuralMemState, device) -> NeuralMemState:
    """walk NeuralMemState pytree,把所有 tensor leaf 移到 device。"""
    moved = tree_map(
        lambda t: t.to(device) if torch.is_tensor(t) else t,
        tuple(state),
    )
    return NeuralMemState(*moved)


def _logit(p: float, eps: float = 1e-6) -> float:
    """logit(p) = log(p / (1-p)),有 eps 防溢出。"""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


@dataclass
class LayerMemState:
    """单层 (hippo, neo) 状态。None 表示未初始化,NeuralMemory.forward 内 lazy init。"""
    hippo: Optional[NeuralMemState] = None
    neo: Optional[NeuralMemState] = None

    def detach(self) -> "LayerMemState":
        return LayerMemState(
            hippo=mem_state_detach(self.hippo) if self.hippo is not None else None,
            neo=mem_state_detach(self.neo) if self.neo is not None else None,
        )

    def to(self, device) -> "LayerMemState":
        return LayerMemState(
            hippo=_mem_state_to(self.hippo, device) if self.hippo is not None else None,
            neo=_mem_state_to(self.neo, device) if self.neo is not None else None,
        )


class XinheMemoryState:
    """模型级 state 容器,per-layer LayerMemState。替代 v8 的单 W 张量。"""

    def __init__(self, layers: Optional[dict[int, LayerMemState]] = None):
        self.layers: dict[int, LayerMemState] = layers if layers is not None else {}

    @classmethod
    def init(cls, layer_indices: list[int]) -> "XinheMemoryState":
        return cls({l: LayerMemState(None, None) for l in layer_indices})

    def detach(self) -> "XinheMemoryState":
        return XinheMemoryState({l: s.detach() for l, s in self.layers.items()})

    def to(self, device) -> "XinheMemoryState":
        return XinheMemoryState({l: s.to(device) for l, s in self.layers.items()})

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


class NeuralMemoryPair(nn.Module):
    """挂载在单个 full-attn 层上的双 NeuralMemory 容器。"""

    def __init__(
        self,
        d_total: int,
        n_heads: int,
        d_head: int,
        hippo_mlp_depth: int = 2,
        hippo_mlp_expansion: float = 2.0,
        neo_mlp_depth: int = 4,
        neo_mlp_expansion: float = 4.0,
        hippo_retention: float = 0.99,
        neo_retention: float = 1.0,
        hippo_base_lr: float = 1e-2,
        neo_base_lr: float = 1e-4,
        chunk_size: int = 64,
        alpha_logit_init: float = -5.0,
        alpha_min_clamp: float = 0.02,
        phase: str = "P-cap",
        gate_entropy_lambda: float = 0.01,
    ):
        super().__init__()
        assert phase in ("P-cap", "Operational"), f"unknown phase: {phase}"
        assert d_total == n_heads * d_head, \
            f"d_total ({d_total}) must equal n_heads ({n_heads}) * d_head ({d_head})"
        self.d_total = d_total
        self.n_heads = n_heads
        self.d_head = d_head
        self.alpha_min = float(alpha_min_clamp)
        self.gate_entropy_lambda = float(gate_entropy_lambda)
        self.phase = phase

        self.hippocampus = self._build_neural_memory(
            d_total, n_heads, d_head, hippo_mlp_depth, hippo_mlp_expansion,
            hippo_retention, hippo_base_lr, chunk_size,
        )
        self.neocortex = self._build_neural_memory(
            d_total, n_heads, d_head, neo_mlp_depth, neo_mlp_expansion,
            neo_retention, neo_base_lr, chunk_size,
        )

        # 内容感知 gate:q from x;<q, r_h>/<q, r_n> 决定权重
        self.gate_q = nn.Linear(d_total, d_total, bias=False)
        nn.init.xavier_uniform_(self.gate_q.weight)

        # alpha 开度:sigmoid(alpha_logit).clamp(alpha_min);受 mem_alpha_override 覆写
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))

        # 控制 mem_out 跟 attn_out 同量级,避免幅值竞争盖住 backbone
        self.mem_rmsnorm = nn.RMSNorm(d_total)

        # daytime_plastic flags(可经 set_daytime_plastic 切换)
        # P-cap:Neo 也开 plastic(lr=neo_base_lr 默认 1e-4),防 mix_gate 学死 Neo
        # Operational:Neo 严格冻结,只在 sleep 打开
        self._daytime_plastic_hippo: bool = True
        self._daytime_plastic_neo: bool = (phase == "P-cap")

    @staticmethod
    def _build_neural_memory(
        d_total: int,
        n_heads: int,
        d_head: int,
        depth: int,
        expansion: float,
        retention: float,
        base_lr: float,
        chunk_size: int,
    ) -> NeuralMemory:
        mlp = MemoryMLP(dim=d_head, depth=depth, expansion_factor=expansion)
        nm = NeuralMemory(
            dim=d_total,
            dim_head=d_head,
            heads=n_heads,
            chunk_size=chunk_size,
            model=mlp,
            init_decay_bias=_logit(1.0 - retention),  # decay = 1 - retention 静态
            default_step_transform_max_lr=base_lr,
            qk_rmsnorm=True,
            per_head_learned_parameters=True,
            momentum=True,
            momentum_order=1,
            pre_rmsnorm=True,
            post_rmsnorm=False,
        )
        # freeze to_decay_factor → 静态 retention(不被外层 backprop 学走)
        for p in nm.to_decay_factor.parameters():
            p.requires_grad = False
        # NeuralMemory 用 `repeat(p, '... -> h ...', h=heads)` 把单 head 参数广播成
        # (H, ...) 后塞进 nn.Parameter,这是 expanded view → 同一物理内存被 H 个位置共享。
        # Adam 等 in-place optimizer 会触发 "more than one element refers to single memory location"
        # 这里强制 clone 出 contiguous Parameter,断开 view-share。
        new_params = nn.ParameterList([
            nn.Parameter(p.detach().clone().contiguous(), requires_grad=p.requires_grad)
            for p in nm.memory_model_parameters
        ])
        nm.memory_model_parameters = new_params
        return nm

    def set_daytime_plastic(
        self,
        hippo: Optional[bool] = None,
        neo: Optional[bool] = None,
    ) -> None:
        """切换白天可塑性。learning_session 阶段 1 应都设为 False(走干净 forward)。"""
        if hippo is not None:
            self._daytime_plastic_hippo = bool(hippo)
        if neo is not None:
            self._daytime_plastic_neo = bool(neo)

    @property
    def daytime_plastic_hippo(self) -> bool:
        return self._daytime_plastic_hippo

    @property
    def daytime_plastic_neo(self) -> bool:
        return self._daytime_plastic_neo

    def forward(
        self,
        x: torch.Tensor,
        layer_state: Optional[LayerMemState] = None,
        mem_alpha_override: Optional[float] = None,
    ) -> tuple[torch.Tensor, LayerMemState, dict]:
        """
        x:           (B, T, d_total)
        layer_state: 上一次 forward 的 LayerMemState,None 时 lazy init
        mem_alpha_override: float 或 None。给 learning_session 阶段 1 传 0.0 走干净路径

        返回:
            x_out:      (B, T, d_total) — 已加 alpha*mem_out 残差
            new_state:  LayerMemState(hippo, neo)
            aux:        {"gate_entropy_reg_loss": Tensor, ...}
        """
        if layer_state is None:
            layer_state = LayerMemState(None, None)

        h_old = layer_state.hippo
        n_old = layer_state.neo

        # 1. Hippo & Neo retrieve(read_before_write=True → retrieve 用入口 weights)
        #    NeuralMemory.forward 同时返回 next_state(包含 store 路径输出)
        r_h, h_new = self.hippocampus(x, state=h_old, read_before_write=True)
        r_n, n_new = self.neocortex(x, state=n_old, read_before_write=True)

        # 2. daytime_plastic=False:state 不演化(扔掉 forward 算的 next_state)
        if not self._daytime_plastic_hippo:
            h_new = h_old
        if not self._daytime_plastic_neo:
            n_new = n_old

        # 3. 内容感知 gate:用 x 投影出 q,跟 r_h / r_n 点积取置信度
        q = self.gate_q(x)
        scale = 1.0 / math.sqrt(q.shape[-1])
        logit_h = (q * r_h).sum(dim=-1, keepdim=True) * scale
        logit_n = (q * r_n).sum(dim=-1, keepdim=True) * scale
        gate = torch.softmax(torch.cat([logit_h, logit_n], dim=-1), dim=-1)  # (B, T, 2)

        # 4. mem_out + RMSNorm
        mem_out = gate[..., 0:1] * r_h + gate[..., 1:2] * r_n
        mem_out = self.mem_rmsnorm(mem_out)

        # 5. alpha 开度
        if mem_alpha_override is not None:
            alpha = torch.tensor(float(mem_alpha_override), device=x.device, dtype=x.dtype)
        else:
            alpha = torch.sigmoid(self.alpha_logit).clamp(min=self.alpha_min)

        x_out = x + alpha * mem_out

        # 6. gate 熵正则:防 gate 单边塌缩(λ * (-H) 加入 loss → 等价 λ 最大化 H)
        gate_entropy = -(gate.clamp_min(1e-9).log() * gate).sum(dim=-1).mean()
        aux = {
            "gate_entropy": gate_entropy.detach(),
            "gate_entropy_reg_loss": -self.gate_entropy_lambda * gate_entropy,
            "alpha_eff": (alpha.detach() if torch.is_tensor(alpha) else torch.tensor(alpha)),
            "gate_mean_h": gate[..., 0].mean().detach(),
            "gate_mean_n": gate[..., 1].mean().detach(),
        }

        return x_out, LayerMemState(hippo=h_new, neo=n_new), aux
