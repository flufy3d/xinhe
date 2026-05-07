"""NeuralMemoryPair (v9) — Hippocampus(快适配)+ Neocortex(慢知识基底)。

两条路径**训练机制完全不同**(这是 v9 的核心架构选择):

  - Hippocampus(海马):浅 MLP(默认 depth=2 exp=2.0),**TTT inner SGD** 路径。
    每 token 当作一次 SGD 例子,vmap(grad) 内层算 ∇W,W ← W - lr·∇W。
    每 episode 内 fast weights 演化,不带跨 episode 持久状态。
    走 NeuralMemory.forward,xinhe 内 patch 了 `read_before_write=True`。

  - Neocortex(大脑皮层):深 MLP(默认 depth=4 exp=4.0),**普通 nn.Module + 标准 backprop**。
    weights 是 nn.Parameter,跨 episode 持久。outer Adam 经 backbone-driven backward
    更新 — 即"白天活动 → 慢慢沉淀的世界知识"。无 vmap,无 fast weights,无 chunk SGD,
    扩深(future)只受标准激活内存约束,不会爆 vmap 梯度张量。

  Sleep 阶段(P-cap 之后)是把 Hippo 累积的活动 replay 到 Neo,推动 Neo 权重一次大更新;
  也通过同样的标准 backprop。

forward:
  1. r_h, h_new = hippocampus(x, state=h_old, read_before_write=True)   # TTT
  2. r_n         = neocortex(x)                                           # 普通 fwd
  3. q = gate_q(x); gate = softmax([⟨q,r_h⟩, ⟨q,r_n⟩] / √d)               # 内容感知
  4. mem_out = gate·{r_h, r_n} → mem_rmsnorm
  5. alpha = sigmoid(alpha_logit).clamp(min);受 mem_alpha_override 覆写
  6. return (x + alpha·mem_out, LayerMemState(hippo=h_new, neo=None), aux)

LayerMemState.neo 永远 None(为兼容旧字段保留位置),Neo 没有 episode 内状态。
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _upgrade_rmsnorms_to_adaptive(module: nn.Module) -> None:
    """递归把 module 子树里所有 `type(m) is nn.RMSNorm` 实例换成 AdaptiveRMSNorm,
    γ 数据原封拷过去。Hippo NeuralMemory 内部的 RMSNorm 由此一并升级。"""
    for name, child in list(module.named_children()):
        if type(child) is nn.RMSNorm:
            shape = child.normalized_shape
            dim = shape[0] if isinstance(shape, (tuple, list)) else int(shape)
            affine = child.weight is not None
            new = AdaptiveRMSNorm(dim, eps=child.eps, elementwise_affine=affine)
            if affine:
                new.weight.data.copy_(child.weight.data)
            setattr(module, name, new)
        else:
            _upgrade_rmsnorms_to_adaptive(child)


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
    """模型级 state 容器,per-layer LayerMemState + 跨 turn mem token snapshots。

    mem_snapshots: list of (B, N_mem, hidden_size) tensors,turn 间累积 ——
    模拟 MAC 的"跨 turn KV 持久化"(用 hidden-state passing 等效)。第 t turn 末
    forward 取 fresh mem 位置 hidden state 追加进 snapshots,下 turn 在序列起头
    重放,让 attention 看到所有历史 turn 的 mem 摘要。
    """

    def __init__(
        self,
        layers: Optional[dict[int, LayerMemState]] = None,
        mem_snapshots: Optional[list] = None,
    ):
        self.layers: dict[int, LayerMemState] = layers if layers is not None else {}
        self.mem_snapshots: list = mem_snapshots if mem_snapshots is not None else []

    @classmethod
    def init(cls, layer_indices: list[int]) -> "XinheMemoryState":
        return cls({l: LayerMemState(None, None) for l in layer_indices}, [])

    def detach(self) -> "XinheMemoryState":
        return XinheMemoryState(
            {l: s.detach() for l, s in self.layers.items()},
            [s.detach() for s in self.mem_snapshots],
        )

    def to(self, device) -> "XinheMemoryState":
        return XinheMemoryState(
            {l: s.to(device) for l, s in self.layers.items()},
            [s.to(device) for s in self.mem_snapshots],
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


class NeocortexBlock(nn.Module):
    """Neo: 多头深 MLP,普通 nn.Module + 标准 backprop。

    与 Hippo(TTT inner SGD)对比:
      - 不持有 episode-scoped fast weights(weights 是 nn.Parameter,跨 episode/batch 持久)
      - 不走 vmap+grad,前向就是 multi-head GeLU MLP
      - 显存仅吃常规 activation,扩深不爆

    结构(每 head 独立 MLP weights,广播到 batch 维度):
      x: (..., d_total)
      → pre_rmsnorm
      → split heads: (..., heads, d_head)
      → for each layer: GeLU(prev) @ W_h    (W_h shape: (heads, d_in, d_out))
      → merge heads: (..., d_total)
    """

    def __init__(
        self,
        d_total: int,
        n_heads: int,
        d_head: int,
        depth: int,
        expansion: float,
        pre_norm: bool = True,
    ):
        super().__init__()
        assert d_total == n_heads * d_head
        self.d_total = d_total
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm = AdaptiveRMSNorm(d_total) if pre_norm else nn.Identity()

        dim_hidden = int(d_head * expansion)
        dims = [d_head] + [dim_hidden] * (depth - 1) + [d_head]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(n_heads, di, do))
            for di, do in zip(dims[:-1], dims[1:])
        ])
        for w in self.weights:
            for h in range(n_heads):
                nn.init.xavier_uniform_(w[h])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_total)
        h = self.norm(x)
        # split into heads: (..., heads, d_head)
        leading = h.shape[:-1]
        h_per_head = h.view(*leading, self.n_heads, self.d_head)
        for ind, w in enumerate(self.weights):
            if ind > 0:
                h_per_head = F.gelu(h_per_head)
            # einsum: ...hd, hde -> ...he
            h_per_head = torch.einsum('...hd,hde->...he', h_per_head, w)
        # merge heads back
        return h_per_head.reshape(*leading, self.d_total)


class NeuralMemoryPair(nn.Module):
    """挂在单个 full-attn 层上的 Hippo(TTT)+ Neo(static MLP)容器。"""

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
        hippo_base_lr: float = 1e-2,
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

        # Hippo:TTT inner SGD via NeuralMemory(快适配,episode 内演化)
        self.hippocampus = self._build_neural_memory(
            d_total, n_heads, d_head, hippo_mlp_depth, hippo_mlp_expansion,
            hippo_retention, hippo_base_lr, chunk_size,
        )
        # Neo:普通深 MLP(慢知识基底,outer Adam 慢更新)
        self.neocortex = NeocortexBlock(
            d_total, n_heads, d_head,
            depth=neo_mlp_depth, expansion=neo_mlp_expansion,
        )

        # 内容感知 gate:q from x;<q, r_h>/<q, r_n> 决定权重
        self.gate_q = nn.Linear(d_total, d_total, bias=False)
        nn.init.xavier_uniform_(self.gate_q.weight)

        # alpha 开度:sigmoid(alpha_logit).clamp(alpha_min);受 mem_alpha_override 覆写
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_logit_init)))

        # 控制 mem_out 跟 attn_out 同量级,避免幅值竞争盖住 backbone
        self.mem_rmsnorm = AdaptiveRMSNorm(d_total)

        # daytime_plastic_hippo:控制 episode 内 fast weights 是否演化(set_daytime_plastic 切)
        # daytime_plastic_neo:Neo 是普通 backprop 路径,这个 flag 仅作 API 兼容,
        #   实际"冻 Neo"应该改 self.neocortex.requires_grad_(False)。
        self._daytime_plastic_hippo: bool = True
        self._daytime_plastic_neo: bool = (phase == "P-cap")

        # Hippo NeuralMemory 内部仍是普通 nn.RMSNorm(γ fp32),autocast 下仍报警告。
        # 升级它们到 AdaptiveRMSNorm:weight 保留 fp32 不动 Adam,forward 即时 cast 走 fused。
        _upgrade_rmsnorms_to_adaptive(self.hippocampus)

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
            new_state:  LayerMemState(hippo=NeuralMemState, neo=None)
            aux:        {"gate_entropy_reg_loss": Tensor, ...}
        """
        if layer_state is None:
            layer_state = LayerMemState(None, None)

        h_old = layer_state.hippo

        # 1. Hippo:TTT inner SGD,read_before_write=True → retrieve 用入口 weights
        r_h, h_new = self.hippocampus(x, state=h_old, read_before_write=True)
        if not self._daytime_plastic_hippo:
            h_new = h_old

        # 2. Neo:普通 multi-head MLP,无 state、无 vmap、无 chunk SGD
        r_n = self.neocortex(x)

        # 3. 内容感知 gate:用 x 投影出 q,跟 r_h / r_n 点积取置信度
        q = self.gate_q(x)
        scale = 1.0 / math.sqrt(q.shape[-1])
        logit_h = (q * r_h).sum(dim=-1, keepdim=True) * scale
        logit_n = (q * r_n).sum(dim=-1, keepdim=True) * scale
        gate = torch.softmax(torch.cat([logit_h, logit_n], dim=-1), dim=-1)  # (B, T, 2)

        # 4. mem_out + RMSNorm
        mem_out = gate[..., 0:1] * r_h + gate[..., 1:2] * r_n
        mem_out = self.mem_rmsnorm(mem_out)

        # 5. alpha 开度。reparam 保证 alpha ∈ [alpha_min, 1] 且全程可导:
        #    alpha = alpha_min + (1 - alpha_min) * sigmoid(alpha_logit)
        #    旧版 sigmoid().clamp(min) 在 sigmoid<min 时梯度断 → alpha_logit 永不学。
        if mem_alpha_override is not None:
            alpha = torch.tensor(float(mem_alpha_override), device=x.device, dtype=x.dtype)
        else:
            alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(self.alpha_logit)

        x_out = x + alpha * mem_out

        # 6. gate 熵正则:防 gate 单边塌缩(λ * (-H) 加入 loss → 等价 λ 最大化 H)
        gate_entropy = -(gate.clamp_min(1e-9).log() * gate).sum(dim=-1).mean()
        aux = {
            "gate_entropy": gate_entropy.detach(),
            "gate_entropy_reg_loss": -self.gate_entropy_lambda * gate_entropy,
            "alpha_eff": (alpha.detach() if torch.is_tensor(alpha) else torch.tensor(alpha)),
            "gate_mean_h": gate[..., 0].mean().detach(),
            "gate_mean_n": gate[..., 1].mean().detach(),
            "mem_out": mem_out,   # MAC 模式下 caller 从这里取原 mem 输出去填 fresh_mem 位置
        }

        return x_out, LayerMemState(hippo=h_new, neo=None), aux
