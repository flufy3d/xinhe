"""离线人格注入协议(零号纪元)的 Expert_persona 模型。

每个 PersonaExpert = 与 Qwen MLP 同形的 SwiGLU(gate_proj + up_proj + down_proj)。
PersonaExpertStack = N 个独立 PersonaExpert,与配置的 layer_indices 一一对应。

两种使用形态:
  1. 离线训练:scripts/train_persona.py 直接跑 PersonaExpertStack,不接 backbone
  2. 推理时挂载到 XinheModel:由 attach_persona_expert(model, ckpt) 自动:
       - 读 ckpt 内嵌的 manifest(layer_indices / hidden_dim / intermediate_dim)
       - 实例化 PersonaExpertStack 并 load 权重
       - monkey-patch 把每层 layer.mlp 替换成 MLPWithPersona(base_mlp + expert)
       - 永久冻结 expert 参数(协议:训完后转只读)
     注入后 backbone.forward_blocks 的每层 mlp 输出自动 += expert(x),不需改 forward 主路径。

down_proj 用 std = init_scale / sqrt(2 * n_layers) 小初始化,确保训前 Expert(x) ≈ 0,
余弦损失初始 ≈ 1,训练稳定。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonaExpert(nn.Module):
    """单层 SwiGLU MLP,与 Qwen MLP 同形。

    forward(x): (..., D) → (..., D)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PersonaExpertStack(nn.Module):
    """N 个独立 Expert,N = len(layer_indices)。

    forward(h): h shape (B, N, D) → out shape (B, N, D),逐层独立 expert。
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        init_scale: float = 0.02,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers 必须 > 0")
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.experts = nn.ModuleList([
            PersonaExpert(hidden_dim, intermediate_dim) for _ in range(n_layers)
        ])
        self._init_weights(init_scale)

    def _init_weights(self, init_scale: float) -> None:
        # gate_proj / up_proj 用标准 init_scale,down_proj 用 init_scale / sqrt(2*N) 压小
        # 让训前 Expert(x) ≈ 0,cos_loss ≈ 1,后续训练单调上升。
        down_std = init_scale / math.sqrt(2 * self.n_layers)
        for ex in self.experts:
            nn.init.normal_(ex.gate_proj.weight, std=init_scale)
            nn.init.normal_(ex.up_proj.weight,   std=init_scale)
            nn.init.normal_(ex.down_proj.weight, std=down_std)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, D) → (B, N, D)。

        每层 expert 接 h[:, i] 独立 forward,stack 回去。
        """
        if h.dim() != 3 or h.size(1) != self.n_layers:
            raise ValueError(
                f"PersonaExpertStack.forward: 期望 (B, {self.n_layers}, D),收到 {tuple(h.shape)}"
            )
        outs = [self.experts[i](h[:, i, :]) for i in range(self.n_layers)]
        return torch.stack(outs, dim=1)


# --------------------------------------------------------------------------- #
# 推理时注入 XinheModel                                                       #
# --------------------------------------------------------------------------- #
#
# 实现路径:forward hook on layer 输出,而**不是** monkey-patch layer.mlp。
#
# 为什么不挂在 layer.mlp 上:
#   训练时 expert 输入 x = hidden_states[layer_idx + 1],即 layer 的**最终输出**
#   (post-attention + post-mlp + 双残差,Qwen RMSNorm 流式 residual stream)。
#   若把 expert 挂在 layer.mlp 上,推理时 x = post_attention_layernorm 的输出
#   (MLP 的输入,RMSNorm 之后的归一化值),与训练分布在量级与方向特征上都不同,
#   expert 输出量级直接污染 MLP 出口 → 推理乱码。
#
# 协议公式 `MLP_out = MLP_base + Expert_persona` 是字面表达,工程上等价于
# "每层 layer 输出累加一个 persona 增量"(因为 mlp_out 的增量经 residual 直通
# layer 输出),后者训推一致,数学等价且分布严格匹配。


class _PersonaLayerHook:
    """挂在 transformer layer 上的 forward hook:layer_out += scale * expert(layer_out)。

    scale 缩放原因:expert 训练时只在"块末 token + 同章下一块方向"上有监督信号。
    推理时 hook 作用于序列每个 token,且对话上下文与小说分布 OOD,
    expert 输出方向可能与当前语境冲突。8 层累积会破坏语言能力 → 乱码。

    max_ratio 硬约束:||delta|| / ||h|| ≤ max_ratio (per-token, last dim RMS)。
    SwiGLU 在 trigger 词(罗兰/女巫等训练高频词)上 gate 突激活,delta 范数
    可能比平均大数十倍,即便 scale=0.3 也会瞬间推 hidden 出语言流形 → 第一个
    token 就乱码。这个 clip 防尖刺,平均位置不影响。

    scale=1.0 = 协议字面公式;0.0 = 等价 enabled=False。
    max_ratio<=0 = 关闭 clip(只用 scale)。

    enabled 开关给 chat.py 做对照实验(关掉等价于协议中的 I_p=0)。
    handle 由 attach_persona_expert 填充,detach 时调用 .remove() 摘除。
    """

    def __init__(
        self,
        persona_expert: PersonaExpert,
        scale: float = 1.0,
        max_ratio: float = 0.1,
    ):
        self.persona_expert = persona_expert
        self.enabled: bool = True
        self.scale: float = scale
        self.max_ratio: float = max_ratio
        self.handle = None     # torch.utils.hooks.RemovableHandle,attach 时填

    def _add_expert(self, h: torch.Tensor) -> torch.Tensor:
        if self.scale == 0.0:
            return h
        delta = self.persona_expert(h)
        if self.scale != 1.0:
            delta = delta * self.scale
        if self.max_ratio > 0:
            # per-token RMS clip:对最后维(hidden)算 RMS,逐 token 缩放避免单点爆炸
            h_f = h.float()
            d_f = delta.float()
            h_rms = h_f.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            d_rms = d_f.pow(2).mean(dim=-1, keepdim=True).sqrt()
            cap = (self.max_ratio * h_rms) / d_rms.clamp_min(1e-6)
            cap = cap.clamp(max=1.0).to(delta.dtype)
            delta = delta * cap
        return h + delta

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        # Qwen 风格 layer.forward 通常直接返回 hidden_states tensor;
        # 若上游版本改回 tuple/dataclass,兼容 (hidden_states, ...) 形态。
        if isinstance(output, tuple):
            return (self._add_expert(output[0]),) + output[1:]
        if hasattr(output, "last_hidden_state"):
            # 极少见(layer 级一般不会 dataclass),但留个兜底
            output.last_hidden_state = self._add_expert(output.last_hidden_state)
            return output
        return self._add_expert(output)


def _layer_compute_dtype(layer: nn.Module) -> tuple[torch.dtype, torch.device]:
    """取 layer 的"计算 dtype/device":锁定 layer.mlp 内的 Linear weight。

    为什么不直接 next(layer.parameters()):
      - 排第一的常是 input_layernorm.weight (RMSNorm,fp32 保留数值稳定)
      - Qwen3.5 是混合 attention(linear_attention + full_attention 交替),
        不同子类型 layer 的参数顺序不同;某些 layer 第一个 dim≥2 weight
        是 linear_attention 内部小 Linear,可能 fp32 → expert 被错 cast。
    layer.mlp 在所有 24 层都存在,内部 SwiGLU 三个 Linear 都是 backbone 统一 dtype(bf16)。
    """
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        for p in mlp.parameters():
            if p.dim() >= 2:
                return p.dtype, p.device
    # fallback:若 layer 没有 .mlp(理论上不应发生),退回扫整 layer
    for p in layer.parameters():
        if p.dim() >= 2:
            return p.dtype, p.device
    p = next(layer.parameters(), None)
    if p is None:
        return torch.float32, torch.device("cpu")
    return p.dtype, p.device


def _resolve_qwen_layers(backbone: nn.Module) -> nn.ModuleList:
    """定位 Qwen 风格 backbone 的 transformer layers 列表。

    QwenBackbone 包装 AutoModelForCausalLM,layers 在 model.model.layers。
    若结构不匹配抛清晰错误,避免静默 monkey-patch 错位置。
    """
    inner = getattr(backbone, "model", None)
    if inner is None:
        raise RuntimeError(
            f"backbone 无 .model 属性 ({type(backbone).__name__});"
            f"PersonaExpert 注入目前只支持 Qwen 风格 backbone"
        )
    text_model = getattr(inner, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise RuntimeError(
            f"backbone.model 缺少 .model.layers ({type(inner).__name__});"
            f"PersonaExpert 注入目前只支持 Qwen 风格 backbone"
        )
    return text_model.layers


def attach_persona_expert(
    xinhe_model,
    ckpt_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    enabled: bool = True,
    scale: float = 1.0,
    max_ratio: float = 0.1,
    skip_last_n: int = 0,
    skip_layers: Sequence[int] = (),
) -> dict:
    """从 train_persona.py 输出的 ckpt 自动挂载 PersonaExpert 到 XinheModel。

    - 读 ckpt 的 layer_indices / hidden_dim / intermediate_dim
    - 实例化 PersonaExpertStack(init_scale=0,等 load_state_dict 覆盖)
    - load 权重,放到与 backbone 同 device/dtype
    - 在每个目标 layer 注册 forward hook:layer_out += expert(layer_out)
    - 全部 expert 参数冻结(协议:Embryonic 阶段结束后转只读)
    - 把 stack 挂到 xinhe_model.persona_expert_stack,把 hooks 挂到 _persona_hooks

    Args:
        xinhe_model: XinheModel 实例(已加载 backbone 与 LoRA)
        ckpt_path: persona_expert.pt(或 step_*.pt)
        map_location: 加载设备
        enabled: 注入后立即激活 expert(False 用于对照实验)

    Returns:
        摘要 dict,含 layer_indices/n_params 等。

    重复 attach 安全:先 detach 已存在的 hook,避免叠加。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"persona ckpt 不存在: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location=map_location, weights_only=True)

    required = {"model", "n_layers", "hidden_dim", "intermediate_dim", "manifest"}
    missing = required - set(sd.keys())
    if missing:
        raise RuntimeError(
            f"persona ckpt 缺字段 {missing};请确认是 train_persona.py 的输出"
        )

    n_layers = int(sd["n_layers"])
    hidden_dim = int(sd["hidden_dim"])
    intermediate_dim = int(sd["intermediate_dim"])
    manifest = sd["manifest"]
    layer_indices = list(manifest.get("layer_indices", []))
    if len(layer_indices) != n_layers:
        raise RuntimeError(
            f"manifest.layer_indices ({layer_indices}) 长度与 n_layers ({n_layers}) 不一致"
        )

    # 形参与 backbone 校验
    backbone = xinhe_model.backbone
    bb_hidden = backbone.get_hidden_size() if hasattr(backbone, "get_hidden_size") else None
    if bb_hidden is not None and bb_hidden != hidden_dim:
        raise RuntimeError(
            f"persona ckpt hidden_dim={hidden_dim} 与 backbone hidden_size={bb_hidden} 不匹配"
        )
    layers = _resolve_qwen_layers(backbone)
    n_backbone_layers = len(layers)
    if max(layer_indices) >= n_backbone_layers or min(layer_indices) < 0:
        raise RuntimeError(
            f"layer_indices {layer_indices} 越界:backbone 共 {n_backbone_layers} 层"
        )

    # 重复 attach:先把旧的 hook 摘掉,避免一次推理过两遍 expert
    detach_persona_expert(xinhe_model)

    # 实例化 + load + 冻结
    stack = PersonaExpertStack(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        init_scale=0.0,        # load_state_dict 会覆盖,这里只占形参
    )
    stack.load_state_dict(sd["model"], strict=True)
    for p in stack.parameters():
        p.requires_grad = False
    stack.eval()

    # 对齐 backbone 的"计算 dtype/device"(避免 RMSNorm fp32 误导,见 _layer_compute_dtype)
    base_dtype, base_device = _layer_compute_dtype(layers[layer_indices[0]])
    stack.to(device=base_device, dtype=base_dtype)

    # 计算实际要挂的 layer 子集
    # skip_last_n:从 layer_indices 末尾跳过 N 层(deepest 通常 cos 最低,污染最大)
    # skip_layers:显式排除某些 backbone layer idx
    skip_set = set(skip_layers)
    n_attach = max(0, len(layer_indices) - max(0, skip_last_n))
    attach_set = {li for li in layer_indices[:n_attach] if li not in skip_set}
    skipped = [li for li in layer_indices if li not in attach_set]

    # 注册 forward hook:layer 输出处累加 expert 增量
    hooks: list[_PersonaLayerHook] = []
    for kv_idx, layer_idx in enumerate(layer_indices):
        if layer_idx not in attach_set:
            continue
        layer = layers[layer_idx]
        # 多卡时 stack 已对齐到 layer_indices[0] 的设备;
        # 严格 per-layer 对齐:每个 expert 单独跟随其目标 layer 设备/计算 dtype
        layer_dtype, layer_device = _layer_compute_dtype(layer)
        stack.experts[kv_idx].to(device=layer_device, dtype=layer_dtype)
        hook = _PersonaLayerHook(
            stack.experts[kv_idx], scale=scale, max_ratio=max_ratio,
        )
        hook.enabled = enabled
        hook.handle = layer.register_forward_hook(hook)
        hooks.append(hook)

    # 挂到 model 上(便于 save / inspect / detach / toggle)
    xinhe_model.persona_expert_stack = stack
    xinhe_model._persona_hooks = hooks

    n_params = sum(p.numel() for p in stack.parameters())
    return {
        "ckpt_path": str(ckpt_path),
        "novel_stem": manifest.get("novel_stem"),
        "layer_indices": layer_indices,
        "attached_layers": sorted(attach_set),
        "skipped_layers": skipped,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "n_params": n_params,
        "wrapped_count": len(hooks),
        "enabled": enabled,
        "scale": scale,
        "max_ratio": max_ratio,
    }


def detach_persona_expert(xinhe_model) -> int:
    """逆操作:摘除所有 _PersonaLayerHook,移除 persona_expert_stack。

    返回摘除的 hook 数。chat.py 的 /persona off 不调这个(它只 toggle enabled);
    detach 用于真正"卸载"——后续 forward 不再额外 hook 调用,零开销。
    """
    n = 0
    hooks: list[_PersonaLayerHook] = getattr(xinhe_model, "_persona_hooks", []) or []
    for hook in hooks:
        if hook.handle is not None:
            hook.handle.remove()
            hook.handle = None
            n += 1
    if hasattr(xinhe_model, "_persona_hooks"):
        delattr(xinhe_model, "_persona_hooks")
    if hasattr(xinhe_model, "persona_expert_stack"):
        delattr(xinhe_model, "persona_expert_stack")
    return n


def set_persona_enabled(xinhe_model, enabled: bool) -> int:
    """切换所有已注入 hook 的 enabled 开关(对照实验用)。返回切换的层数。"""
    hooks: list[_PersonaLayerHook] = getattr(xinhe_model, "_persona_hooks", []) or []
    for hook in hooks:
        hook.enabled = enabled
    return len(hooks)


def set_persona_scale(xinhe_model, scale: float) -> int:
    """运行时调整所有 expert 注入强度(0=纯 base,1=协议字面)。返回切换的层数。"""
    hooks: list[_PersonaLayerHook] = getattr(xinhe_model, "_persona_hooks", []) or []
    for hook in hooks:
        hook.scale = float(scale)
    return len(hooks)


def set_persona_max_ratio(xinhe_model, max_ratio: float) -> int:
    """运行时调整 ||delta||/||h|| 上限(0=关闭 clip,默认 0.1)。返回切换的层数。"""
    hooks: list[_PersonaLayerHook] = getattr(xinhe_model, "_persona_hooks", []) or []
    for hook in hooks:
        hook.max_ratio = float(max_ratio)
    return len(hooks)
