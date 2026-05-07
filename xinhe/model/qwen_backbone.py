"""
Qwen Backbone 适配器

包装 HuggingFace transformers 的 Qwen 模型为统一 backbone 接口。
支持 Qwen3-0.6B 及同系列模型。
"""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM

from .backbone import BackboneBase
from .config import XinheConfig


class QwenBackbone(nn.Module, BackboneBase):
    """
    Qwen backbone: 通过 transformers AutoModel 加载。

    支持 Qwen3.5 系列 (0.8B / 4B / 9B)，混合 attention 架构。
    """

    def __init__(self, config: XinheConfig):
        nn.Module.__init__(self)
        self.config = config

        # 多卡时用 device_map="auto" 分散显存，单卡不用（避免 accelerate 把层放到 CPU）
        device_map = "auto" if torch.cuda.device_count() > 1 else None
        self.model = AutoModelForCausalLM.from_pretrained(
            config.backbone_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map,
        )
        self._hidden_size = self.model.config.hidden_size
        self._gradient_checkpointing = config.gradient_checkpointing

        # 冻结主干参数
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # 局部 torch.compile:只编 full_attention 层(我们挂 NeuralMemoryPair 的层)。
        # 跳过 linear_attention 层 — 它们用 fla.modules.FusedRMSNormGated(Triton),
        # 让 Dynamo 在 backward 时遭遇 fla.utils.get_multiprocessor_count(@functools.cache)
        # 和 triton 的 cuda_utils.get_device_properties(C 扩展)→ Dynamo warn_once 噪音。
        # 让 linear_attention 走 eager 反而能享受 FLA Triton 内核的原生加速。
        # NeuralMemoryPair 的 hook 也不进 compile 边界 → 不触发 Dynamo ↔ vmap+grad 的
        # saved_tensors_hooks 冲突。多卡 device_map="auto" 时跳过(compile 跨设备不稳)。
        if (getattr(config, "compile_backbone_layers", False)
                and torch.cuda.device_count() <= 1):
            try:
                compiled_count = 0
                for i, layer in enumerate(self.model.model.layers):
                    if getattr(layer, "layer_type", None) != "full_attention":
                        continue
                    self.model.model.layers[i] = torch.compile(
                        layer, mode="default", fullgraph=False, dynamic=False,
                    )
                    compiled_count += 1
                total_layers = len(self.model.model.layers)
                print(f"[torch.compile] 已编译 {compiled_count}/{total_layers} 个 full_attention "
                      f"层(linear_attention 走 eager 享受 FLA Triton 内核)")
            except Exception as e:
                print(f"[torch.compile] 跳过(异常: {e})")

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[callable] = None,
    ) -> torch.Tensor:
        # Qwen 使用 RoPE，需要构建 position embeddings
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        # 逐层跑 transformer blocks（多卡时跟随 layer 设备移动）
        for layer_idx, layer in enumerate(self.model.model.layers):
            # State read hook（在 backbone 层之前，checkpoint 之外）
            if layer_hook is not None:
                hidden_states = layer_hook(hidden_states, layer_idx)

            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(layer_device)
            position_embeddings = (
                position_embeddings[0].to(layer_device),
                position_embeddings[1].to(layer_device),
            )
            if self._gradient_checkpointing and self.training:
                hidden_states = torch_checkpoint(
                    self._layer_forward,
                    layer, hidden_states, attention_mask, position_embeddings,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.model.model.norm(hidden_states)
        return hidden_states

    @staticmethod
    def _layer_forward(layer, hidden_states, attention_mask, position_embeddings):
        return layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_num_layers(self) -> int:
        return len(self.model.model.layers)

    def get_hook_layer_indices(self) -> list[int]:
        """只在 full attention 层前执行 hook（DeltaNet 层跳过）"""
        return [i for i, layer in enumerate(self.model.model.layers)
                if getattr(layer, 'layer_type', None) == 'full_attention']
