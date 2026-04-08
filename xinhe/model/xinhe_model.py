"""
XinheModel — 顶层模型

组合 backbone (MiniMind) + StatePlugin，实现:
- 带持久状态的 forward pass
- 带状态的文本生成
- Burn-in 初始化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import XinheConfig
from .backbone import BackboneBase
from .state_plugin import StatePlugin
from .lora import inject_lora, get_lora_params


class XinheModel(nn.Module):
    """
    心核模型: Backbone + StatePlugin

    前向传播:
        embed → plugin.inject → backbone.forward_blocks → plugin.extract_and_update → logits + state
    """

    def __init__(self, config: XinheConfig, backbone: Optional[BackboneBase] = None):
        super().__init__()
        self.config = config

        # Backbone
        if backbone is not None:
            self.backbone = backbone
        elif config.backbone_type == "qwen":
            from .qwen_backbone import QwenBackbone
            self.backbone = QwenBackbone(config)
        else:
            from .minimind_backbone import MiniMindBackbone
            self.backbone = MiniMindBackbone(config)

        # 注入 LoRA
        if config.freeze_backbone and config.lora_rank > 0:
            replaced = inject_lora(
                self.backbone,
                target_modules=config.lora_target_modules,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

        # StatePlugin
        self.plugin = StatePlugin(
            n_state=config.n_state,
            state_dim=config.state_dim,
            state_scale_init=config.state_scale_init,
            gate_bias_init=config.gate_bias_init,
        )

        # LM head (复用 backbone 的)
        self.lm_head = self.backbone.get_lm_head()

    def forward(
        self,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
    ) -> dict:
        """
        带状态的前向传播。

        参数:
            input_ids: (B, T) token ids
            state: (B, n_state, D) 当前持久状态
            labels: (B, T) 可选，用于计算 loss
            pad_token_id: 可选，padding token id，提供时自动遮蔽 padding

        返回:
            dict:
                logits: (B, T, V)
                state_next: (B, n_state, D)
                loss: scalar (如果提供了 labels)
        """
        # 1. 嵌入内容 token
        content_emb = self.backbone.embed(input_ids)  # (B, T, D)

        # 2. 注入状态 (自动检测 padding)
        content_mask = (input_ids != pad_token_id) if pad_token_id is not None else None
        hidden_states, mask = self.plugin.inject(state, content_emb, content_mask=content_mask)

        # 3. Transformer forward
        output = self.backbone.forward_blocks(hidden_states, attention_mask=mask)

        # 4. 提取新状态 + gate 更新（多卡时将输出移回 plugin 所在设备）
        plugin_device = next(self.plugin.parameters()).device
        output = output.to(plugin_device)
        state = state.to(plugin_device)
        content_output, state_next = self.plugin.extract_and_update(output, state)

        # 5. 计算 logits (只对内容部分)
        logits = self.lm_head(content_output)  # (B, T, V)

        result = {
            "logits": logits,
            "state_next": state_next,
        }

        # 6. 计算 loss (如果有 labels)
        if labels is not None:
            # 标准 next-token prediction: logits[:, :-1] vs labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # 当所有 labels 都是 -100 时 (如非 recall segment)，loss 设为 0 避免 NaN
            valid_count = (shift_labels != -100).sum()
            if valid_count > 0:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                # value token 准确率 (argmax 匹配, 零额外开销)
                valid_mask = shift_labels.view(-1) != -100
                preds = shift_logits.view(-1, shift_logits.size(-1))[valid_mask].argmax(dim=-1)
                targets = shift_labels.view(-1)[valid_mask]
                result["correct"] = (preds == targets).sum()
                result["total"] = valid_count
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                result["correct"] = 0
                result["total"] = 0
            result["loss"] = loss

        return result

    def setup_device(self, device: torch.device):
        """
        单卡: 整个模型移到 device。
        多卡: backbone 已由 device_map 分配，只移 plugin。
        """
        if torch.cuda.device_count() > 1:
            self.plugin.to(device)
        else:
            self.to(device)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """创建空白初始状态"""
        device = next(self.parameters()).device
        return self.plugin.blank_state(batch_size, device=device)

    @torch.no_grad()
    def burn_in(self, token_ids_list: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        """
        Burn-in: 将一段文本消化进状态（如 system prompt）。

        参数:
            token_ids_list: segment 列表，每个 (T,) 或 (1, T)
            batch_size: batch 大小

        返回:
            state: (B, n_state, D) 消化后的状态
        """
        state = self.init_state(batch_size)

        for token_ids in token_ids_list:
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0).expand(batch_size, -1)
            result = self.forward(token_ids, state)
            state = result["state_next"]

        return state

    @torch.no_grad()
    def generate_with_state(
        self,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.85,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
        token_callback: Optional[callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        带状态的文本生成 (自回归)。

        参数:
            input_ids: (B, T) prompt token ids
            state: (B, n_state, D) 当前状态
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling
            eos_token_id: 终止 token id
            repetition_penalty: 重复惩罚 (>1.0 抑制重复)

        返回:
            generated_ids: (B, T + new_tokens) 生成的完整序列
            state_next: (B, n_state, D) 生成后的状态
        """
        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()

        # 用输入 state 做 forward，获取首个 logits
        # 注意：生成过程中 state 保持不变，模拟训练时一个 segment 一次 forward 的行为
        result = self.forward(input_ids, state)
        next_logits = result["logits"][:, -1, :]  # (B, V)

        for _ in range(max_new_tokens):
            # 重复惩罚: 降低已出现 token 的概率
            if repetition_penalty != 1.0:
                for b in range(B):
                    prev_tokens = generated[b].unique()
                    for token_id in prev_tokens:
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty

            # 温度缩放
            next_logits = next_logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(B):
                    next_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            if token_callback is not None:
                token_callback(next_token[0, 0].item())

            # 检查终止
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 用完整已生成序列 + 原始 state 重新 forward
            # state 不在生成循环中更新，避免信息被反复覆盖
            result = self.forward(generated, state)
            next_logits = result["logits"][:, -1, :]

        # 最终用完整序列 + 原始 state 做一次 forward，得到本轮结束后的 state
        result = self.forward(generated, state)
        state_next = result["state_next"]

        return generated, state_next

    def get_trainable_params(self) -> list[nn.Parameter]:
        """收集所有可训练参数 (StatePlugin + LoRA)"""
        params = []

        # StatePlugin 参数
        for param in self.plugin.parameters():
            if param.requires_grad:
                params.append(param)

        # LoRA 参数
        lora_params = get_lora_params(self.backbone)
        params.extend(lora_params)

        return params

    def get_trainable_param_count(self) -> int:
        """可训练参数数量"""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_total_param_count(self) -> int:
        """总参数数量"""
        return sum(p.numel() for p in self.parameters())

    def state_stats(self, state: torch.Tensor) -> dict:
        """获取状态分析统计"""
        return self.plugin.get_state_stats(state)
