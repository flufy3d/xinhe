"""
XinheModel (v7) — 顶层模型

组合 backbone + Hippocampus，实现:
- 带持久状态的 forward pass（通过 layer_hook 注入 W 读）
- 带状态的文本生成
- Burn-in 初始化

v7 大一统：短期记忆 W 由 Hippocampus 统一承载（单张量 W: (B,H,d_v,d_k)），
不再有 FactInterface / TurnInterface 分流，不再有 DualState 包装。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import XinheConfig
from .backbone import BackboneBase
from .hippocampus import Hippocampus
from .lora import inject_lora, get_lora_params


class XinheModel(nn.Module):
    """
    心核模型: Backbone + Hippocampus（短期记忆 W）

    前向传播 (v7):
        embed → backbone.forward_blocks(layer_hook=hippocampus.read_layer(W))
             → hippocampus.write_from_content(W_old, content) with γ 衰减
             → logits + W
    """

    def __init__(self, config: XinheConfig, backbone: Optional[BackboneBase] = None):
        super().__init__()
        self.config = config

        # Backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            from .qwen_backbone import QwenBackbone
            self.backbone = QwenBackbone(config)

        # 注入 LoRA
        if config.freeze_backbone and config.lora_rank > 0:
            inject_lora(
                self.backbone,
                target_modules=config.lora_target_modules,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

        # Hippocampus (v7 大一统: 短期记忆 W)
        # 只在 full attention 层前注入 state（DeltaNet 层跳过）
        self._hook_layer_indices = self.backbone.get_hook_layer_indices()
        n_hook_layers = len(self._hook_layer_indices)
        self._hook_layer_set = set(self._hook_layer_indices)
        self.hippocampus = Hippocampus(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            n_layers=n_hook_layers,
            read_scale_init=config.read_scale_init,
            beta_bias_init=config.beta_bias_init,
            delta_backend=getattr(config, "delta_backend", "auto"),
        )

        # LM head (复用 backbone 的)
        self.lm_head = self.backbone.get_lm_head()

        # forward 入口设置；_forward_impl 内引用，便于外层 checkpoint 包裹
        self._pad_token_id: Optional[int] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        state: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        带状态的前向传播。

        参数:
            input_ids: (B, T) token ids
            state: W 张量 (B, H, d_v, d_k)
            labels: (B, T) 可选，用于计算 loss
            pad_token_id: 可选，padding token id，提供时自动遮蔽 padding
            weights: (B, T) 可选，每 token 的 loss 权重 (VALUE token=5.0, 其他=1.0, pad=0.0);
                     为 None 时等价于全量均匀 loss (向后兼容)

        返回:
            dict:
                logits: (B, T, V)
                state_next: W 张量 (B, H, d_v, d_k)
                loss: scalar (如果提供了 labels)
        """
        # pad_token_id 是 Python int / None，不进 checkpoint 的 tensor 派发链
        # （use_reentrant=False 仅允许 tensor / None inputs），改成实例属性传递
        self._pad_token_id = pad_token_id

        if getattr(self.config, "per_segment_checkpoint", False) and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                input_ids, state, labels, weights,
                use_reentrant=False,
            )
        return self._forward_impl(input_ids, state, labels, weights)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        W_in: torch.Tensor,
        labels: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
    ) -> dict:
        """forward 主体，独立成方法以便外层 torch.utils.checkpoint 包裹。

        所有原本闭包捕获的非 leaf 张量（W）现在通过 W_in 入参传入；pad_token_id
        通过 self._pad_token_id 读取（Python int / None 不进 checkpoint 派发链）。
        返回 dict 内值都是 tensor 或 None，满足 checkpoint(use_reentrant=False) 契约。
        """
        B, T = input_ids.shape
        device = input_ids.device
        pad_token_id = self._pad_token_id

        # 1. 嵌入内容 token
        content_emb = self.backbone.embed(input_ids)  # (B, T, D)

        # 2. 构建 layer_hook: 只在 full attention 层前执行线性读
        hook_layer_set = self._hook_layer_set
        hook_idx_map = {layer_idx: kv_idx for kv_idx, layer_idx in enumerate(self._hook_layer_indices)}

        def state_read_hook(hidden_states, layer_idx):
            if layer_idx not in hook_layer_set:
                return hidden_states
            kv_idx = hook_idx_map[layer_idx]
            return self.hippocampus.read_layer(hidden_states, W_in, kv_idx)

        # 3. 标准因果 mask（只有 content，无 state token）
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=content_emb.dtype),
            diagonal=1,
        )  # (T, T)

        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)  # (B, T) True=真实 token
            # padding 列设为 -inf
            pad_col = torch.zeros(B, 1, T, device=device, dtype=content_emb.dtype)
            pad_col.masked_fill_(~padding_mask.unsqueeze(1), float("-inf"))
            mask = causal.unsqueeze(0).unsqueeze(0) + pad_col.unsqueeze(2)  # (B, 1, T, T)
        else:
            mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # 4. Position IDs: 简单 0..T-1
        position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)

        # 5. Backbone forward（通过 layer_hook 注入 state）
        content_output = self.backbone.forward_blocks(
            content_emb, attention_mask=mask, position_ids=position_ids,
            layer_hook=state_read_hook,
        )

        # 6. 写侧: Delta Rule + γ 衰减 更新 W（多卡时移回 plugin 设备）
        interface_device = next(self.hippocampus.parameters()).device
        content_output = content_output.to(interface_device)
        W_local = W_in.to(interface_device)
        W_next = self.hippocampus.write_from_content(W_local, content_output)

        # 7. 计算 logits (只对内容部分)
        logits = self.lm_head(content_output)  # (B, T, V)

        result = {
            "logits": logits,
            "state_next": W_next,
        }

        # 8. 计算 loss (如果有 labels)
        if labels is not None:
            # 标准 next-token prediction: logits[:, :-1] vs labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # 当所有 labels 都是 -100 时 (如非 recall segment)，loss 设为 0 避免 NaN
            valid_count = (shift_labels != -100).sum()
            if valid_count > 0:
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                if weights is not None:
                    # Per-token weighted loss:
                    #   - VALUE token weight = sample.weight_per_span（Stage0 hard 默认 5.0,Stage1 hard=3.0,soft=1.5）
                    #   - lm_only segment weight = 0.3（保 backbone 流畅但抑制 W 写入）
                    #   - false segment weight = 0（labels 全 -100,不参与梯度）
                    # weight 已在 DataLoader 层（conversation.tokenize_turn）按 train_loss 三态预乘好,这里直接用。
                    shift_weights = weights[:, 1:].contiguous().view(-1).to(flat_logits.dtype)
                    # safe labels: -100 位置换成 0, 用 weights=0 屏蔽 (避免 CE 报错)
                    safe_labels = flat_labels.clamp(min=0)
                    per_token = F.cross_entropy(flat_logits, safe_labels, reduction="none")
                    # -100 位置的 weights 已经是 0, 直接加权平均
                    w_sum = shift_weights.sum().clamp(min=1e-8)
                    loss = (per_token * shift_weights).sum() / w_sum
                else:
                    # 向后兼容: 均匀 loss
                    loss = F.cross_entropy(
                        flat_logits, flat_labels, ignore_index=-100,
                    )
                # value token 准确率 (argmax 匹配, 零额外开销)
                valid_mask = flat_labels != -100
                preds = flat_logits[valid_mask].argmax(dim=-1)
                targets = flat_labels[valid_mask]
                result["correct"] = (preds == targets).sum()
                result["total"] = valid_count
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                # 0-dim tensor 替 Python int，兼容 checkpoint(use_reentrant=False) 输出契约
                result["correct"] = torch.tensor(0, device=logits.device)
                result["total"] = torch.tensor(0, device=logits.device)

            result["loss"] = loss

        return result

    def setup_device(self, device: torch.device):
        """
        单卡: 整个模型移到 device。
        多卡: backbone 已由 device_map 分配，只移 hippocampus。
        """
        if torch.cuda.device_count() > 1:
            self.hippocampus.to(device)
        else:
            self.to(device)

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """创建空白初始状态 W: (B, H, d_v, d_k)。"""
        device = next(self.parameters()).device
        return self.hippocampus.blank_state(batch_size, device=device)

    @torch.no_grad()
    def burn_in(self, token_ids_list: list[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        """
        Burn-in: 将一段文本消化进状态（如 system prompt）。

        参数:
            token_ids_list: segment 列表，每个 (T,) 或 (1, T)
            batch_size: batch 大小

        返回:
            state: (B, H, d_v, d_k) 消化后的 W
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
            state: (B, H, d_v, d_k) 当前 W
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling
            eos_token_id: 终止 token id
            repetition_penalty: 重复惩罚 (>1.0 抑制重复)

        返回:
            generated_ids: (B, T + new_tokens) 生成的完整序列
            state_next: (B, H, d_v, d_k) 生成后的 W
        """
        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()

        # 用输入 state 做 forward，获取首个 logits
        # 注意：生成过程中 state 保持不变，模拟训练时一个 segment 一次 forward 的行为
        result = self.forward(input_ids, state)
        next_logits = result["logits"][:, -1, :].clone()   # (B, V),clone 让 result 可释放
        del result   # 显式释放 (B, T, V) ~ 数百 MB,避免 forward 内存累积导致 OOM

        for _ in range(max_new_tokens):
            # 重复惩罚:矢量化版 — 直接 gather/scatter 已出现 token 的 logits,
            # 移掉 unique() + Python token_id loop(老版本在长序列 + OOM 边缘 GPU 上是元凶)
            if repetition_penalty != 1.0:
                # generated: (B, T_cur),next_logits: (B, V)
                seen_logits = next_logits.gather(1, generated)         # (B, T_cur)
                penalized = torch.where(
                    seen_logits > 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty,
                )
                # 多次出现的 token 会被 scatter 重复写入 → 与原版 unique loop 等价
                next_logits.scatter_(1, generated, penalized)

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
            del next_logits, probs   # 释放上一轮中间张量,降低长序列累积压力
            result = self.forward(generated, state)
            next_logits = result["logits"][:, -1, :].clone()
            del result

        # 最终用完整序列 + 原始 state 做一次 forward，得到本轮结束后的 state
        result = self.forward(generated, state)
        state_next = result["state_next"]

        return generated, state_next

    def get_trainable_params(self) -> list[nn.Parameter]:
        """收集所有可训练参数 (Hippocampus + LoRA, 仅 requires_grad=True 的)"""
        params = []

        # Hippocampus 参数
        for param in self.hippocampus.parameters():
            if param.requires_grad:
                params.append(param)

        # LoRA 参数(尊重 freeze_lora 设置)
        for param in get_lora_params(self.backbone):
            if param.requires_grad:
                params.append(param)

        return params

    def get_trainable_param_count(self) -> int:
        """可训练参数数量"""
        return sum(p.numel() for p in self.get_trainable_params())

    def get_total_param_count(self) -> int:
        """总参数数量"""
        return sum(p.numel() for p in self.parameters())

    def state_stats(self, state: torch.Tensor) -> dict:
        """获取 Hippocampus 状态统计。state: W 张量 (B,H,d_v,d_k)。"""
        return self.hippocampus.get_state_stats(state)
