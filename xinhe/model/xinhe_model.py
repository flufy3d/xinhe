"""
XinheModel (v9) — 顶层模型

组合 backbone + 双 NeuralMemory(Hippocampus + Neocortex)per full-attn 层。
v9 抛弃 LoRA(backbone 全冻),write/read 在每层 fused 进行(不再有末尾全局写)。
state 是 XinheMemoryState(per-layer LayerMemState 字典)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import XinheConfig
from .backbone import BackboneBase
from .neural_memory_pair import NeuralMemoryPair, LayerMemState, XinheMemoryState


class XinheModel(nn.Module):
    """
    心核模型 v9: Backbone + ModuleDict[layer_idx → NeuralMemoryPair]

    forward:
        embed → backbone.forward_blocks(layer_hook=memory_hook(W))
             → memory_hook 内 NeuralMemoryPair forward(retrieve+store+gate+alpha)
             → 末层 hidden → lm_head → logits
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

        # Memory:每个 full-attn 层挂一个 NeuralMemoryPair
        self._hook_layer_indices = self.backbone.get_hook_layer_indices()
        self._hook_layer_set = set(self._hook_layer_indices)
        d_total = config.n_heads * config.head_dim
        # ModuleDict 的 key 必须是 str
        self.memory = nn.ModuleDict({
            str(layer_idx): NeuralMemoryPair(
                d_total=d_total,
                n_heads=config.n_heads,
                d_head=config.head_dim,
                hippo_mlp_depth=config.hippo_mlp_depth,
                hippo_mlp_expansion=config.hippo_mlp_expansion,
                neo_mlp_depth=config.neo_mlp_depth,
                neo_mlp_expansion=config.neo_mlp_expansion,
                hippo_retention=config.hippo_retention,
                hippo_base_lr=config.hippo_base_lr,
                chunk_size=config.mem_chunk_size,
                alpha_logit_init=config.alpha_logit_init,
                alpha_min_clamp=config.alpha_min_clamp,
                phase=config.phase,
                gate_entropy_lambda=config.gate_entropy_lambda,
            )
            for layer_idx in self._hook_layer_indices
        })

        # 把 Neo(普通 MLP,无 vmap+grad)也 compile。Hippo 因 inner SGD 仍走 eager,
        # 同一 pair forward 内的 Neo 单独 compile 可独立加速。
        if (getattr(config, "compile_backbone_layers", False)
                and torch.cuda.device_count() <= 1):
            try:
                for pair in self.memory.values():
                    pair.neocortex = torch.compile(
                        pair.neocortex, mode="default", fullgraph=False, dynamic=False,
                    )
                print(f"[torch.compile] 已编译 {len(self.memory)} 个 Neo 路径")
            except Exception as e:
                print(f"[torch.compile neo] 跳过(异常: {e})")

        # 投影:d_total ↔ hidden_size(NeuralMemoryPair 在 d_total 子空间工作,
        # backbone 输出是 hidden_size 维度。两者通常相等,但保留投影以备 head_dim 配置不齐)
        if d_total == config.hidden_size:
            self._d_total_in = nn.Identity()
            self._d_total_out = nn.Identity()
        else:
            self._d_total_in = nn.Linear(config.hidden_size, d_total, bias=False)
            self._d_total_out = nn.Linear(d_total, config.hidden_size, bias=False)
            nn.init.xavier_uniform_(self._d_total_in.weight)
            nn.init.zeros_(self._d_total_out.weight)  # 起步贡献为 0

        self.lm_head = self.backbone.get_lm_head()

        self._pad_token_id: Optional[int] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        mem_alpha_override: Optional[float] = None,
    ) -> dict:
        """
        参数:
            input_ids: (B, T)
            state: XinheMemoryState
            labels: (B, T) 可选
            pad_token_id: padding 屏蔽
            weights: (B, T) per-token loss 权重
            mem_alpha_override: float|None。给 learning_session 阶段 1 传 0.0 走干净路径

        返回 dict:
            logits, state_next, aux_loss, loss(labels 给定时), correct, total
        """
        self._pad_token_id = pad_token_id

        if getattr(self.config, "per_segment_checkpoint", False) and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                input_ids, state, labels, weights, mem_alpha_override,
                use_reentrant=False,
            )
        return self._forward_impl(input_ids, state, labels, weights, mem_alpha_override)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        mem_alpha_override: Optional[float],
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device
        pad_token_id = self._pad_token_id

        content_emb = self.backbone.embed(input_ids)  # (B, T, hidden_size)

        # Memory hook:每个 full-attn 层调 NeuralMemoryPair forward
        hook_layer_set = self._hook_layer_set
        # 用闭包 mut 收集 next_state + aux
        new_layers: dict[int, LayerMemState] = {}
        aux_loss_terms: list[torch.Tensor] = []

        def memory_hook(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx not in hook_layer_set:
                return hidden_states
            pair: NeuralMemoryPair = self.memory[str(layer_idx)]
            old_state = state.get(layer_idx, LayerMemState(None, None))
            x_in = self._d_total_in(hidden_states)
            x_out, new_state, aux = pair(x_in, layer_state=old_state,
                                         mem_alpha_override=mem_alpha_override)
            new_layers[layer_idx] = new_state
            aux_loss_terms.append(aux["gate_entropy_reg_loss"])
            return hidden_states + self._d_total_out(x_out - x_in)

        # 标准因果 mask
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=device, dtype=content_emb.dtype),
            diagonal=1,
        )
        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)
            pad_col = torch.zeros(B, 1, T, device=device, dtype=content_emb.dtype)
            pad_col.masked_fill_(~padding_mask.unsqueeze(1), float("-inf"))
            mask = causal.unsqueeze(0).unsqueeze(0) + pad_col.unsqueeze(2)
        else:
            mask = causal.unsqueeze(0).unsqueeze(0)

        position_ids = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)

        content_output = self.backbone.forward_blocks(
            content_emb, attention_mask=mask, position_ids=position_ids,
            layer_hook=memory_hook,
        )

        logits = self.lm_head(content_output)

        # 聚合 state next
        # 没经过 hook 的 layer_idx 沿用原 state(若有)
        merged: dict[int, LayerMemState] = dict(state.layers) if state.layers else {}
        merged.update(new_layers)
        state_next = XinheMemoryState(merged)

        aux_loss = (
            torch.stack(aux_loss_terms).sum()
            if aux_loss_terms
            else torch.zeros((), device=logits.device, dtype=logits.dtype)
        )

        result = {
            "logits": logits,
            "state_next": state_next,
            "aux_loss": aux_loss,
        }

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid_count = (shift_labels != -100).sum()
            if valid_count > 0:
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                if weights is not None:
                    shift_weights = weights[:, 1:].contiguous().view(-1).to(flat_logits.dtype)
                    safe_labels = flat_labels.clamp(min=0)
                    per_token = F.cross_entropy(flat_logits, safe_labels, reduction="none")
                    w_sum = shift_weights.sum().clamp(min=1e-8)
                    ce_loss = (per_token * shift_weights).sum() / w_sum
                else:
                    ce_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
                valid_mask = flat_labels != -100
                preds = flat_logits[valid_mask].argmax(dim=-1)
                targets = flat_labels[valid_mask]
                result["correct"] = (preds == targets).sum()
                result["total"] = valid_count
                # gate 熵正则加进总 loss
                result["loss"] = ce_loss + aux_loss
            else:
                result["loss"] = torch.tensor(0.0, device=logits.device, requires_grad=True)
                result["correct"] = torch.tensor(0, device=logits.device)
                result["total"] = torch.tensor(0, device=logits.device)

        return result

    def setup_device(self, device: torch.device):
        if torch.cuda.device_count() > 1:
            self.memory.to(device)
            if not isinstance(self._d_total_in, nn.Identity):
                self._d_total_in.to(device)
                self._d_total_out.to(device)
        else:
            self.to(device)

    def init_state(self, batch_size: int = 1) -> XinheMemoryState:
        """创建空白初始状态。NeuralMemoryPair forward 内 lazy init weights。"""
        return XinheMemoryState.init(self._hook_layer_indices)

    @torch.no_grad()
    def burn_in(self, token_ids_list: list[torch.Tensor], batch_size: int = 1) -> XinheMemoryState:
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
        state: XinheMemoryState,
        max_new_tokens: int = 256,
        temperature: float = 0.85,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
        token_callback: Optional[callable] = None,
    ) -> tuple[torch.Tensor, XinheMemoryState]:
        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()

        result = self.forward(input_ids, state)
        next_logits = result["logits"][:, -1, :].clone()
        del result

        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0:
                seen_logits = next_logits.gather(1, generated)
                penalized = torch.where(
                    seen_logits > 0,
                    seen_logits / repetition_penalty,
                    seen_logits * repetition_penalty,
                )
                next_logits.scatter_(1, generated, penalized)

            next_logits = next_logits / temperature

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(B):
                    next_logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if token_callback is not None:
                token_callback(next_token[0, 0].item())

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            del next_logits, probs
            result = self.forward(generated, state)
            next_logits = result["logits"][:, -1, :].clone()
            del result

        result = self.forward(generated, state)
        state_next = result["state_next"]

        return generated, state_next

    def get_trainable_params(self) -> list[nn.Parameter]:
        """收集所有可训练参数(memory + d_total 投影)"""
        params = [p for p in self.memory.parameters() if p.requires_grad]
        if not isinstance(self._d_total_in, nn.Identity):
            params += [p for p in self._d_total_in.parameters() if p.requires_grad]
            params += [p for p in self._d_total_out.parameters() if p.requires_grad]
        return params

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def get_total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_stats(self, state: XinheMemoryState) -> dict:
        """聚合 per-layer Hippo/Neo weight norm + alpha + gate entropy(简单平均)。"""
        stats = {"layers": {}, "n_layers": len(state.layers)}
        alphas = []
        for layer_idx, lyr in state.items():
            pair: NeuralMemoryPair = self.memory[str(layer_idx)]
            alpha = torch.sigmoid(pair.alpha_logit).item()
            alphas.append(alpha)
            stats["layers"][layer_idx] = {"alpha": alpha}
        stats["alpha_mean"] = sum(alphas) / max(len(alphas), 1)
        return stats
