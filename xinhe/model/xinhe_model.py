"""
XinheModel (v9.5) — 顶层模型

组合 backbone + 双 NeuralMemory(Hippocampus + Neocortex)per full-attn 层。
v9.5 paper-faithful 重构:
  - 删 input-level persistent_mem(soft prompt),改 per-layer K/V(每个 full_attention 层独立,
    在 attention 内拼接,paper Titans MAC Eq 11-13 形态)
  - 删 past_snapshots(跨 turn hidden carry,paper 没有这个,跨 turn 走 NM weights 演化)
  - 保留 fresh_mem(NM 输出承载点,paper M_t(K) 的简化版固定 N_m 位置)
  - 删 mac_inject_logit gating(实测 inject 全程 = 1.0,直接 += mem_out 即可,paper 形态)
  - 恢复 LoRA on q/k/v/o:frozen backbone 适配 MAC 输入的根因修复(MAC=producer / LoRA=consumer)
state 是 XinheMemoryState(per-layer LayerMemState 字典,无 mem_snapshots)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from cut_cross_entropy import linear_cross_entropy

from .config import XinheConfig
from .backbone import BackboneBase
from .neural_memory_pair import NeuralMemoryPair, LayerMemState, XinheMemoryState


@torch.no_grad()
def _chunked_argmax(h: torch.Tensor, W: torch.Tensor, v_chunk: int = 4096) -> torch.Tensor:
    """argmax(h @ W.T) without materializing (N, V). h:(N,D), W:(V,D) -> (N,)"""
    N = h.shape[0]
    V = W.shape[0]
    running_max = torch.full((N,), float("-inf"), device=h.device, dtype=torch.float32)
    running_argmax = torch.zeros((N,), dtype=torch.long, device=h.device)
    for v_start in range(0, V, v_chunk):
        v_end = min(v_start + v_chunk, V)
        logits_chunk = (h @ W[v_start:v_end].T).float()
        chunk_max, chunk_argmax = logits_chunk.max(dim=-1)
        update = chunk_max > running_max
        running_max = torch.where(update, chunk_max, running_max)
        running_argmax = torch.where(update, chunk_argmax + v_start, running_argmax)
    return running_argmax


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

        # LoRA 注入(在 backbone freeze 后、per-layer K/V wrap 前):
        # frozen q/k/v 投影看不懂 MAC 引入的 OOD prefix(per-layer K_pers / fresh_mem),
        # LoRA 是低秩旁路,与 MAC 是 producer/consumer 协同(MAC 放 prefix,LoRA 学怎么读)。
        # 零初始化 → 启动时增量 0 不破坏 frozen backbone。
        if config.freeze_backbone and getattr(config, "lora_rank", 0) > 0:
            from .lora import inject_lora
            replaced = inject_lora(
                self.backbone.model,  # HF AutoModel(QwenBackbone 内的 self.model)
                target_modules=config.lora_target_modules,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            print(f"[LoRA] 注入 {len(replaced)} 个 Linear "
                  f"(rank={config.lora_rank}, alpha={config.lora_alpha}, "
                  f"target={config.lora_target_modules})")

        # per-layer K/V persistent memory(paper Titans MAC 严格形态):
        # 在 LoRA 注入之后包装 self_attn,wrapper 复用 LoRA 后的 q/k/v/o_proj。
        # 只对 QwenBackbone 有效;其他 backbone 没这个方法时跳过(MockBackbone 用 noop)。
        n_per_layer = int(getattr(config, "n_persistent_per_layer", 0))
        if n_per_layer > 0 and hasattr(self.backbone, "wrap_persistent_kv"):
            self.backbone.wrap_persistent_kv(n_per_layer)

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

        # Hippo NM compile:lazy trace 在 forward 首次入口(配 Dynamo 三连让 outer loop 稳态)
        hippo_compile_on = sum(
            1 for pair in self.memory.values()
            if getattr(pair.hippocampus, "use_compile_chunk_loop", False)
        )
        if hippo_compile_on > 0:
            print(f"[torch.compile] 已编译 {hippo_compile_on} 个 Hippo 路径")

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

        # MAC: fresh_mem (NM 输出承载) — paper M_t(K) 简化为 N_m 个固定位置
        # 每 turn 末插 N_mem 个 fresh 位置,NM forward 把 mem_out 写到这 N_mem 位置
        # 注:input-level persistent_mem (soft prompt) 已删,改用 per-layer K/V(在 attention 内拼接)
        # 注:past_snapshots (跨 turn hidden carry) 已删,跨 turn 走 NM weights 演化(paper way)
        self.n_mem_tokens = int(getattr(config, "n_mem_tokens", 0))
        if self.n_mem_tokens > 0:
            self.mem_token_init = nn.Parameter(
                torch.randn(self.n_mem_tokens, config.hidden_size)
                * (config.hidden_size ** -0.5)
            )
        else:
            self.register_parameter("mem_token_init", None)

        self._pad_token_id: Optional[int] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        mem_alpha_override: Optional[float] = None,
        compute_logits: bool = True,
    ) -> dict:
        """
        参数:
            input_ids: (B, T)
            state: XinheMemoryState
            labels: (B, T) 可选
            pad_token_id: padding 屏蔽
            weights: (B, T) per-token loss 权重
            mem_alpha_override: float|None。给 learning_session 阶段 1 传 0.0 走干净路径
            compute_logits: True 时材化 (B,T,V) logits(eval/tests 路径);
                            False 时走 cut_cross_entropy 快路径,result["logits"]=None,
                            correct/total 用 chunked argmax 算(省 ~3GB / 4 turns)。

        返回 dict:
            logits, state_next, aux_loss, loss(labels 给定时), correct, total
        """
        self._pad_token_id = pad_token_id

        if getattr(self.config, "per_segment_checkpoint", False) and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                input_ids, state, labels, weights, mem_alpha_override, compute_logits,
                use_reentrant=False,
            )
        return self._forward_impl(input_ids, state, labels, weights, mem_alpha_override, compute_logits)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        mem_alpha_override: Optional[float],
        compute_logits: bool = True,
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device
        pad_token_id = self._pad_token_id

        content_emb = self.backbone.embed(input_ids)  # (B, T, hidden_size)

        # v9.5 paper-faithful 序列布局(从前到后):
        #   [fresh_mem (N_m), real (T)]
        # input-level persistent_mem 已删 → 改 per-layer K/V(每个 full_attention 层 attention 内拼接)
        # past_snapshots 已删 → 跨 turn carry 走 NM weights 演化(paper way)
        # fresh_mem 仍在 real 前面 → 因果 mask 让 real 能 attend 到 fresh_mem
        # NeuralMemory 在每层只把 mem_out 写到 fresh_mem 位置,其他位置 hidden 不动
        # → memory 信息通过 attention(real → fresh_mem)进入 real 表示
        N_m = self.n_mem_tokens if self.mem_token_init is not None else 0

        if N_m > 0:
            fresh = self.mem_token_init.to(content_emb.dtype).unsqueeze(0).expand(B, -1, -1)
            content_emb = torch.cat([fresh, content_emb], dim=1)

        # 关键索引(loss 切片 + 每层 NM 写入定位用)
        fresh_start = 0
        fresh_end = N_m
        real_start = fresh_end
        real_end = real_start + T

        # Memory hook(pure MAC):
        #   1. NM forward:得 mem_out (full sequence,所有位置都有)
        #   2. 只把 mem_out[fresh_start:fresh_end] 加到 hidden 同位置
        #   3. real 位置不变 — 它们通过 attention 自己去 query fresh_mem
        # 注意 NM 仍读全序列(包括 fresh_mem 自己的 init),所以 fresh_mem 位置的 mem_out
        #   = "NM 对当前 query=mem_init 的回应",物理意义:memory 把检索结果写进笔记本
        hook_layer_set = self._hook_layer_set
        new_layers: dict[int, LayerMemState] = {}
        aux_loss_terms: list[torch.Tensor] = []

        def memory_hook(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx not in hook_layer_set:
                return hidden_states
            pair: NeuralMemoryPair = self.memory[str(layer_idx)]
            old_state = state.get(layer_idx, LayerMemState(None, None))
            x_in = self._d_total_in(hidden_states)
            _x_out, new_state, aux = pair(x_in, layer_state=old_state,
                                          mem_alpha_override=mem_alpha_override)
            new_layers[layer_idx] = new_state
            aux_loss_terms.append(aux["gate_entropy_reg_loss"])
            if N_m == 0:
                return hidden_states  # 没启 mem tokens → memory 通路无写入点(纯 baseline)
            mem_proj = self._d_total_out(aux["mem_out"])  # (B, T_ext, hidden)
            # paper-faithful:直接 += mem_out 到 fresh_mem 位置(实测 sigmoid gating 全程≈1)
            # learning_session phase 1 走 mem_alpha_override=0.0 屏蔽注入(NM 内部 alpha 也=0)
            fresh_proj = mem_proj[:, fresh_start:fresh_end, :]
            if mem_alpha_override is not None:
                fresh_proj = float(mem_alpha_override) * fresh_proj
            delta = torch.zeros_like(hidden_states)
            delta[:, fresh_start:fresh_end, :] = fresh_proj
            return hidden_states + delta

        # 标准因果 mask(只覆盖 fresh_mem 前缀 + real 段)
        T_ext = content_emb.shape[1]
        causal = torch.triu(
            torch.full((T_ext, T_ext), float("-inf"), device=device, dtype=content_emb.dtype),
            diagonal=1,
        )
        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)             # (B, T)
            # 顺序: [fresh_mem(N_m) | real(T)]
            # fresh_mem 段永远 valid;real 段按 input_ids 判 pad
            pre_valid = torch.ones(B, real_start, dtype=torch.bool, device=device)
            padding_mask = torch.cat([pre_valid, padding_mask], dim=1)
            pad_col = torch.zeros(B, 1, T_ext, device=device, dtype=content_emb.dtype)
            pad_col.masked_fill_(~padding_mask.unsqueeze(1), float("-inf"))
            mask = causal.unsqueeze(0).unsqueeze(0) + pad_col.unsqueeze(2)
        else:
            mask = causal.unsqueeze(0).unsqueeze(0)

        position_ids = torch.arange(T_ext, dtype=torch.long, device=device).unsqueeze(0)

        content_output = self.backbone.forward_blocks(
            content_emb, attention_mask=mask, position_ids=position_ids,
            layer_hook=memory_hook,
        )

        # 切回真实 token 段(loss 只在 T 上)
        content_output = content_output[:, real_start:real_end, :]

        # 聚合 state next:per-layer NM state(跨 turn 通过 NM weights 演化承载,无 mem_snapshots)
        # 没经过 hook 的 layer_idx 沿用原 state(若有)
        merged: dict[int, LayerMemState] = dict(state.layers) if state.layers else {}
        merged.update(new_layers)
        state_next = XinheMemoryState(merged)

        ref_dtype = content_output.dtype
        ref_device = content_output.device
        aux_loss = (
            torch.stack(aux_loss_terms).sum()
            if aux_loss_terms
            else torch.zeros((), device=ref_device, dtype=ref_dtype)
        )

        if compute_logits:
            # eval/tests 路径:材化 (B,T,V) logits(由调用方读 result["logits"] argmax)
            logits = self.lm_head(content_output)
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
                    result["loss"] = ce_loss + aux_loss
                else:
                    result["loss"] = torch.tensor(0.0, device=ref_device, requires_grad=True)
                    result["correct"] = torch.tensor(0, device=ref_device)
                    result["total"] = torch.tensor(0, device=ref_device)
            return result

        # 训练快路径:cut_cross_entropy 不材化 (B,T,V) logits → 省 ~3GB / 4 turns
        result = {
            "logits": None,
            "state_next": state_next,
            "aux_loss": aux_loss,
        }
        if labels is None:
            return result

        shift_h = content_output[:, :-1, :].contiguous()           # (B, T-1, D)
        shift_labels = labels[:, 1:].contiguous()                  # (B, T-1)
        flat_h = shift_h.view(-1, shift_h.size(-1))                # (B*(T-1), D)
        flat_labels = shift_labels.view(-1)                        # (B*(T-1),)
        valid_mask = flat_labels != -100
        valid_count = valid_mask.sum()

        if valid_count == 0:
            result["loss"] = torch.tensor(0.0, device=ref_device, requires_grad=True)
            result["correct"] = torch.tensor(0, device=ref_device)
            result["total"] = torch.tensor(0, device=ref_device)
            return result

        lm_w = self.lm_head.weight  # (V, D)
        if weights is not None:
            per_token = linear_cross_entropy(
                flat_h, lm_w, flat_labels,
                ignore_index=-100, reduction="none",
            )  # (B*(T-1),) — ignored 位置 cut CE 自动给 0
            shift_weights = weights[:, 1:].contiguous().view(-1).to(per_token.dtype)
            shift_weights = shift_weights * valid_mask.to(shift_weights.dtype)
            w_sum = shift_weights.sum().clamp(min=1e-8)
            ce_loss = (per_token * shift_weights).sum() / w_sum
        else:
            ce_loss = linear_cross_entropy(
                flat_h, lm_w, flat_labels,
                ignore_index=-100, reduction="mean",
            )

        # chunked argmax(no_grad,不材化 (N, V))
        valid_h = flat_h[valid_mask].detach()
        preds = _chunked_argmax(valid_h, lm_w)
        targets = flat_labels[valid_mask]
        result["correct"] = (preds == targets).sum()
        result["total"] = valid_count
        result["loss"] = ce_loss + aux_loss
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
        """收集所有可训练参数(memory + d_total 投影 + MAC 参数 + LoRA + per-layer K/V)。

        策略:用 backbone.parameters() requires_grad 过滤,自动覆盖 LoRA 注入的 lora_A/B
        和后续 per-layer K/V 包装的 K_pers/V_pers(它们都在 backbone 子模块内)。
        """
        params = [p for p in self.memory.parameters() if p.requires_grad]
        if not isinstance(self._d_total_in, nn.Identity):
            params += [p for p in self._d_total_in.parameters() if p.requires_grad]
            params += [p for p in self._d_total_out.parameters() if p.requires_grad]
        if self.mem_token_init is not None and self.mem_token_init.requires_grad:
            params.append(self.mem_token_init)
        # LoRA 注入的 lora_A/B + 后续 per-layer K/V 的 K_pers/V_pers 都挂在 backbone 内
        params += [p for p in self.backbone.parameters() if p.requires_grad]
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
