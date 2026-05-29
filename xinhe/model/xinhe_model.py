"""
XinheModel — 顶层模型(单全局 QueryHead + HippoDelta 架构)

组合 backbone + QueryHead + 单全局 HippoDelta + MAC-R/MAL 注入。
forward:
  1. embedding 低层 h_pre → QueryHead → q
  2. retrieve(M_prev, q) → mem_out  ← 唯一一次 read
  3. MAC-R: W_mac(mem_out) 拼输入序列前缀;MAL: W_mal(mem_out) 中后层残差
  4. write: global_write_layer 用 real 段 hidden store → M_next
state 是 XinheMemoryState(per-layer LayerMemState 字典),只在 _global_write_idx 一处持有 HippoDeltaState。
backbone addons:LoRA(qkvo) + per-layer K/V(可选,Plan B 关停)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from cut_cross_entropy import linear_cross_entropy

from .config import XinheConfig
from .backbone import BackboneBase
from .neural_memory_pair import LayerMemState, XinheMemoryState


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
    心核模型(单全局 QueryHead + HippoDelta):Backbone + 单 HippoDelta + QueryHead + 注入投影。

    forward 全程走 `_forward_query_head`:
        embed → QueryHead → retrieve(M_prev) → MAC-R 前缀注入
            → backbone.forward_blocks(MAL@mid + write@global_write_idx)
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

        # 单全局记忆架构:不再 per-full-attn-layer 挂 Pair,只在 _global_write_idx 一处 write,
        # _forward_query_head 内一次 read。hook_layer_indices 仅用于解析 global_write_layer 物理位置。
        self._hook_layer_indices = self.backbone.get_hook_layer_indices()
        self._hook_layer_set = set(self._hook_layer_indices)
        d_total = config.n_heads * config.head_dim

        # 投影:d_total ↔ hidden_size(NM 在 d_total 子空间工作,backbone 输出是 hidden_size 维度。
        # 两者通常相等,但保留投影以备 head_dim 配置不齐)
        if d_total == config.hidden_size:
            self._d_total_in = nn.Identity()
            self._d_total_out = nn.Identity()
        else:
            self._d_total_in = nn.Linear(config.hidden_size, d_total, bias=False)
            self._d_total_out = nn.Linear(d_total, config.hidden_size, bias=False)
            nn.init.xavier_uniform_(self._d_total_in.weight)
            nn.init.zeros_(self._d_total_out.weight)  # 起步贡献为 0

        self.lm_head = self.backbone.get_lm_head()

        # === QueryHead 单全局记忆(目标架构;唯一 forward 路径)===
        # use_query_head 已单值化为 True,留作 ckpt/yaml 兼容标记。
        from .query_head import QueryHead
        from .neural_memory_pair import AdaptiveRMSNorm
        from .hippo_delta import HippoDelta
        self.use_query_head = True
        self.n_query = int(getattr(config, "n_query", 16))
        d_key = int(getattr(config, "d_key", 256))
        d_value = int(getattr(config, "d_value", 128))
        self.global_hippo = HippoDelta(
            d_model=d_total, d_key=d_key, d_value=d_value, n_heads=config.n_heads,
            tau_k_init=float(getattr(config, "tau_k_init", 1.0)),
            beta_bias_init=float(getattr(config, "beta_bias_init", 0.0)),
            gated=bool(getattr(config, "gated_delta", False)),
            delta_backend=str(getattr(config, "delta_backend", "auto")),
            spectral_norm_cap=float(getattr(config, "spectral_norm_cap", 10.0)),
        )
        mem_dim = d_value
        self.query_head = QueryHead(config.hidden_size, d_key, self.n_query)
        # 注入投影:mem_out(mem_dim=d_value) → hidden
        self.W_mac = nn.Linear(mem_dim, config.hidden_size, bias=False)
        self.W_mal = nn.Linear(mem_dim, config.hidden_size, bias=False)
        # W_mac 非零:MAC-R 是主表达通路,零 init 切断 QueryHead 梯度(dLoss/dmem=dLoss/dmac·W_mac=0)
        nn.init.xavier_uniform_(self.W_mac.weight)
        # W_mal 零:残差零起步不扰 frozen backbone(配 α=σ(-3)≈0.05);梯度仍会从 0 长出
        nn.init.zeros_(self.W_mal.weight)
        self.mal_alpha_logit = nn.Parameter(
            torch.tensor(float(getattr(config, "mal_alpha_init", -3.0)))
        )
        self.global_mem_rmsnorm = AdaptiveRMSNorm(mem_dim)
        n_layers = self.backbone.get_num_layers()
        gw = int(getattr(config, "global_write_layer", -1))
        self._global_write_idx = self._hook_layer_indices[gw] if gw < 0 else gw
        assert self._global_write_idx in self._hook_layer_set, \
            f"global_write_layer 解析到 {self._global_write_idx},不是 full_attention 层"
        mal = int(getattr(config, "mal_inject_layer", -3))
        self._mal_target_idx = (n_layers + mal) if mal < 0 else mal
        # 自检:MAL 目标层类型(Qwen3.5 是 full/linear 混合;落在 linear 信号可能被 token-mix 稀释,
        # forward_blocks 的 hook 仍对每层都调用,只是观察用)
        try:
            _layers = self.backbone.model.model.layers
            self._mal_target_layer_type = getattr(_layers[self._mal_target_idx], "layer_type", "unknown")
        except Exception:
            self._mal_target_layer_type = "unknown"
        print(f"[QueryHead 单全局/delta] write@L{self._global_write_idx} "
              f"MAL@L{self._mal_target_idx}({self._mal_target_layer_type}) "
              f"n_query={self.n_query} d_key={d_key} mem_dim={mem_dim}")

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
        *,
        mem_mac_override: Optional[float] = None,
        mem_mal_override: Optional[float] = None,
    ) -> dict:
        """
        参数:
            input_ids: (B, T)
            state: XinheMemoryState
            labels: (B, T) 可选
            pad_token_id: padding 屏蔽
            weights: (B, T) per-token loss 权重
            mem_alpha_override: float|None。NM-zero(0.0)同时关 MAC-R 与 MAL
            compute_logits: True 时材化 (B,T,V) logits(eval/tests 路径);
                            False 时走 cut_cross_entropy 快路径,result["logits"]=None,
                            correct/total 用 chunked argmax 算(省 ~3GB / 4 turns)。
            mem_mac_override: 仅 probe/debug 用。独立控制 MAC-R 前缀 alpha;非 None 时
                              覆盖 mem_alpha_override 对 MAC-R 的作用。
            mem_mal_override: 仅 probe/debug 用。独立控制 MAL 残差 alpha;非 None 时
                              覆盖 mem_alpha_override 对 MAL 的作用。
                              两者 default None → 现有 trainer/eval 路径完全不受影响。

        返回 dict:
            logits, state_next, aux_loss, loss(labels 给定时), correct, total
        """
        self._pad_token_id = pad_token_id

        if getattr(self.config, "per_segment_checkpoint", False) and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._forward_impl,
                input_ids, state, labels, weights, mem_alpha_override, compute_logits,
                mem_mac_override, mem_mal_override,
                use_reentrant=False,
            )
        return self._forward_impl(
            input_ids, state, labels, weights, mem_alpha_override, compute_logits,
            mem_mac_override, mem_mal_override,
        )

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        mem_alpha_override: Optional[float],
        compute_logits: bool = True,
        mem_mac_override: Optional[float] = None,
        mem_mal_override: Optional[float] = None,
    ) -> dict:
        # 已单值化:统一走 QueryHead 单全局路径
        return self._forward_query_head(
            input_ids, state, labels, weights, mem_alpha_override, compute_logits,
            mem_mac_override, mem_mal_override,
        )

    def _global_read(self, q, old_hippo):
        """单全局 read (delta):HippoDelta.retrieve。返回 (B,n_q,d_value)。"""
        M = old_hippo.M if old_hippo is not None else None
        return self.global_hippo.retrieve(M, q)

    def _global_write_step(self, x_write, old_hippo):
        """单全局 write (delta):HippoDelta.write。返回新 hippo state。"""
        return self.global_hippo.write(x_write, old_hippo)

    def _forward_query_head(
        self,
        input_ids: torch.Tensor,
        state: XinheMemoryState,
        labels: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        mem_alpha_override: Optional[float],
        compute_logits: bool = True,
        mem_mac_override: Optional[float] = None,
        mem_mal_override: Optional[float] = None,
    ) -> dict:
        """单全局 QueryHead forward(唯一 forward 路径):
          1. embedding 低层 h_pre → QueryHead → q
          2. global_hippo.retrieve(M_prev, q) → mem_out  ← 唯一一次 read
          3. MAC-R: W_mac(mem_out) 拼输入序列前缀;MAL: W_mal(mem_out) 中后层残差(同一 mem_out)
          4. write: global_write 层用 real 段 hidden store → M_next(供下一 turn read)
        mem_alpha_override=0.0(NM-zero)同时关 MAC-R 与 MAL,验证 backbone 没偷记。
        mem_mac_override / mem_mal_override(probe 用):独立 ablate MAC vs MAL,非 None 时
        覆盖 mem_alpha_override 对该通路的作用。
        """
        B, T = input_ids.shape
        device = input_ids.device
        pad_token_id = self._pad_token_id
        content_emb = self.backbone.embed(input_ids)            # (B, T, hidden)

        # 1. 低层 h_pre 的"最后一个非 pad token" → q(右 pad 时 [:, -1] 会取到 pad)
        if pad_token_id is not None:
            last_idx = (input_ids != pad_token_id).long().sum(dim=1).clamp(min=1) - 1
            h_last = content_emb[torch.arange(B, device=device), last_idx]   # (B, hidden)
        else:
            h_last = content_emb[:, -1]
        q = self.query_head(h_last)                             # (B, n_q, d_total)

        # 2. 单次 read:retrieve_only(M_prev)。Phase 2 pause mix_gate → mem_out = rmsnorm(r_h)
        gkey = self._global_write_idx
        old_layer = (state.get(gkey, LayerMemState(None, None))
                     if state is not None else LayerMemState(None, None))
        old_hippo = old_layer.hippo
        r_h = self._global_read(q, old_hippo)               # (B, n_q, d_value)
        mem_out = self.global_mem_rmsnorm(r_h)

        # 3. MAC-R 前缀(动态投影,QueryHead retrieve 的 mem_out → hidden)
        mac_tokens = self.W_mac(mem_out)                        # (B, n_q, hidden)
        # MAC-R alpha 路由:mem_mac_override(probe)优先;退化到 mem_alpha_override(NM-zero)
        mac_alpha_val = mem_mac_override if mem_mac_override is not None else mem_alpha_override
        if mac_alpha_val is not None:
            mac_tokens = float(mac_alpha_val) * mac_tokens
        N_m = self.n_query
        content_emb = torch.cat([mac_tokens.to(content_emb.dtype), content_emb], dim=1)
        fresh_start, fresh_end, real_start, real_end = 0, N_m, N_m, N_m + T

        new_layers: dict[int, LayerMemState] = {}
        aux_loss_terms: list[torch.Tensor] = []
        lambda_div = float(getattr(self.config, "lambda_div", 0.0))
        if lambda_div > 0:
            from .query_head import QueryHead as _QH
            # L_q-div = -λ·diversity(diversity↑ → loss↓,逼不同 turn 的 q 分离,防 mode collapse)
            aux_loss_terms.append(-lambda_div * _QH.cosine_diversity(q))
        # nm_aux(简化版,单全局):mac_tokens 经 lm_head 预测当前 turn value 首 token
        # 逼 mem 通路必须含 value 信息(否则 mac_tokens 经 lm_head 不会预测对 → loss 大)
        # 任何 override(NM-zero 或 probe MAC/MAL 独立 ablation)都跳过 nm_aux(避免 ablation 测出虚高数字)
        _any_override = (mem_alpha_override is not None
                         or mem_mac_override is not None
                         or mem_mal_override is not None)
        nm_aux_weight = float(getattr(self.config, "nm_aux_weight", 0.0))
        if (nm_aux_weight > 0 and labels is not None and weights is not None
                and not _any_override):
            vmask = (weights > 0.5)                                     # (B, T) value-span 位置
            has_v = vmask.any(dim=1)                                    # (B,)
            if has_v.any():
                first_pos = vmask.long().argmax(dim=1)                  # (B,) 首个 value 位置
                v_tok = labels.gather(1, first_pos.unsqueeze(1)).squeeze(1)  # (B,) value 首 token id
                n_q = mac_tokens.shape[1]
                tgt = v_tok.unsqueeze(1).expand(-1, n_q).reshape(-1)
                tgt = torch.where(
                    has_v.unsqueeze(1).expand(-1, n_q).reshape(-1),
                    tgt, torch.full_like(tgt, -100),
                )
                mac_flat = mac_tokens.reshape(-1, mac_tokens.size(-1)).to(torch.bfloat16)
                nm_ce = linear_cross_entropy(
                    mac_flat, self.lm_head.weight, tgt,
                    ignore_index=-100, reduction="mean",
                )
                aux_loss_terms.append(nm_aux_weight * nm_ce)
        mal_target = self._mal_target_idx

        def memory_hook(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            # MAL:中后层残差注入(mem_out 池化广播到 real 段;与 MAC-R 同源 mem_out,不抢 credit)
            if layer_idx == mal_target:
                # MAL alpha 路由:mem_mal_override(probe)优先;退化到 mem_alpha_override(NM-zero);
                # 二者皆 None 时走学习参数 σ(mal_alpha_logit)
                mal_alpha_val = mem_mal_override if mem_mal_override is not None else mem_alpha_override
                if mal_alpha_val is None:
                    alpha = torch.sigmoid(self.mal_alpha_logit)
                else:
                    alpha = float(mal_alpha_val)
                mal_vec = self.W_mal(mem_out).mean(dim=1, keepdim=True)      # (B, 1, hidden)
                delta = torch.zeros_like(hidden_states)
                delta[:, real_start:real_end, :] = alpha * mal_vec
                hidden_states = hidden_states + delta
            # write:单全局 M 更新(只在 global_write 层 = full_attention 物理层)
            if layer_idx == gkey:
                x_write = self._d_total_in(hidden_states[:, real_start:real_end, :])
                new_hippo = self._global_write_step(x_write, old_hippo)
                new_layers[gkey] = LayerMemState(hippo=new_hippo, neo=None)
            return hidden_states

        # ===== mask / forward / slice / loss 与旧路径同口径 =====
        T_ext = content_emb.shape[1]
        causal = torch.triu(
            torch.full((T_ext, T_ext), float("-inf"), device=device, dtype=content_emb.dtype),
            diagonal=1,
        )
        if pad_token_id is not None:
            padding_mask = (input_ids != pad_token_id)
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
        content_output = content_output[:, real_start:real_end, :]

        merged = dict(state.layers) if (state is not None and state.layers) else {}
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
            logits = self.lm_head(content_output)
            result = {"logits": logits, "state_next": state_next, "aux_loss": aux_loss}
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

        result = {"logits": None, "state_next": state_next, "aux_loss": aux_loss}
        if labels is None:
            return result
        shift_h = content_output[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_h = shift_h.view(-1, shift_h.size(-1))
        flat_labels = shift_labels.view(-1)
        valid_mask = flat_labels != -100
        valid_count = valid_mask.sum()
        if valid_count == 0:
            result["loss"] = torch.tensor(0.0, device=ref_device, requires_grad=True)
            result["correct"] = torch.tensor(0, device=ref_device)
            result["total"] = torch.tensor(0, device=ref_device)
            return result
        lm_w = self.lm_head.weight
        if weights is not None:
            per_token = linear_cross_entropy(flat_h, lm_w, flat_labels, ignore_index=-100, reduction="none")
            shift_weights = weights[:, 1:].contiguous().view(-1).to(per_token.dtype)
            shift_weights = shift_weights * valid_mask.to(shift_weights.dtype)
            w_sum = shift_weights.sum().clamp(min=1e-8)
            ce_loss = (per_token * shift_weights).sum() / w_sum
        else:
            ce_loss = linear_cross_entropy(flat_h, lm_w, flat_labels, ignore_index=-100, reduction="mean")
        valid_h = flat_h[valid_mask].detach()
        preds = _chunked_argmax(valid_h, lm_w)
        targets = flat_labels[valid_mask]
        result["correct"] = (preds == targets).sum()
        result["total"] = valid_count
        result["loss"] = ce_loss + aux_loss
        return result

    def setup_device(self, device: torch.device):
        # 单全局架构没有 per-layer memory ModuleDict,整 module 一并搬到设备
        self.to(device)

    def init_state(self, batch_size: int = 1) -> XinheMemoryState:
        """创建空白初始状态(HippoDelta 在 first write 内 lazy init M)。"""
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
        pad_token_id: Optional[int] = None,
    ) -> tuple[torch.Tensor, XinheMemoryState]:
        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()
        if pad_token_id is not None:
            self._pad_token_id = pad_token_id

        result = self.forward(input_ids, state, pad_token_id=pad_token_id)
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

        # 关键:state 演化必须用 turn_max_tokens padded 输入,匹配训练分布。
        # 变长 chat 输入下 delta 写入分布与训练不一致,
        # 实测变长 chat 跨 turn 召回 6%,padded 后 100%(scripts/_scratch probe 验证)。
        seg_len = getattr(self.config, "turn_max_tokens", 128)
        pad_id = getattr(self, "_pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        B_, T_gen = generated.shape
        if T_gen < seg_len:
            pad_tensor = torch.full(
                (B_, seg_len - T_gen), int(pad_id),
                dtype=generated.dtype, device=generated.device,
            )
            generated_for_state = torch.cat([generated, pad_tensor], dim=1)
        elif T_gen > seg_len:
            generated_for_state = generated[:, :seg_len]
        else:
            generated_for_state = generated
        result = self.forward(generated_for_state, state, pad_token_id=int(pad_id))
        state_next = result["state_next"]

        return generated, state_next

    def get_trainable_params(self) -> list[nn.Parameter]:
        """收集所有可训练参数:单全局 QueryHead/HippoDelta/W_mac/W_mal/MAL α/d_total 投影 + LoRA + per-layer K/V。

        策略:用 backbone.parameters() requires_grad 过滤,自动覆盖 LoRA 注入的 lora_A/B
        和后续 per-layer K/V 包装的 K_pers/V_pers(它们都在 backbone 子模块内)。
        """
        params: list[nn.Parameter] = []
        if not isinstance(self._d_total_in, nn.Identity):
            params += [p for p in self._d_total_in.parameters() if p.requires_grad]
            params += [p for p in self._d_total_out.parameters() if p.requires_grad]
        # 单全局:QueryHead + HippoDelta + 注入投影(W_mac/W_mal) + MAL α
        for mod in (self.query_head, self.global_hippo, self.W_mac, self.W_mal, self.global_mem_rmsnorm):
            params += [p for p in mod.parameters() if p.requires_grad]
        if self.mal_alpha_logit.requires_grad:
            params.append(self.mal_alpha_logit)
        # LoRA 注入的 lora_A/B + 后续 per-layer K/V 的 K_pers/V_pers 都挂在 backbone 内
        params += [p for p in self.backbone.parameters() if p.requires_grad]
        return params

    def get_trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())

    def get_total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_stats(self, state: XinheMemoryState) -> dict:
        """聚合 per-layer 状态简要(目前仅层数)。"""
        return {"layers": {}, "n_layers": len(state.layers)}
