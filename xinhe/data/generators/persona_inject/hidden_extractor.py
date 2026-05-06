"""冻结 Qwen backbone 跑 forward,取每个块**末 token** 的指定层 hidden。

设计要点:
  - 与 QwenBackbone 一致用 AutoModelForCausalLM 加载;forward 跑 model.model 子模块
    (纯文本 Qwen3_5Model,跳过 lm_head 与 vision encoder)
  - output_hidden_states=True → tuple[T+1] of (B, T, D);idx 0 是 embed 输出
  - 块定长 block_size,直接 [:, -1, :] 取末 token,不需要 attention_mask
  - bf16 + no_grad,显存极省

注意:HuggingFace 的 hidden_states[i+1] = layer i 的输出(经过该 block 完整的
attention+mlp+residual,但**未过最后的 final_layernorm**)。这是协议想要的"块末尾隐状态"。
"""
from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM

from xinhe.data.generators.persona_inject.block_splitter import Block


def extract_hidden_states(
    blocks: list[Block],
    *,
    backbone_model_path: str,
    layer_indices: Sequence[int],
    batch_size: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    progress_every: int = 50,
) -> torch.Tensor:
    """跑冻结 backbone,返回 Tensor[N_blocks, len(layer_indices), hidden_dim] in bf16(CPU)。

    Args:
        blocks: split_to_blocks 产出。所有块必须等长(block_size)。
        backbone_model_path: 与 XinheConfig.backbone_model_path 一致。
        layer_indices: 要保存的层 idx 列表(0-based,layer 0 = 第 1 个 transformer block)。
                       注意:hidden_states[idx + 1] 才是 layer idx 的输出(idx 0 是 embed)。
        batch_size: forward batch,bf16 下 0.8B 跑 8 ~= 显存 4GB。
        device: cuda / cpu。
        dtype: 模型加载与计算的 dtype,默认 bf16。
        progress_every: 每多少 batch 打一次进度。

    Returns:
        Tensor[N, len(layer_indices), hidden_dim],dtype = bf16,device = cpu(节省显存)。
    """
    if not blocks:
        raise ValueError("blocks 为空,无可提取数据")

    block_size = len(blocks[0].token_ids)
    for b in blocks:
        if len(b.token_ids) != block_size:
            raise ValueError(
                f"块长度不一致: block {b.block_id} 长度 {len(b.token_ids)},预期 {block_size}"
            )

    # 与 QwenBackbone 一致:AutoModelForCausalLM,纯文本 forward 走 model.model 子树
    # (跳过 lm_head 节省显存;Qwen3.5 多模态权重存在但不喂图像不会触发 vision)
    print(f"  [extractor] 加载 backbone: {backbone_model_path} (dtype={dtype})")
    full = AutoModelForCausalLM.from_pretrained(
        backbone_model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    full.eval()
    for p in full.parameters():
        p.requires_grad = False
    full.to(device)
    text_model = full.model     # Qwen3_5Model(text-only),与 QwenBackbone.embed/forward_blocks 同源

    cfg = text_model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", None)
                   or cfg.text_config.num_hidden_layers)
    hidden_size = int(getattr(cfg, "hidden_size", None)
                      or cfg.text_config.hidden_size)

    # 校验 layer_indices 不越界
    layer_indices = list(layer_indices)
    if max(layer_indices) >= n_layers or min(layer_indices) < 0:
        raise ValueError(
            f"layer_indices {layer_indices} 越界:backbone 只有 {n_layers} 层 (0..{n_layers-1})"
        )

    print(
        f"  [extractor] backbone: {n_layers} 层, hidden_size={hidden_size}, "
        f"提取层 {layer_indices} (共 {len(layer_indices)} 层)"
    )
    print(f"  [extractor] 块数={len(blocks)}, block_size={block_size}, batch_size={batch_size}")

    # hidden_states[i+1] = layer i 输出。我们要 layer_indices 对应的 hidden_states idx
    hs_indices = [i + 1 for i in layer_indices]

    out = torch.empty(
        (len(blocks), len(layer_indices), hidden_size),
        dtype=dtype,
        device="cpu",
    )

    n_batches = (len(blocks) + batch_size - 1) // batch_size
    with torch.no_grad():
        for bi in range(n_batches):
            batch = blocks[bi * batch_size : (bi + 1) * batch_size]
            input_ids = torch.tensor(
                [b.token_ids for b in batch], dtype=torch.long, device=device
            )
            outputs = text_model(
                input_ids=input_ids, output_hidden_states=True, use_cache=False
            )
            # hidden_states: tuple of (B, T, D),长度 = n_layers + 1
            hs = outputs.hidden_states
            # 取末 token 的指定层 → (B, len(layer_indices), D)
            stacked = torch.stack([hs[idx][:, -1, :] for idx in hs_indices], dim=1)
            out[bi * batch_size : bi * batch_size + len(batch)] = stacked.to("cpu", dtype=dtype)

            if (bi + 1) % progress_every == 0 or bi == n_batches - 1:
                print(f"  [extractor] {bi + 1}/{n_batches} batches", flush=True)

    # 释放 backbone 显存(generator 后续不再用)
    del full
    del text_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return out


def get_backbone_dims(backbone_model_path: str) -> tuple[int, int, int]:
    """轻量探查 backbone 的 (n_layers, hidden_size, intermediate_size),不加载权重。

    用 AutoConfig 即可,启动期校验 layer_indices / persona_expert intermediate_dim。
    """
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(backbone_model_path, trust_remote_code=True)
    # Qwen3.5 的字段在 text_config 子树下
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    return (
        int(cfg.num_hidden_layers),
        int(cfg.hidden_size),
        int(cfg.intermediate_size),
    )
