"""非交互对照测试:base vs persona-on vs persona-off,看输出乱不乱码。

用法 (远端):
  uv run python -u scripts/test_persona.py \
      --persona-ckpt checkpoints/persona_expert/novel_63f62e40b306/persona_expert.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.model.persona_expert import (
    attach_persona_expert, detach_persona_expert,
    set_persona_enabled, set_persona_scale,
)
from xinhe.data.conversation import ensure_chat_template


PROMPTS = [
    "你好",
    "你叫什么名字?",
    "罗兰,你今天在做什么?",
    "讲一个关于女巫的故事",
]


def make_gen(model, tokenizer, device, eos_id, max_new_tokens=80, temperature=0.7):
    def gen(prompt: str, state: torch.Tensor) -> str:
        msg = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)],
                           dtype=torch.long, device=device)
        with torch.no_grad():
            out, _ = model.generate_with_state(
                ids, state, max_new_tokens=max_new_tokens,
                eos_token_id=eos_id, temperature=temperature, top_p=0.95,
                repetition_penalty=1.1,
            )
        new_ids = out[0, ids.size(1):].tolist()
        s = tokenizer.decode(new_ids, skip_special_tokens=False)
        for tag in ["<|im_end|>", "</s>", "<|endoftext|>"]:
            s = s.replace(tag, "")
        return s.strip()
    return gen


def run_block(name: str, gen_fn, state, prompts):
    print(f"\n{'='*20} {name} {'='*20}", flush=True)
    for p in prompts:
        out = gen_fn(p, state)
        print(f"USER: {p}", flush=True)
        print(f"BOT : {out}", flush=True)
        print("-" * 60, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3.5-0.8b.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="Hippocampus + LoRA ckpt(可选;不给就 fresh init)")
    ap.add_argument("--persona-ckpt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    # --- 加载 config + model ---
    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = XinheModel(config)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "hippocampus_state" in ckpt:
            model.hippocampus.load_state_dict(ckpt["hippocampus_state"], strict=True)
        from xinhe.model.lora import LoRALinear
        lora_state = ckpt.get("lora_state", {})
        for name, module in model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state:
                    module.lora_B.data = lora_state[f"{name}.lora_B"]
        print(f"  loaded hippocampus + lora from {args.checkpoint}", flush=True)

    model.to(device)
    model.eval()

    # --- tokenizer ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    gen = make_gen(model, tokenizer, device, eos,
                   max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    state = model.init_state(batch_size=1).to(device)

    # --- 1) baseline:无 persona ---
    run_block("BASE (no persona attached)", gen, state, PROMPTS)

    # --- 2) attach persona,跳过最深 1 层(layer 23 训练 cos 仅 0.55,污染最大),scale=0 启动 ---
    summary = attach_persona_expert(model, args.persona_ckpt,
                                    map_location=device, enabled=True,
                                    scale=0.0, skip_last_n=1)
    print(f"\n  attached persona: novel={summary['novel_stem']!r} "
          f"all_layers={summary['layer_indices']} "
          f"attached={summary['attached_layers']} skipped={summary['skipped_layers']} "
          f"hooks={summary['wrapped_count']} params={summary['n_params']/1e6:.1f}M",
          flush=True)

    # 简单 expert 权重量级诊断
    for kv, li in enumerate(summary["layer_indices"]):
        ex = model.persona_expert_stack.experts[kv]
        rms_g = ex.gate_proj.weight.float().pow(2).mean().sqrt().item()
        rms_u = ex.up_proj.weight.float().pow(2).mean().sqrt().item()
        rms_d = ex.down_proj.weight.float().pow(2).mean().sqrt().item()
        print(f"  [layer {li}] expert weight RMS  gate={rms_g:.4f} up={rms_u:.4f} down={rms_d:.4f}  "
              f"dtype={ex.gate_proj.weight.dtype}",
              flush=True)

    # --- 3) 多档 scale 对比:找 sweet spot ---
    for s in (0.3, 0.5, 1.0, 1.5):
        set_persona_scale(model, s)
        run_block(f"PERSONA skip_last_n=1, scale={s}", gen, state, PROMPTS[:3])

    # --- 4) detach,完全卸载 ---
    n = detach_persona_expert(model)
    print(f"\n  detach: removed {n} hooks", flush=True)
    run_block("AFTER DETACH", gen, state, PROMPTS[:2])


if __name__ == "__main__":
    main()
