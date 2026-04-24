"""
T2: β 分布诊断探针

目的: 检验"chat turn token 污染 W"的假设 —— 看现有 ckpt 下:
    memory prompt 的 β 是否比 chat prompt 的 β 系统性高？
    chat prompt 的 β 是否接近 sigmoid(0)≈0.5（写强度 50%，严重污染）？

如果 chat β 系统性接近 0.5 → 确认需要 beta_bias shift（用 shift_beta_bias.py）
如果 chat β 已经普遍很低（<0.1）→ 说明 plugin 已经学会区分，beta_bias 不用改

用法:
    python scripts/probe_beta.py --ckpt checkpoints/curriculum/11_all_mix.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.data.conversation import ensure_chat_template


# Memory 类 prompt（典型 reveal fact）
MEMORY_PROMPTS = [
    ("我叫陈杰。", "好的，陈杰，很高兴认识你！"),
    ("我今年35岁。", "好的，你35岁，记住了。"),
    ("我住在北京。", "好的，北京是个好地方！"),
    ("我喜欢弹吉他。", "好的，弹吉他是个好爱好！"),
    ("我是程序员。", "好的，程序员这个职业不错！"),
    ("我家有只橘猫。", "好的，橘猫一定很可爱！"),
    ("我的编号是12345。", "好的，编号12345，已记录。"),
    ("我最爱吃火锅。", "火锅确实美味！"),
    ("我是Alice。", "你好，Alice！"),
    ("我叫欧阳明月。", "好的，欧阳明月，记住了。"),
]

# Chat 类 prompt（典型闲聊，无持久信息）
CHAT_PROMPTS = [
    ("今天天气真好。", "是啊，阳光不错，适合出去走走。"),
    ("你觉得这部电影怎么样？", "看过的人都说不错，剧情挺抓人的。"),
    ("最近工作忙吗？", "还行吧，都是些琐事。"),
    ("我有点累了。", "那就早点休息吧，明天再说。"),
    ("你在干嘛？", "没干啥，就是在聊天。"),
    ("晚饭吃什么？", "随便点，看心情。"),
    ("下雨了。", "带伞了吗？别淋着。"),
    ("我想去旅行。", "去哪儿呀？有目的地吗？"),
    ("周末有什么安排？", "可能在家看看书吧。"),
    ("心情不太好。", "怎么啦？聊聊。"),
]


@torch.no_grad()
def compute_beta(model, tokenizer, user: str, assistant: str, device, seg_len=256):
    """跑完整 forward，抽出 beta_proj 输出 = sigmoid(beta_w @ content + beta_b)，
    返回 (T_valid, H) 的 β 矩阵（去 padding）。"""
    ensure_chat_template(tokenizer)
    from xinhe.data.conversation import tokenize_turn
    ids, labels, _w = tokenize_turn(
        tokenizer, user, assistant, seg_len, compute_loss=True,
    )
    ids = ids.unsqueeze(0).to(device)
    pad_id = tokenizer.pad_token_id

    # Valid token mask (非 pad + 非 prefix 的 assistant + user 全部内容 token)
    valid_mask = (ids[0] != pad_id).cpu()

    # 先跑 embed → backbone 得到 content_output，和 fact_plugin.write_from_content 里一致
    state = model.init_state(1).to(device)
    from types import SimpleNamespace
    # Hack: 复用 model.forward 的内部流程需要访问 content_output，
    # 简单办法：跑一次 forward 让 state_next 更新，然后从 plugin 重算 β
    # 但我们更想要的是"看到 content 时 β 会是多少" —— 重算才行。

    emb = model.backbone.embed(ids)
    T = ids.shape[1]
    device_t = ids.device
    causal = torch.triu(
        torch.full((T, T), float("-inf"), device=device_t, dtype=emb.dtype),
        diagonal=1,
    )
    pad_col = torch.zeros(1, 1, T, device=device_t, dtype=emb.dtype)
    pad_col.masked_fill_(~(ids != pad_id).unsqueeze(1), float("-inf"))
    mask = causal.unsqueeze(0).unsqueeze(0) + pad_col.unsqueeze(2)
    position_ids = torch.arange(T, dtype=torch.long, device=device_t).unsqueeze(0)

    # Layer hook: 同 xinhe_model.forward 里做状态读
    hook_idx_map = {li: i for i, li in enumerate(model._hook_layer_indices)}

    def hook(hidden, layer_idx):
        if layer_idx not in model._hook_layer_set:
            return hidden
        return model.fact_interface.read_layer(hidden, state, hook_idx_map[layer_idx])

    content_output = model.backbone.forward_blocks(
        emb, attention_mask=mask, position_ids=position_ids, layer_hook=hook,
    )

    # 手动算 β = sigmoid(beta_proj(content_output))
    plugin = model.fact_interface
    c = content_output.to(plugin.beta_proj.weight.dtype)
    beta_logits = F.linear(c, plugin.beta_proj.weight, plugin.beta_proj.bias)
    beta = torch.sigmoid(beta_logits)  # (1, T, H)
    beta_valid = beta[0][valid_mask].cpu().float()  # (T_valid, H)
    return beta_valid


def summarize(label: str, all_betas: list[torch.Tensor]):
    """输出一组 β tensor 的统计（mean / std / p10 / p50 / p90）。"""
    cat = torch.cat(all_betas, dim=0)  # (N, H)
    print(f"\n=== {label} (n_tokens={cat.shape[0]}, n_heads={cat.shape[1]}) ===")
    print(f"  mean β = {cat.mean().item():.4f}")
    print(f"  std β  = {cat.std().item():.4f}")
    flat = cat.flatten()
    print(f"  p10/p50/p90 = "
          f"{torch.quantile(flat, 0.1).item():.4f} / "
          f"{torch.quantile(flat, 0.5).item():.4f} / "
          f"{torch.quantile(flat, 0.9).item():.4f}")
    # Per-head mean
    head_means = cat.mean(dim=0).tolist()
    print(f"  per-head mean β (头数={len(head_means)}):")
    for i, m in enumerate(head_means):
        print(f"    H{i:2d}: {m:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/curriculum/11_all_mix.pt")
    p.add_argument("--config", type=str, default="configs/qwen3.5-0.8b.yaml")
    p.add_argument("--out", type=str, default=None,
                   help="可选: 结果以 JSON 形式保存到此路径")
    args = p.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    from scripts.evaluate import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config, args.ckpt, device)
    model.eval()

    print(f"=== T2 β 分布诊断 ===")
    print(f"ckpt:   {args.ckpt}")
    print(f"config: {args.config}")
    print(f"device: {device}")

    mem_betas = []
    for u, a in MEMORY_PROMPTS:
        beta = compute_beta(model, tokenizer, u, a, device, config.segment_length)
        mem_betas.append(beta)

    chat_betas = []
    for u, a in CHAT_PROMPTS:
        beta = compute_beta(model, tokenizer, u, a, device, config.segment_length)
        chat_betas.append(beta)

    summarize("Memory prompts", mem_betas)
    summarize("Chat prompts", chat_betas)

    mem_all = torch.cat(mem_betas, dim=0)
    chat_all = torch.cat(chat_betas, dim=0)
    diff = mem_all.mean().item() - chat_all.mean().item()
    print(f"\n=== 对比 ===")
    print(f"mean β (memory - chat) = {diff:+.4f}")
    if diff > 0.15:
        print(f"  ✓ memory β 显著高于 chat β → plugin 已经在区分")
    elif diff > 0.0:
        print(f"  ~ 有区分但不强，观察")
    else:
        print(f"  ✗ chat β ≥ memory β，W 污染风险")

    if chat_all.mean().item() > 0.3:
        print(f"\n⚠️  chat β 均值 {chat_all.mean().item():.4f} 偏高（>0.3），")
        print(f"    建议跑 shift_beta_bias.py --delta -2.0 减弱 chat 写强度")
    else:
        print(f"\n✓ chat β 均值 {chat_all.mean().item():.4f} 已经在可接受范围")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "ckpt": args.ckpt,
                "memory": {
                    "mean": mem_all.mean().item(),
                    "std": mem_all.std().item(),
                    "per_head_mean": mem_all.mean(dim=0).tolist(),
                },
                "chat": {
                    "mean": chat_all.mean().item(),
                    "std": chat_all.std().item(),
                    "per_head_mean": chat_all.mean(dim=0).tolist(),
                },
                "diff_mean": diff,
            }, f, indent=2)
        print(f"\n结果保存到 {args.out}")


if __name__ == "__main__":
    main()
