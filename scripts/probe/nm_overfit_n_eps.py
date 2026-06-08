"""
Over-fit N 个 episode 探针 — 在 N 个 distinct (entity, value) 上同时训,
验证 NM 模块的 generalization 能力。

如果 N=1 work(已证)→ N=10 work → N=100 work → N=200 应该也 work。
若 N=10 失败,说明 mechanism 在 multi-entity 上有问题。

用法:
  python scripts/probe/nm_overfit_n_eps.py --n-eps 10 --steps 500
"""
import argparse
import sys
import json
from pathlib import Path

import torch
from torch.amp import autocast

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import ensure_chat_template, tokenize_turn
from transformers import AutoTokenizer


def load_eps(val_path, n):
    eps = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            convs = ep["conversations"]
            # 至少有 1 个 write + 1 个 read 的 episode
            has_write = False
            has_read = False
            for i in range(0, len(convs) - 1, 2):
                a = convs[i + 1] if i + 1 < len(convs) else {}
                if a.get("value"):
                    if a["value"] and not any(v in convs[i].get("content", "") for v in a["value"]):
                        has_read = True
                    else:
                        has_write = True
            if has_write and has_read:
                eps.append(ep)
            if len(eps) >= n:
                break
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pcap_skeleton_5080_v5.yaml")
    ap.add_argument("--val", type=str, default="data/skeleton/val.jsonl")
    ap.add_argument("--n-eps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seg-len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    config.compile_backbone_layers = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"=== over-fit {args.n_eps} episodes ===")
    print(f"  lora_rank={config.lora_rank} K_pers={config.n_persistent_per_layer}")
    print(f"  nm_aux={config.nm_aux_weight} steps={args.steps} lr={args.lr}")

    model = XinheModel(config).to(device)
    model.train()

    tok = AutoTokenizer.from_pretrained(str(Path(config.backbone_model_path).resolve()),
                                         trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)

    eps = load_eps(args.val, args.n_eps)
    print(f"  loaded {len(eps)} episodes")

    # tokenize all episodes
    ep_data = []
    for ep_idx, ep in enumerate(eps):
        convs = ep["conversations"]
        turns = []
        for i in range(0, len(convs) - 1, 2):
            if i + 1 >= len(convs): break
            u, a = convs[i], convs[i + 1]
            encoded = tokenize_turn(
                tok, u["content"], a["content"], args.seg_len,
                train_loss=a.get("train_loss", "true"),
                value_spans=a.get("value_span"),
                weight_per_span=a.get("weight_per_span", 0.0),
            )
            if encoded is None: continue
            ids, labels, weights = encoded
            is_read = a.get("value") and not any(v in u["content"] for v in a["value"])
            turns.append({
                "ids": ids.unsqueeze(0).to(device),
                "labels": labels.unsqueeze(0).to(device),
                "weights": weights.unsqueeze(0).to(device),
                "is_read": is_read,
                "value": a.get("value"),
            })
        ep_data.append({"sid": ep["skeleton_id"], "turns": turns})

    nm_params = list(model.get_trainable_params())
    opt = torch.optim.AdamW(nm_params, lr=args.lr, weight_decay=0.1)

    @torch.no_grad()
    def eval_all():
        model.eval()
        n_total = 0
        n_correct = 0
        for ep in ep_data:
            state = model.init_state()
            for t in ep["turns"]:
                with autocast("cuda", dtype=dtype):
                    out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                                pad_token_id=tok.pad_token_id)
                state = out["state_next"]
                if t["is_read"]:
                    logits = out["logits"]
                    vmask = t["weights"] > 0.5
                    if vmask.any():
                        pos = vmask.long().argmax(dim=1).item()
                        pred = logits[0, pos - 1].argmax().item()
                        gold = t["labels"][0, pos].item()
                        n_total += 1
                        n_correct += int(pred == gold)
        model.train()
        return n_correct, n_total

    init_corr, init_tot = eval_all()
    print(f"  init: read first-token correct = {init_corr}/{init_tot} ({100*init_corr/(init_tot+1e-9):.1f}%)")

    for step in range(1, args.steps + 1):
        # 随机选一个 episode 训
        ep_idx = step % len(ep_data)
        ep = ep_data[ep_idx]
        opt.zero_grad()
        state = model.init_state()
        total_loss = torch.zeros((), device=device)
        for t in ep["turns"]:
            with autocast("cuda", dtype=dtype):
                out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                            pad_token_id=tok.pad_token_id)
            state = out["state_next"]
            total_loss = total_loss + out["loss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(nm_params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            corr, tot = eval_all()
            print(f"  step {step:4d}  total={total_loss.item():.3f}  read_correct={corr}/{tot} ({100*corr/(tot+1e-9):.1f}%)")

    print(f"\n=== verdict ===")
    final_corr, final_tot = eval_all()
    pct = 100 * final_corr / (final_tot + 1e-9)
    if pct >= 80:
        print(f"  ✅ {args.n_eps} ep over-fit work:{pct:.1f}% recall")
    elif pct >= 30:
        print(f"  ⚠ {args.n_eps} ep over-fit partial:{pct:.1f}% recall - mechanism strained")
    else:
        print(f"  ❌ {args.n_eps} ep over-fit fail:{pct:.1f}% recall - mechanism has limit")


if __name__ == "__main__":
    main()
