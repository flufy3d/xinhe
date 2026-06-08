"""
Over-fit 单 episode 探针 — fresh init + 1 个 skeleton episode + 重复 N 步训练,
看 read first-token loss 是否真能降到 ~0。

如果能 over-fit:mechanism work,失败是 generalization / 数据多样性问题。
如果不能:mechanism 有 hidden bug,debug 必须深入 code。

用法:
  python scripts/probe/nm_overfit_one_ep.py [--config configs/pcap_skeleton_5080_v4.yaml] [--steps 500]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pcap_skeleton_5080_v4.yaml")
    ap.add_argument("--val", type=str, default="data/skeleton/val.jsonl")
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seg-len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--shortcut", action="store_true", help="开 shortcut suppression")
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    config.compile_backbone_layers = False
    config.shortcut_suppression = args.shortcut

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"=== over-fit one episode ===")
    print(f"  config: {args.config}")
    print(f"  lora_rank={config.lora_rank} K_pers={config.n_persistent_per_layer}")
    print(f"  nm_aux_weight={config.nm_aux_weight} shortcut={args.shortcut}")
    print(f"  steps={args.steps} lr={args.lr}")

    model = XinheModel(config).to(device)
    model.train()

    tok = AutoTokenizer.from_pretrained(str(Path(config.backbone_model_path).resolve()),
                                         trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)

    # 读 episode
    with open(args.val, encoding="utf-8") as f:
        ep = None
        for i, line in enumerate(f):
            if i == args.episode_idx:
                ep = json.loads(line)
                break
    convs = ep["conversations"]
    print(f"  skeleton={ep['skeleton_id']} n_turns={len(convs)}")

    # 拼 turn pair + tokenize
    turn_tensors = []
    for i in range(0, len(convs) - 1, 2):
        u = convs[i]
        a = convs[i + 1]
        if u["role"] != "user" or a["role"] != "assistant": continue
        encoded = tokenize_turn(
            tok, u["content"], a["content"], args.seg_len,
            train_loss=a.get("train_loss", "true"),
            value_spans=a.get("value_span"),
            weight_per_span=a.get("weight_per_span", 0.0),
        )
        ids, labels, weights = encoded
        is_read = a.get("value") and not any(v in u["content"] for v in a["value"])
        turn_tensors.append({
            "ids": ids.unsqueeze(0).to(device),
            "labels": labels.unsqueeze(0).to(device),
            "weights": weights.unsqueeze(0).to(device),
            "is_read": is_read,
            "value": a.get("value"),
            "u": u["content"][:40],
            "a": a["content"][:40],
        })

    print(f"  encoded turns: {len(turn_tensors)}")
    for i, t in enumerate(turn_tensors):
        marker = "★READ★" if t["is_read"] else "write " if t["value"] else "      "
        print(f"    turn {i} {marker} v={t['value']} ids_len={t['ids'].shape[1]}")

    # optimizer:NM 模块 + W_mac/W_mal + 单 scalar
    nm_params = list(model.get_trainable_params())
    opt = torch.optim.AdamW(nm_params, lr=args.lr, weight_decay=0.1)
    print(f"  optimizer: {sum(p.numel() for p in nm_params)} trainable params")

    # eval helper
    @torch.no_grad()
    def eval_read():
        """跑一遍 episode,记 read turn 的 ce_loss 和 first-token argmax 是否对。"""
        model.eval()
        state = model.init_state()
        read_ce = None
        read_correct = None
        for t in turn_tensors:
            with autocast("cuda", dtype=dtype):
                out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                            pad_token_id=tok.pad_token_id)
            state = out["state_next"]
            if t["is_read"]:
                read_ce = out["ce_loss"].item()
                # argmax check: 看 weights>0.5 那一位 logit argmax 跟 label 是否一致
                logits = out["logits"]  # (B, T, V)
                vmask = t["weights"] > 0.5
                if vmask.any():
                    pos = vmask.long().argmax(dim=1).item()
                    pred = logits[0, pos - 1].argmax().item()  # next-token pred at pos-1 → label at pos
                    gold = t["labels"][0, pos].item()
                    read_correct = (pred == gold)
        model.train()
        return read_ce, read_correct

    # 训练循环
    last_read_ce, last_correct = eval_read()
    print(f"  init eval: read_ce={last_read_ce:.3f} correct={last_correct}")

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        state = model.init_state()
        total_loss = torch.zeros((), device=device)
        for t in turn_tensors:
            with autocast("cuda", dtype=dtype):
                out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                            pad_token_id=tok.pad_token_id)
            state = out["state_next"]
            total_loss = total_loss + out["loss"]

            if args.shortcut and t["is_read"]:
                # 仅对 read turn 加 shortcut
                with autocast("cuda", dtype=dtype):
                    out_zero = model(t["ids"], state.detach() if hasattr(state, 'detach') else state,
                                     labels=t["labels"], weights=t["weights"],
                                     pad_token_id=tok.pad_token_id,
                                     compute_logits=False, mem_alpha_override=0.0)
                ce_on = out["ce_loss"]
                ce_zero = out_zero["ce_loss"]
                pen = torch.clamp(ce_on - ce_zero + 0.3, min=0.0)
                total_loss = total_loss + 2.0 * pen

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(nm_params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            read_ce, correct = eval_read()
            print(f"  step {step:4d}  total={total_loss.item():.3f}  read_ce={read_ce:.3f}  correct={correct}")
            last_read_ce = read_ce
            last_correct = correct

    print(f"\n=== verdict ===")
    if last_correct:
        print(f"  ✅ over-fit 成功:read_ce={last_read_ce:.3f},first-token argmax 命中")
        print(f"     → mechanism work,真训练 read=0% 是 generalization / 数据多样性问题")
    else:
        print(f"  ❌ over-fit 失败:read_ce={last_read_ce:.3f},first-token argmax 没命中")
        print(f"     → mechanism 有 hidden bug,需深入 code")


if __name__ == "__main__":
    main()
