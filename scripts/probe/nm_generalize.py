"""
泛化探针 — 在 N 个 train episode 上训(逼 memorize),同时 eval:
  (a) 那 N 个 train episode 的 read-turn 召回
  (b) M 个 DISJOINT held-out episode 的 read-turn 召回(从另一个文件)

判读(per-layer Delta read 已证 memorize 100%):
  train→高 且 val→高   = 真泛化 → scale 数据/步数冲 95
  train→高 但 val→0    = memorize-only,泛化是墙 → 转 entity 多样性 + query 增强

用法:
  python scripts/probe/nm_generalize.py --config configs/pcap_skeleton_5080_v19b.yaml \
    --train data/skeleton/train.jsonl --val data/skeleton/val.jsonl \
    --train-eps 30 --eval-eps 30 --steps 1500 --lr 3e-4 --log-every 100
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


def load_eps(path, n):
    eps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line)
            convs = ep["conversations"]
            has_write = has_read = False
            for i in range(0, len(convs) - 1, 2):
                a = convs[i + 1] if i + 1 < len(convs) else {}
                if a.get("value"):
                    if not any(v in convs[i].get("content", "") for v in a["value"]):
                        has_read = True
                    else:
                        has_write = True
            if has_write and has_read:
                eps.append(ep)
            if len(eps) >= n:
                break
    return eps


def tokenize_eps(eps, tok, seg_len, device):
    out = []
    for ep in eps:
        convs = ep["conversations"]
        turns = []
        for i in range(0, len(convs) - 1, 2):
            if i + 1 >= len(convs):
                break
            u, a = convs[i], convs[i + 1]
            enc = tokenize_turn(
                tok, u["content"], a["content"], seg_len,
                train_loss=a.get("train_loss", "true"),
                value_spans=a.get("value_span"),
                weight_per_span=a.get("weight_per_span", 0.0),
            )
            if enc is None:
                continue
            ids, labels, weights = enc
            is_read = a.get("value") and not any(v in u["content"] for v in a["value"])
            turns.append({
                "ids": ids.unsqueeze(0).to(device),
                "labels": labels.unsqueeze(0).to(device),
                "weights": weights.unsqueeze(0).to(device),
                "is_read": is_read,
            })
        out.append({"sid": ep.get("skeleton_id"), "turns": turns})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pcap_skeleton_5080_v19b.yaml")
    ap.add_argument("--train", type=str, default="data/skeleton/train.jsonl")
    ap.add_argument("--val", type=str, default="data/skeleton/val.jsonl")
    ap.add_argument("--train-eps", type=int, default=30)
    ap.add_argument("--eval-eps", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seg-len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--shortcut", action="store_true",
                    help="开 Margin-Based shortcut suppression(per-turn 双 forward,逼用 memory)")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lambda-sc", type=float, default=2.0)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    config.compile_backbone_layers = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"=== generalize: train {args.train_eps} / eval held-out {args.eval_eps} ===")
    print(f"  lora_rank={config.lora_rank} read_mode={getattr(config,'read_mode','?')} "
          f"read_scale_init={getattr(config,'read_scale_init','?')} steps={args.steps} lr={args.lr}")
    print(f"  shortcut={args.shortcut} margin={args.margin} lambda_sc={args.lambda_sc}")

    model = XinheModel(config).to(device)
    model.train()

    tok = AutoTokenizer.from_pretrained(str(Path(config.backbone_model_path).resolve()),
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)

    train_raw = load_eps(args.train, args.train_eps)
    val_raw = load_eps(args.val, args.eval_eps)
    print(f"  loaded train={len(train_raw)} val={len(val_raw)}")
    train_data = tokenize_eps(train_raw, tok, args.seg_len, device)
    val_data = tokenize_eps(val_raw, tok, args.seg_len, device)

    nm_params = list(model.get_trainable_params())
    opt = torch.optim.AdamW(nm_params, lr=args.lr, weight_decay=0.1)

    @torch.no_grad()
    def eval_set(data, mem_override=None):
        # mem_override=0.0 → NM-zero ablation(read 注入全关,测无记忆 baseline)
        model.eval()
        n_tot = n_cor = 0
        for ep in data:
            state = model.init_state()
            for t in ep["turns"]:
                with autocast("cuda", dtype=dtype):
                    out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                                pad_token_id=tok.pad_token_id, mem_alpha_override=mem_override)
                state = out["state_next"]
                if t["is_read"]:
                    # weights>1.5 只选 value token(wps=2.5/5.0);旧 >0.5 会选到 assistant
                    # preamble 首 token → 虚高假象(2026-06-05 定位,见 feedback_eval_metric_must_target_value_token)
                    vmask = t["weights"] > 1.5
                    if vmask.any():
                        pos = vmask.long().argmax(dim=1).item()
                        pred = out["logits"][0, pos - 1].argmax().item()
                        gold = t["labels"][0, pos].item()
                        n_tot += 1
                        n_cor += int(pred == gold)
        model.train()
        return n_cor, n_tot

    train_eval = train_data[:40]  # 抽样 eval,避免大 N 时全量 eval 拖慢
    tc, tt = eval_set(train_eval)
    vc, vt = eval_set(val_data)
    vzc, vzt = eval_set(val_data, mem_override=0.0)
    print(f"  init: train {100*tc/(tt+1e-9):.1f}%  val(NM-on) {100*vc/(vt+1e-9):.1f}%  "
          f"val(NM-zero) {100*vzc/(vzt+1e-9):.1f}%")

    for step in range(1, args.steps + 1):
        ep = train_data[step % len(train_data)]
        opt.zero_grad()
        state = model.init_state()
        loss = torch.zeros((), device=device)
        gap_sum = 0.0; gap_n = 0
        for t in ep["turns"]:
            with autocast("cuda", dtype=dtype):
                out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                            pad_token_id=tok.pad_token_id)
                turn_loss = out["loss"]
                if args.shortcut:
                    # NM-zero 反事实 forward(同一 state,带 grad,不 detach)→ margin penalty
                    out_z = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                                  pad_token_id=tok.pad_token_id, mem_alpha_override=0.0)
                    ce_on = out.get("ce_loss", out["loss"])
                    ce_zero = out_z.get("ce_loss", out_z["loss"])
                    penalty = torch.clamp(ce_on - ce_zero + args.margin, min=0.0)
                    turn_loss = turn_loss + args.lambda_sc * penalty
                    gap_sum += float((ce_zero.detach() - ce_on.detach()).item()); gap_n += 1
            state = out["state_next"]
            loss = loss + turn_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nm_params, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            tc, tt = eval_set(train_eval)
            vc, vt = eval_set(val_data)
            vzc, vzt = eval_set(val_data, mem_override=0.0)
            von = 100*vc/(vt+1e-9); vzero = 100*vzc/(vzt+1e-9)
            gap_str = f"  gap={gap_sum/(gap_n+1e-9):+.3f}" if args.shortcut else ""
            print(f"  step {step:5d}  loss={loss.item():.3f}  "
                  f"train={100*tc/(tt+1e-9):5.1f}%  "
                  f"val={von:5.1f}% (zero={vzero:4.1f}% Δ={von-vzero:+5.1f}) ({vc}/{vt}){gap_str}")

    print("\n=== verdict ===")
    tc, tt = eval_set(train_eval)
    vc, vt = eval_set(val_data)
    vzc, vzt = eval_set(val_data, mem_override=0.0)
    tp = 100 * tc / (tt + 1e-9)
    vp = 100 * vc / (vt + 1e-9)
    vz = 100 * vzc / (vzt + 1e-9)
    print(f"  train {tp:.1f}% | val NM-on {vp:.1f}% / NM-zero {vz:.1f}% / Δ {vp-vz:+.1f}pp")
    if vp >= 60 and (vp - vz) >= 30:
        print(f"  ✅ 真泛化且记忆驱动 → scale 数据+步数冲 95")
    elif vp < 20:
        print(f"  ❌ 泛化失败")
    else:
        print(f"  ⚠ 部分泛化")


if __name__ == "__main__":
    main()
