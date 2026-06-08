"""把 nm_generalize 的 eval_set 口径施加到一个真 checkpoint 上,判 probe 55% vs real 14.5% 是
eval 方法学差(H1)还是训练差(H2)。

eval_set 口径(teacher-forced first-token):forward 整个 turn 的 ids,read turn 取 weights>0.5 的
首 value token 位置 pos,pred=logits[pos-1].argmax,gold=labels[pos]。NM-on(override=None)+
NM-zero(override=0.0)。与 validate_memory 的 _check_first_token(prefix forward + offset_mapping
定位)对比:若本脚本在 v22 ckpt 上给 ~14.5% → 两口径一致 = H2(训练差);若 ~50% → H1(eval 差)。

用法: .venv-linux/bin/python scripts/probe/_probe_eval_ckpt.py \
  --checkpoint checkpoints/xinhe_step_6000.pt --config configs/pcap_skeleton_5080_v22.yaml \
  --val data/skeleton/val.jsonl --eval-eps 50
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.amp import autocast

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from scripts.evaluate import load_model_and_tokenizer
from scripts.probe.nm_generalize import load_eps, tokenize_eps


@torch.no_grad()
def eval_set(model, tok, data, device, dtype, mem_override=None, vthresh=1.5):
    # vthresh: value token 的 weight=wps(2.5/5.0)> base lm_weight(1.0)。
    #   weights>1.5 只选 value token(修正旧 weights>0.5 误选到 preamble 首 token 的 bug)。
    model.eval()
    n_tot = n_cor = n_skip = 0
    for ep in data:
        state = model.init_state()
        for t in ep["turns"]:
            with autocast("cuda", dtype=dtype):
                out = model(t["ids"], state, labels=t["labels"], weights=t["weights"],
                            pad_token_id=tok.pad_token_id, mem_alpha_override=mem_override)
            state = out["state_next"]
            if t["is_read"]:
                vmask = t["weights"] > vthresh
                if vmask.any():
                    pos = vmask.long().argmax(dim=1).item()
                    pred = out["logits"][0, pos - 1].argmax().item()
                    gold = t["labels"][0, pos].item()
                    n_tot += 1
                    n_cor += int(pred == gold)
                else:
                    n_skip += 1
    if n_skip:
        print(f"  [warn] {n_skip} read-turn 无 weight>{vthresh} 的 value token(被跳过)")
    return n_cor, n_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val", default="data/skeleton/val.jsonl")
    ap.add_argument("--train", default="data/skeleton/train.jsonl")
    ap.add_argument("--eval-eps", type=int, default=50)
    ap.add_argument("--train-eps", type=int, default=40)
    ap.add_argument("--seg-len", type=int, default=256)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt.get("config"), XinheConfig):
        config = ckpt["config"]
    config.compile_backbone_layers = False
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model, tok = load_model_and_tokenizer(config, args.checkpoint, device)

    val_raw = load_eps(args.val, args.eval_eps)
    train_raw = load_eps(args.train, args.train_eps)
    val_data = tokenize_eps(val_raw, tok, args.seg_len, device)
    train_data = tokenize_eps(train_raw, tok, args.seg_len, device)
    print(f"  loaded train={len(train_data)} val={len(val_data)}")

    tc, tt = eval_set(model, tok, train_data, device, dtype)
    vc, vt = eval_set(model, tok, val_data, device, dtype)
    vzc, vzt = eval_set(model, tok, val_data, device, dtype, mem_override=0.0)
    von = 100 * vc / (vt + 1e-9)
    vzero = 100 * vzc / (vzt + 1e-9)
    print(f"\n=== probe-eval(eval_set 口径)on {args.checkpoint} ===")
    print(f"  train  {100*tc/(tt+1e-9):5.1f}% ({tc}/{tt})")
    print(f"  val NM-on   {von:5.1f}% ({vc}/{vt})")
    print(f"  val NM-zero {vzero:5.1f}% ({vzc}/{vzt})")
    print(f"  val Δ(memory) {von - vzero:+5.1f}pp")


if __name__ == "__main__":
    main()
