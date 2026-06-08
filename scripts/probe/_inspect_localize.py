"""逐 read-turn 对比两套 first-token 口径,定位 validate_memory(14.5%) vs eval_set(66.7%)的差。
为每个 val read turn 打印:
  value 串 / weight_per_span /
  eval_set:  pos, gold=labels[pos], pred=argmax(logits[pos-1])  (用 tokenize_turn 的 weights mask)
  valmem:    target_pos, target_tok, pred                        (用 _locate_value_token)
看两者 gold token 是否同一个、各自 pred 是否命中 → 判哪套在测 value、哪套有 bug。

用法: .venv-linux/bin/python scripts/probe/_inspect_localize.py \
  --checkpoint checkpoints/xinhe_step_6000.pt --config configs/pcap_skeleton_5080_v22.yaml --n 8
"""
import argparse, sys, json
from pathlib import Path
import torch
from torch.amp import autocast

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.data.conversation import tokenize_turn
from scripts.evaluate import load_model_and_tokenizer
from scripts.validate_memory import _locate_value_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--val", default="data/skeleton/val.jsonl")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seg-len", type=int, default=256)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt.get("config"), XinheConfig):
        config = ckpt["config"]
    config.compile_backbone_layers = False
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(config, args.checkpoint, device)

    def dec(t):
        return repr(tok.decode([int(t)]))

    shown = 0
    with open(args.val, encoding="utf-8") as f:
        for line in f:
            if shown >= args.n:
                break
            ep = json.loads(line)
            convs = ep["conversations"]
            state = model.init_state(1).to(device)
            for i in range(0, len(convs) - 1, 2):
                u = convs[i].get("content", "")
                a = convs[i + 1] if i + 1 < len(convs) else {}
                asst = a.get("content", "")
                vspans = a.get("value_span") or []
                wps = float(a.get("weight_per_span", 0.0) or 0.0)
                for s, e in vspans:
                    vstr = asst[s:e]
                    is_recall = vstr not in u
                    if not is_recall:
                        continue
                    if shown >= args.n:
                        break
                    # --- eval_set 口径 ---
                    ids, labels, weights = tokenize_turn(
                        tok, u, asst, args.seg_len, train_loss=a.get("train_loss", "true"),
                        value_spans=[[int(s), int(e)]], weight_per_span=wps or 1.0)
                    ids_d = ids.unsqueeze(0).to(device)
                    labels_d = labels.unsqueeze(0).to(device)
                    weights_d = weights.unsqueeze(0).to(device)
                    with autocast("cuda", dtype=torch.bfloat16):
                        out = model(ids_d, state, labels=labels_d, weights=weights_d,
                                    pad_token_id=tok.pad_token_id)
                    vmask = weights_d > 0.5
                    es_pos = vmask.long().argmax(dim=1).item()
                    es_gold = labels_d[0, es_pos].item()
                    es_pred = out["logits"][0, es_pos - 1].argmax().item()
                    n_valtok = int(vmask.sum().item())

                    # --- validate_memory 口径 ---
                    full_ids, vm_tok, vm_pos = _locate_value_token(tok, u, asst, (int(s), int(e)))
                    if full_ids is not None and vm_pos:
                        pref = torch.tensor([full_ids[:vm_pos]], device=device)
                        with autocast("cuda", dtype=torch.bfloat16):
                            o2 = model(pref, state, pad_token_id=tok.pad_token_id)
                        vm_pred = o2["logits"][0, -1].argmax().item()
                    else:
                        vm_tok = vm_pred = -1

                    print(f"\n[{shown}] value={vstr!r}  wps={wps}  n_valtok(weights>0.5)={n_valtok}")
                    print(f"  eval_set : pos={es_pos} gold={dec(es_gold)} pred={dec(es_pred)} "
                          f"{'HIT' if es_pred==es_gold else 'miss'}")
                    print(f"  valmem   : pos={vm_pos} gold={dec(vm_tok)} pred={dec(vm_pred)} "
                          f"{'HIT' if vm_pred==vm_tok else 'miss'}")
                    shown += 1
                # evolve state teacher-forced
                ids2, _l, _w = tokenize_turn(tok, u, asst, args.seg_len,
                                             train_loss=a.get("train_loss", "true"),
                                             value_spans=vspans, weight_per_span=wps)
                with autocast("cuda", dtype=torch.bfloat16):
                    o = model(ids2.unsqueeze(0).to(device), state, pad_token_id=tok.pad_token_id)
                state = o["state_next"]


if __name__ == "__main__":
    main()
