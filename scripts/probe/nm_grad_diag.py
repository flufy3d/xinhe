"""
NM 梯度诊断 — 跑一个真实 skeleton episode,记每个 NM 模块的梯度 norm。
read=0% 三轮无变化,先确认 NM 模块到底有没有收到梯度信号。

用法:
  python scripts/probe/nm_grad_diag.py [--ckpt path] [--config configs/pcap_skeleton_5080_v3.yaml]
"""
import argparse
import sys
import json
from pathlib import Path
from collections import OrderedDict

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
    ap.add_argument("--ckpt", type=str, default=None,
                    help="若给,加载 ckpt;否则用 fresh init")
    ap.add_argument("--config", type=str, default="configs/pcap_skeleton_5080_v3.yaml")
    ap.add_argument("--val", type=str, default="data/skeleton/val.jsonl")
    ap.add_argument("--episode-idx", type=int, default=0, help="拿 val 第几条")
    ap.add_argument("--seg-len", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--nm-aux", type=float, default=1.0)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt.get("config"), XinheConfig):
            config = ckpt["config"]
    # 关 compile 跑 eager
    config.compile_backbone_layers = False
    config.nm_aux_weight = args.nm_aux

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    model = XinheModel(config).to(device)
    if args.ckpt:
        # load ckpt
        if "backbone_addons_state" in ckpt:
            addons = {k: v.to(device) for k, v in ckpt["backbone_addons_state"].items()}
            model.backbone.load_state_dict(addons, strict=False)
        if "qhead_state" in ckpt:
            qh = ckpt["qhead_state"]
            model.query_head.load_state_dict(qh["query_head"])
            model.W_mac.load_state_dict(qh["W_mac"])
            model.W_mal.load_state_dict(qh["W_mal"])
            model.global_mem_rmsnorm.load_state_dict(qh["global_mem_rmsnorm"])
            with torch.no_grad():
                model.mal_alpha_logit.copy_(qh["mal_alpha_logit"].to(device))
            model.global_hippo.load_state_dict(qh["global_hippo"])
        print(f"  loaded ckpt: {args.ckpt}")
    else:
        print("  fresh init")

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
    if ep is None:
        print(f"  episode {args.episode_idx} not found"); return

    convs = ep["conversations"]
    print(f"  skeleton: {ep['skeleton_id']} | n_turns: {len(convs)}")

    # 分 turn pair (user + asst)
    turns = []
    for i in range(0, len(convs) - 1, 2):
        if i + 1 >= len(convs): break
        u = convs[i]
        a = convs[i + 1]
        if u["role"] != "user" or a["role"] != "assistant": continue
        turns.append((u, a))
    print(f"  pair turns: {len(turns)}")

    # tokenize 每 turn
    turn_tensors = []
    for ti, (u, a) in enumerate(turns):
        # tokenize_turn 接受单个 turn dict 列表 -> ids/labels/weights
        encoded = tokenize_turn(
            tok, u["content"], a["content"], args.seg_len,
            train_loss=a.get("train_loss", "true"),
            value_spans=a.get("value_span"),
            weight_per_span=a.get("weight_per_span", 0.0),
        )
        if encoded is None: continue
        ids, labels, weights = encoded
        is_read = a.get("value") and not any(v in u["content"] for v in a["value"])
        turn_tensors.append({
            "ids": torch.tensor(ids, device=device).unsqueeze(0),
            "labels": torch.tensor(labels, device=device).unsqueeze(0),
            "weights": torch.tensor(weights, device=device).unsqueeze(0),
            "is_read": is_read,
            "value": a.get("value"),
            "u": u["content"][:50],
            "a": a["content"][:50],
        })

    print(f"  encoded turns: {len(turn_tensors)}")
    for i, t in enumerate(turn_tensors):
        marker = "★READ★" if t["is_read"] else "      "
        v = t["value"] or []
        print(f"    turn {i} {marker} U:{t['u'][:40]:42s} A:{t['a'][:40]:42s} v={v}")

    # 跑 forward,1 个 tbptt block(全 episode)
    state = model.init_state()
    total_loss = torch.zeros((), device=device)

    for ti, t in enumerate(turn_tensors):
        with autocast("cuda", dtype=dtype):
            out = model(
                t["ids"], state, labels=t["labels"], weights=t["weights"],
                pad_token_id=tok.pad_token_id,
            )
        state = out["state_next"]
        total_loss = total_loss + out["loss"]
        marker = "★READ★" if t["is_read"] else "write " if t["value"] else "      "
        ce = out["ce_loss"].item() if "ce_loss" in out else float("nan")
        aux = out["aux_loss"].item() if "aux_loss" in out else float("nan")
        print(f"    turn {ti} {marker} ce={ce:.3f} aux={aux:.3f} loss={out['loss'].item():.3f}")

    print(f"\n  total_loss = {total_loss.item():.3f}")

    # backward
    model.zero_grad()
    total_loss.backward()

    # 收集 grad norm
    print("\n=== NM 模块梯度 norm ===")
    nm_modules = {
        "QueryHead.proj": model.query_head.proj if hasattr(model.query_head, "proj") else model.query_head,
        "W_mac": model.W_mac,
        "W_mal": model.W_mal,
        "global_mem_rmsnorm": model.global_mem_rmsnorm,
        "global_hippo": model.global_hippo,
        "mal_alpha_logit": None,  # 单 scalar,特殊处理
    }
    print(f"  {'module':30s} {'n_params':>10s} {'grad_norm':>12s} {'param_norm':>12s} {'rel_grad':>12s}")
    for name, mod in nm_modules.items():
        if name == "mal_alpha_logit":
            p = model.mal_alpha_logit
            gn = p.grad.norm().item() if p.grad is not None else 0.0
            pn = p.norm().item()
            print(f"  {name:30s} {p.numel():>10d} {gn:>12.4e} {pn:>12.4e} {(gn/(pn+1e-12)):>12.4e}")
            continue
        total_gn = 0.0
        total_pn = 0.0
        n_params = 0
        for pn_name, p in mod.named_parameters():
            if p.grad is None:
                continue
            total_gn += p.grad.norm().item() ** 2
            total_pn += p.norm().item() ** 2
            n_params += p.numel()
        total_gn = total_gn ** 0.5
        total_pn = total_pn ** 0.5
        rel = total_gn / (total_pn + 1e-12)
        print(f"  {name:30s} {n_params:>10d} {total_gn:>12.4e} {total_pn:>12.4e} {rel:>12.4e}")

    print("\n=== Backbone addons 梯度 norm ===")
    # LoRA
    lora_gn = 0.0; lora_pn = 0.0; lora_np = 0
    kp_gn = 0.0; kp_pn = 0.0; kp_np = 0
    for name, p in model.backbone.named_parameters():
        if p.grad is None: continue
        if "lora" in name.lower():
            lora_gn += p.grad.norm().item() ** 2
            lora_pn += p.norm().item() ** 2
            lora_np += p.numel()
        elif "k_pers" in name.lower() or "v_pers" in name.lower():
            kp_gn += p.grad.norm().item() ** 2
            kp_pn += p.norm().item() ** 2
            kp_np += p.numel()
    print(f"  {'LoRA':30s} {lora_np:>10d} {lora_gn**0.5:>12.4e} {lora_pn**0.5:>12.4e}")
    print(f"  {'K/V_pers':30s} {kp_np:>10d} {kp_gn**0.5:>12.4e} {kp_pn**0.5:>12.4e}")

    print("\n=== 判读 ===")
    print("  grad_norm == 0 → 该模块没有收到梯度信号(BUG!)")
    print("  rel_grad 太小 → 学不动 / 学得慢")
    print("  ★READ★ turn 的 aux 应该 > 0(nm_aux 推 NM 召回 value)")


if __name__ == "__main__":
    main()
