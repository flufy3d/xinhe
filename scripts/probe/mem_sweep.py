"""GPU memory sweep:5080 16GB 上找最大 (batch, tbptt, shortcut) 配置。

每个配置跑 1 个完整 TBPTT 窗口(tbptt_turns × forward + backward + optimizer.step),
用 torch.cuda.max_memory_allocated 拿 peak,失败 = OOM 直接 catch。

用法:
  uv run python scripts/probe/mem_sweep.py --device cuda
  # 自定义 grid:
  uv run python scripts/probe/mem_sweep.py --batches 1,2,4 --tbptt 4,8 --seg-len 256
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


def measure_one(
    cfg: XinheConfig,
    batch: int,
    tbptt: int,
    seg_len: int,
    shortcut: bool,
    dev: torch.device,
) -> dict:
    """单配置测 1 个 tbptt 窗口 forward+backward 的 peak mem(GB)。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    try:
        torch.manual_seed(0)
        model = XinheModel(cfg).to(dev)
        model.train()
        opt = torch.optim.AdamW(model.get_trainable_params(), lr=3e-4)

        # 模拟 tbptt 窗口:tbptt 个 turn 累积 loss → 一次 backward
        state = model.init_state(batch).to(dev)
        ids = torch.randint(5, 50000, (batch, seg_len), device=dev)
        labels = ids.clone()
        weights = torch.ones_like(ids, dtype=torch.float32)

        accumulated = None
        for _ in range(tbptt):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, state, labels=labels, weights=weights,
                            pad_token_id=0, compute_logits=False)
            state = out["state_next"]
            loss = out["loss"]

            # shortcut:再跑一次 NM-zero forward 带 grad
            if shortcut:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_z = model(ids, state, labels=labels, weights=weights,
                                  pad_token_id=0, compute_logits=False,
                                  mem_alpha_override=0.0)
                loss_zero = out_z["loss"]
                penalty = torch.clamp(loss - loss_zero + 0.5, min=0.0)
                loss = loss + penalty

            accumulated = loss if accumulated is None else accumulated + loss

        (accumulated / tbptt).backward()
        opt.step()
        opt.zero_grad()

        peak = torch.cuda.max_memory_allocated() / 1e9
        # 真训练还要留 ~1-2 GB buffer 给 eval / probe / fragmentation
        del model, opt, state, out, loss, accumulated, ids, labels, weights
        torch.cuda.empty_cache()
        gc.collect()
        return {"ok": True, "peak_gb": peak, "err": None}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {"ok": False, "peak_gb": float("nan"), "err": "OOM"}
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {"ok": False, "peak_gb": float("nan"),
                "err": f"{type(e).__name__}: {str(e)[:100]}"}


def main():
    ap = argparse.ArgumentParser(description="GPU mem sweep for 5080 16GB")
    ap.add_argument("--config", default="configs/pcap_skeleton.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batches", default="1,2,4",
                    help="逗号分隔,例 1,2,4,8")
    ap.add_argument("--tbptt", default="4,8")
    ap.add_argument("--seg-len", type=int, default=256)
    ap.add_argument("--shortcut", default="0,1",
                    help="shortcut on/off,0=off,1=on,默认两都测")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用,跳过")
        return

    dev = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}  total: {total_mem:.2f} GB")
    print()

    cfg, _ = XinheConfig.from_yaml(args.config)
    cfg.use_query_head = True
    cfg.compile_backbone_layers = False

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    tbptts = [int(x) for x in args.tbptt.split(",") if x.strip()]
    shortcuts = [bool(int(x)) for x in args.shortcut.split(",") if x.strip()]

    print(f"sweep: batch ∈ {batches}, tbptt ∈ {tbptts}, shortcut ∈ {shortcuts}, "
          f"seg_len = {args.seg_len}")
    print(f"config: lora={cfg.lora_rank} K_pers={cfg.n_persistent_per_layer}")
    print()
    print(f"{'batch':>5} {'tbptt':>5} {'short':>5} {'peak GB':>10} {'pct':>6}  status")
    print("-" * 60)

    results = []
    for b in batches:
        for t in tbptts:
            for s in shortcuts:
                r = measure_one(cfg, b, t, args.seg_len, s, dev)
                pct = (r["peak_gb"] / total_mem * 100) if r["ok"] else float("nan")
                status = "OK" if r["ok"] else r["err"]
                sym = "✓" if r["ok"] else "✗"
                print(f"{b:>5} {t:>5} {'on' if s else 'off':>5} "
                      f"{r['peak_gb']:>10.2f} {pct:>5.1f}%  {sym} {status}")
                results.append({"batch": b, "tbptt": t, "shortcut": s, **r})

    print()
    # 最大可行配置(留 1.5 GB buffer 给 eval/probe/fragmentation)
    BUFFER_GB = 1.5
    safe = [r for r in results if r["ok"] and r["peak_gb"] < total_mem - BUFFER_GB]
    if safe:
        # 按 (batch * tbptt, shortcut) 降序
        safe.sort(key=lambda r: (r["batch"] * r["tbptt"], int(r["shortcut"])), reverse=True)
        print(f"\n推荐(留 {BUFFER_GB} GB buffer):")
        for r in safe[:3]:
            print(f"  batch={r['batch']} tbptt={r['tbptt']} "
                  f"shortcut={'on' if r['shortcut'] else 'off'}  → {r['peak_gb']:.2f} GB")


if __name__ == "__main__":
    main()
