"""一次性 smoke：单独测某个 OpenRouter 模型生成质量,不污染 train.jsonl。

用法:
    python scripts/smoke_openrouter_model.py \
        --model openai/gpt-oss-safeguard-20b:nitro \
        --n 10 \
        --out data/v8/stage1/smoke_oss20b.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.stage1.beat_planner import BeatPlanner
from xinhe.data.stage1.driver import _generate_one_5beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id 含 '/'")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/v8/stage1/smoke_or.jsonl"))
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--dict-split", default="train")
    ap.add_argument("--beat3-min-chars", type=int, default=500)
    ap.add_argument("--beat3-tolerance", type=float, default=0.75)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    fp = open(args.out, "a", encoding="utf-8")

    rng = random.Random(args.seed)
    planner = BeatPlanner(
        dict_split=args.dict_split,
        beat3_min_chars=args.beat3_min_chars,
        beat3_chars_tolerance=args.beat3_tolerance,
    )

    print(f"=== smoke {args.model} n={args.n} ===")
    print(f"  prompt 字数显示 = {args.beat3_min_chars}/{args.beat3_tolerance:.2f} = {int(args.beat3_min_chars/args.beat3_tolerance)}")
    print(f"  validator 门槛 = {args.beat3_min_chars}")
    print()

    n_ok = 0
    t0 = time.time()
    for i in range(args.n):
        ts = time.time()
        s = _generate_one_5beat(
            rng, planner,
            model=args.model,
            dict_split=args.dict_split,
            max_retries=3,
        )
        dt = time.time() - ts
        if s is None:
            print(f"[{i+1}/{args.n}] FAIL ({dt:.1f}s)")
            continue
        d = s.to_dict()
        # Beat 3 字数
        b3 = sum(len([c for c in t["content"] if "一" <= c <= "鿿"])
                 for t in d["conversations"]
                 if t.get("role") == "assistant" and t.get("train_loss") == "lm_only")
        facts = [(f["canonical_value"], f["scope"]) for f in d["meta"]["canonical_facts"]]
        n_turns = d["meta"]["n_turns"]
        recall = d["meta"]["recall_form"]
        warmup = d["meta"].get("warmup_pairs", 0)
        print(f"[{i+1}/{args.n}] OK ({dt:.1f}s) n_turns={n_turns} beat3_zh={b3} recall={recall} warmup={warmup} facts={facts}")
        fp.write(json.dumps(d, ensure_ascii=False) + "\n")
        fp.flush()
        n_ok += 1

    fp.close()
    total = time.time() - t0
    print()
    print(f"=== 总结 ===")
    print(f"  通过 {n_ok}/{args.n} ({n_ok*100/args.n:.0f}%)")
    print(f"  总耗时 {total:.1f}s, 平均 {total/args.n:.1f}s/条")
    print(f"  落盘 → {args.out}")


if __name__ == "__main__":
    main()
