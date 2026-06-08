"""按 mix_dynamic 一样的 seed shuffle 取前 N 条,dump 给 validate_memory.py 用。"""
import json
import random
import sys

src = sys.argv[1]
n = int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
out = sys.argv[4]

with open(src, encoding="utf-8") as f:
    items = [json.loads(ln) for ln in f if ln.strip()]

rng = random.Random(seed)
rng.shuffle(items)
taken = items[:n]

with open(out, "w", encoding="utf-8") as f:
    for s in taken:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"dump {len(taken)} ep -> {out}")
