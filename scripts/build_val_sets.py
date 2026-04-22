"""
生成 4 个 val jsonl (VALUE / WorldQA / Refusal / Compositional)。

用法:
    python scripts/build_val_sets.py \
        --cache-dir data/cache \
        --out-dir data/val \
        --n-value 200 --n-worldqa 150 --n-refusal 200 --n-compositional 100
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.generate_persona_data import generate_val_sets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--out-dir", type=str, default="data/val")
    p.add_argument("--n-value", type=int, default=200)
    p.add_argument("--n-worldqa", type=int, default=150)
    p.add_argument("--n-refusal", type=int, default=200)
    p.add_argument("--n-compositional", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    paths = generate_val_sets(
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        n_value=args.n_value,
        n_worldqa=args.n_worldqa,
        n_refusal=args.n_refusal,
        n_compositional=args.n_compositional,
        seed=args.seed,
    )
    print("\n[完成]")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
