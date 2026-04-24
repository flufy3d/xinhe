"""
生成 persona 联合评测的 val jsonl 集合 (v7.1)

通过 registry 遍历所有注册的 val 类别。覆盖：
  value / worldqa / refusal / compositional / rapid_overwrite
  verbatim / reference_back / context_followup / topic_continuation
  entity_tracking / irrelevant_forget / multi_slot_retention

用法:
    python scripts/build_val_sets.py --out-dir data/val --cache-dir data/cache
    python scripts/build_val_sets.py --value 200 --verbatim 100 --reference_back 100
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.curriculum_data import generate_val_sets, DEFAULT_VAL_SIZES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--out-dir", type=str, default="data/val")
    p.add_argument("--seed", type=int, default=12345)
    # 每个 val 数量可单独覆盖
    for name, default_n in DEFAULT_VAL_SIZES.items():
        p.add_argument(f"--{name}", type=int, default=default_n,
                       help=f"{name} val 集样本数 (默认 {default_n})")
    args = p.parse_args()

    sizes = {name: getattr(args, name.replace("-", "_"), default)
             for name, default in DEFAULT_VAL_SIZES.items()}
    paths = generate_val_sets(
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        sizes=sizes,
        seed=args.seed,
    )
    print("\n[完成]")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
