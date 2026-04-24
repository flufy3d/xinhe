"""
统一数据生成入口 (v7.1) —— 薄壳调用 curriculum_data.generate_data()

不再分发 memory/persona 类型。所有 stage 通过 declarative spec 描述 turn_kinds + patterns 混合，
由 registry 统一生成。

用法:
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --stage 0a_basic_rw
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --all
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --stage 1_persona_unified --force
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.curriculum_data import generate_data as _gen_data


def generate_stage_data(stage: dict, stage_name: str, force: bool = False) -> tuple[str, str]:
    """为课程阶段生成 train.jsonl + val.jsonl。缓存感知（force=False 时检查是否已存在）。"""
    data_cfg = stage.get("data", {})
    out_dir = data_cfg.get("out_dir", f"data/curriculum/{stage_name}")
    train_path = Path(out_dir) / "train.jsonl"
    val_path = Path(out_dir) / "val.jsonl"

    if not force and train_path.exists() and val_path.exists():
        expected = data_cfg.get("num_train") or data_cfg.get("num_episodes", 10000)
        with open(train_path, "r", encoding="utf-8") as f:
            actual = sum(1 for ln in f if ln.strip())
        if actual >= expected * 0.95:
            print(f"  [缓存] 使用已有数据: {out_dir} ({actual} 条)")
            return str(train_path), str(val_path)

    return _gen_data(data_cfg, stage_name, out_dir="data/curriculum")


def main():
    parser = argparse.ArgumentParser(description="统一数据生成 (v7.1)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from xinhe.model.config import XinheConfig
    _config, curriculum = XinheConfig.from_yaml(args.config)

    if not curriculum:
        print("配置文件中没有课程定义")
        sys.exit(1)

    stages_to_gen = []
    if args.stage:
        matched = [s for s in curriculum if s["name"] == args.stage]
        if not matched:
            names = [s["name"] for s in curriculum]
            print(f"阶段 '{args.stage}' 不存在。可用: {names}")
            sys.exit(1)
        stages_to_gen = matched
    elif args.all:
        stages_to_gen = curriculum
    else:
        parser.print_help()
        sys.exit(1)

    for stage in stages_to_gen:
        name = stage["name"]
        print(f"\n{'='*50}\n  生成数据: {name}\n{'='*50}")
        train_path, val_path = generate_stage_data(stage, name, force=args.force)
        print(f"  → {train_path}")
        print(f"  → {val_path}")


if __name__ == "__main__":
    main()
