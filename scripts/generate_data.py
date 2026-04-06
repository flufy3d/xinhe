"""
统一数据生成入口

根据课程阶段配置中的 data.type 自动分发:
  - type: memory (默认) — 调用 generate_memory_data
  - type: think — 调用 generate_think_data (自动缓存)

用法:
    # 为指定阶段生成数据
    python scripts/generate_data.py --config configs/curriculum_qwen.yaml --stage 14_think

    # 为所有阶段生成数据
    python scripts/generate_data.py --config configs/curriculum_qwen.yaml --all

    # 强制重新生成 (忽略缓存)
    python scripts/generate_data.py --config configs/curriculum_qwen.yaml --stage 14_think --force
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def generate_stage_data(
    stage: dict,
    stage_name: str,
    force: bool = False,
    model_path: str = None,
) -> tuple[str, str]:
    """
    为课程阶段生成数据，返回 (train_path, val_path)。

    根据 data.type 自动分发:
      - "memory" (默认): 每次重新生成
      - "think": 有缓存则跳过，无缓存则生成
    """
    data_cfg = stage.get("data", {})
    data_type = data_cfg.get("type", "memory")
    out_dir = data_cfg.get("out_dir", f"data/curriculum/{stage_name}")

    if data_type == "memory":
        return _generate_memory(data_cfg, out_dir)

    elif data_type == "think":
        return _generate_think(data_cfg, out_dir, force, model_path)

    else:
        raise ValueError(f"未知的数据类型: {data_type}")


def _generate_memory(data_cfg: dict, out_dir: str) -> tuple[str, str]:
    """生成记忆训练数据"""
    from xinhe.data.generate_memory_data import generate_data

    return generate_data(
        out_dir=out_dir,
        num_train=data_cfg.get("num_train", 5000),
        num_val=data_cfg.get("num_val", 200),
        min_distance=data_cfg.get("min_distance", 1),
        max_distance=data_cfg.get("max_distance", 4),
        max_turns=data_cfg.get("max_turns", 16),
        num_facts=data_cfg.get("num_facts", 1),
        num_fillers=data_cfg.get("num_fillers", 0),
        no_pre_filler=data_cfg.get("no_pre_filler", False),
        max_pre_filler=data_cfg.get("max_pre_filler", 3),
        no_overwrite=data_cfg.get("no_overwrite", False),
        overwrite_ratio=data_cfg.get("overwrite_ratio", 0.4),
        entity_ratio=data_cfg.get("entity_ratio", 0.0),
        recall_ratio=data_cfg.get("recall_ratio", 0.0),
        same_category=data_cfg.get("same_category", 0.0),
        ai_recall_ratio=data_cfg.get("ai_recall_ratio", 0.0),
        think_ratio=data_cfg.get("think_ratio", 0.0),
        think_lang=data_cfg.get("think_lang", "en"),
        seed=data_cfg.get("seed", 42),
    )


def _generate_think(
    data_cfg: dict, out_dir: str, force: bool, model_path: str = None,
) -> tuple[str, str]:
    """生成 think 训练数据 (自动缓存)"""
    train_path = str(Path(out_dir) / "train.jsonl")
    val_path = str(Path(out_dir) / "val.jsonl")

    # 缓存检查: think 数据生成慢，有数据就跳过
    if not force and Path(train_path).exists() and Path(val_path).exists():
        expected = data_cfg.get("num_think", 5000) + data_cfg.get("num_memory", 5000)
        with open(train_path, "r", encoding="utf-8") as f:
            actual = sum(1 for line in f if line.strip())
        if actual >= expected * 0.95:  # 允许 5% 容差 (失败跳过的)
            print(f"  [缓存] 使用已有数据: {out_dir} ({actual} 条)")
            return train_path, val_path
        print(f"  [数据不足] {actual}/{expected} 条，重新生成")

    # --force 时清掉中间文件，重新生成
    if force:
        for f in [train_path, val_path,
                  str(Path(out_dir) / "_think_train.jsonl"),
                  str(Path(out_dir) / "_think_val.jsonl")]:
            Path(f).unlink(missing_ok=True)

    from xinhe.data.generate_think_data import generate_think_data

    mp = model_path or data_cfg.get("model_path", "./models/qwen3-0.6b")

    return generate_think_data(
        out_dir=out_dir,
        num_think=data_cfg.get("num_think", 5000),
        num_memory=data_cfg.get("num_memory", 5000),
        num_val_think=data_cfg.get("num_val_think", 100),
        num_val_memory=data_cfg.get("num_val_memory", 100),
        model_path=mp,
        device=data_cfg.get("device", "cuda"),
        max_new_tokens=data_cfg.get("max_new_tokens", 512),
        seed=data_cfg.get("seed", 42),
        memory_max_turns=data_cfg.get("memory_max_turns", 8),
        memory_num_facts=data_cfg.get("memory_num_facts", 5),
        memory_entity_ratio=data_cfg.get("memory_entity_ratio", 0.2),
        memory_recall_ratio=data_cfg.get("memory_recall_ratio", 0.2),
        memory_overwrite_ratio=data_cfg.get("memory_overwrite_ratio", 0.2),
        memory_same_category=data_cfg.get("memory_same_category", 0.3),
        memory_think_ratio=data_cfg.get("memory_think_ratio", 0.0),
        memory_think_lang=data_cfg.get("memory_think_lang", "en"),
        ratio_fact=data_cfg.get("ratio_fact", 0.55),
        ratio_continuation=data_cfg.get("ratio_continuation", 0.20),
        ratio_heartbeat=data_cfg.get("ratio_heartbeat", 0.15),
        ratio_logic=data_cfg.get("ratio_logic", 0.10),
        gen_batch_size=data_cfg.get("gen_batch_size", 16),
    )


def main():
    parser = argparse.ArgumentParser(description="统一数据生成")
    parser.add_argument("--config", type=str, required=True, help="课程配置文件")
    parser.add_argument("--stage", type=str, default=None, help="指定阶段名")
    parser.add_argument("--all", action="store_true", help="生成所有阶段数据")
    parser.add_argument("--force", action="store_true", help="强制重新生成 (忽略缓存)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="backbone 路径 (think 数据生成用)")
    args = parser.parse_args()

    from xinhe.model.config import XinheConfig
    config, curriculum = XinheConfig.from_yaml(args.config)

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
        data_type = stage.get("data", {}).get("type", "memory")
        print(f"\n{'='*50}")
        print(f"  生成数据: {name} (type={data_type})")
        print(f"{'='*50}")
        train_path, val_path = generate_stage_data(
            stage, name, force=args.force,
            model_path=args.model_path or config.backbone_model_path,
        )
        print(f"  → {train_path}")
        print(f"  → {val_path}")


if __name__ == "__main__":
    main()
