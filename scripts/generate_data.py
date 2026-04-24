"""
统一数据生成入口

根据课程阶段配置中的 data.type 自动分发:
  - type: memory  — 调用 generate_memory_data（0a_fact_bootstrap 用）
  - type: persona — 调用 generate_persona_data（0b / stage 1 用）

用法:
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --stage 0b_turn_bootstrap
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --all
    python scripts/generate_data.py --config configs/persona_unified_0.8b.yaml --stage 1_persona_unified_dual --force
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
) -> tuple[str, str]:
    """
    为课程阶段生成数据，返回 (train_path, val_path)。

    根据 data.type 自动分发:
      - "memory"  — 调用 generate_memory_data（每次重新生成）
      - "persona" — 调用 generate_persona_data（已生成则用缓存）
    """
    data_cfg = stage.get("data", {})
    data_type = data_cfg.get("type", "memory")
    out_dir = data_cfg.get("out_dir", f"data/curriculum/{stage_name}")

    if data_type == "memory":
        return _generate_memory(data_cfg, out_dir)

    elif data_type == "persona":
        return _generate_persona(data_cfg, out_dir, force)

    else:
        raise ValueError(f"未知的数据类型: {data_type}")


def _generate_memory(data_cfg: dict, out_dir: str) -> tuple[str, str]:
    """生成记忆训练数据"""
    from xinhe.data.generate_memory_data import generate_data

    return generate_data(
        out_dir=out_dir,
        num_train=data_cfg.get("num_train", 5000),
        num_val=data_cfg.get("num_val", 100),
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
        categories=data_cfg.get("categories", None),
        seed=data_cfg.get("seed", 42),
    )


def _generate_persona(data_cfg: dict, out_dir: str, force: bool) -> tuple[str, str]:
    """生成 persona 统一训练数据（缓存感知：已存在则跳过）"""
    train_path = str(Path(out_dir) / "train.jsonl")
    val_path = str(Path(out_dir) / "val.jsonl")

    # 缓存检查
    if not force and Path(train_path).exists() and Path(val_path).exists():
        expected = data_cfg.get("num_train", 40000)
        with open(train_path, "r", encoding="utf-8") as f:
            actual = sum(1 for ln in f if ln.strip())
        if actual >= expected * 0.95:
            print(f"  [缓存] 使用已有数据: {out_dir} ({actual} 条)")
            return train_path, val_path

    from xinhe.data.generate_persona_data import generate_data as _gen_persona

    return _gen_persona(
        out_dir=out_dir,
        num_train=data_cfg.get("num_train", 40000),
        num_val=data_cfg.get("num_val", 500),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        turn_mix=data_cfg.get("turn_mix", None),
        min_turns=data_cfg.get("min_turns", 12),
        max_turns=data_cfg.get("max_turns", 20),
        seed=data_cfg.get("seed", 42),
        stress_retention_ratio=data_cfg.get("stress_retention_ratio", 0.0),
        multi_slot_retention_ratio=data_cfg.get("multi_slot_retention_ratio", 0.0),
        variable_distance_ratio=data_cfg.get("variable_distance_ratio", 0.0),
        fact_vs_transient_ratio=data_cfg.get("fact_vs_transient_ratio", 0.0),
        rapid_overwrite_ratio=data_cfg.get("rapid_overwrite_ratio", 0.0),
        decay_awareness_ratio=data_cfg.get("decay_awareness_ratio", 0.0),
        verbatim_recall_ratio=data_cfg.get("verbatim_recall_ratio", 0.0),
        meta_recall_ratio=data_cfg.get("meta_recall_ratio", 0.0),
        adversarial_temporal_ratio=data_cfg.get("adversarial_temporal_ratio", 0.0),
    )


def main():
    parser = argparse.ArgumentParser(description="统一数据生成")
    parser.add_argument("--config", type=str, required=True, help="课程配置文件")
    parser.add_argument("--stage", type=str, default=None, help="指定阶段名")
    parser.add_argument("--all", action="store_true", help="生成所有阶段数据")
    parser.add_argument("--force", action="store_true", help="强制重新生成 (忽略缓存)")
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
        )
        print(f"  → {train_path}")
        print(f"  → {val_path}")


if __name__ == "__main__":
    main()
