"""
v8 数据生成入口 —— Stage 0 / Stage 1 dispatcher。

读取 curriculum_v8 配置，按 stage.data.stage_kind 分发到 stage0 / stage1 generator。

用法:
    python scripts/generate_data.py --config configs/persona_unified_v8_0.8b.yaml --stage 0_atomic_skeletons
    python scripts/generate_data.py --config configs/persona_unified_v8_0.8b.yaml --all

    # smoke 量（覆写 yaml 数量，用于快速验证）
    python scripts/generate_data.py --config ... --stage 1_5beat_natural --n-train 500 --n-val 50

    # 强制重生成（默认 Stage 1 自动 resume；Stage 0 用 --force 覆盖）
    python scripts/generate_data.py --config ... --stage 1_5beat_natural --force

  缓存感知: 默认若 train.jsonl + val.jsonl 已存在则跳过。--force 强制重生成。
  Stage 1 内部支持流式写 + 断点续跑（resume=True 是默认），中途崩中断不丢已写。
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.config import validate_stage_config
from xinhe.data.stage0.runner import generate_stage0_dataset
from xinhe.data.stage1.driver import generate_stage1_dataset
from xinhe.data.stage2.mixer import generate_stage2_dataset


def _generate_stage0(stage_cfg: dict, *, split: str, n_override: int | None = None) -> tuple[int, int]:
    data_cfg = stage_cfg.get("data", {})
    training_cfg = stage_cfg.get("training", {})
    out_dir = Path(data_cfg.get("out_dir", "data/v8/stage0"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_override if n_override is not None else data_cfg.get(f"num_{split}", data_cfg.get("num_train", 1000))
    seed_base = int(data_cfg.get("seed", 42))
    seed = seed_base + (0 if split == "train" else 1)
    out_path = out_dir / f"{split}.jsonl"
    rejected = out_dir / f"{split}.rejected.jsonl"
    max_turns = int(training_cfg["max_turns_per_episode"])  # validate_stage_config 已确保存在
    print(f"  [stage0/{split}] 生成 {n} 条 → {out_path} (max_turns={max_turns})")
    return generate_stage0_dataset(
        out_path,
        n_samples=n,
        seed=seed,
        skeleton_weights=data_cfg.get("skeleton_weights"),
        force_relation=data_cfg.get("force_relation"),
        dict_split=split,
        distance_distribution=data_cfg.get("distance_bucket"),
        rejected_path=rejected,
        progress_every=max(1, n // 10),
        max_turns=max_turns,
    )


def _generate_stage1(stage_cfg: dict, *, split: str, n_override: int | None = None,
                     model_override: str | None = None,
                     out_suffix: str = "",
                     seed_offset: int = 0,
                     workers_override: int | None = None) -> tuple[int, int]:
    data_cfg = stage_cfg.get("data", {})
    out_dir = Path(data_cfg.get("out_dir", "data/v8/stage1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_override if n_override is not None else data_cfg.get(f"num_{split}", data_cfg.get("num_train", 100))
    seed_base = int(data_cfg.get("seed", 43))
    seed = seed_base + (0 if split == "train" else 1) + seed_offset
    out_path = out_dir / f"{split}{out_suffix}.jsonl"
    rejected = out_dir / f"{split}{out_suffix}.rejected.jsonl"
    model = model_override or data_cfg.get("model", "deepseek-v4-flash")
    print(f"  [stage1/{split}] 生成 {n} 条 → {out_path} (model={model})")
    return generate_stage1_dataset(
        out_path,
        n_samples=n,
        seed=seed,
        mix=data_cfg.get("mix"),
        dict_split=split,
        n_canonical_range=tuple(data_cfg.get("n_canonical_range", [1, 3])),
        n_turns_range=tuple(data_cfg.get("n_turns_range", [10, 14])),
        beat3_min_turns=int(data_cfg.get("beat3_min_turns", 1)),
        beat3_min_chars=int(data_cfg.get("beat3_min_chars", 500)),
        beat3_chars_tolerance=float(data_cfg.get("beat3_chars_tolerance", 0.8)),
        workers=workers_override if workers_override is not None else int(data_cfg.get("workers", 4)),
        model=model,
        rejected_path=rejected,
    )


def generate_stage(stage_cfg: dict, force: bool = False,
                   n_train_override: int | None = None,
                   n_val_override: int | None = None,
                   model_override: str | None = None,
                   out_suffix: str = "",
                   seed_offset: int = 0,
                   workers_override: int | None = None) -> None:
    name = stage_cfg.get("name", "?")
    # 启动期校验 + 派生：配错在这里立刻报，不让生成器跑出会被 dataloader 截的数据
    validate_stage_config(name, stage_cfg)

    data_cfg = stage_cfg.get("data", {})
    kind = data_cfg.get("stage_kind", "stage0")
    out_dir = Path(data_cfg.get("out_dir", f"data/v8/{kind}"))

    train_path = out_dir / f"train{out_suffix}.jsonl"
    val_path = out_dir / f"val{out_suffix}.jsonl"

    expected_train = n_train_override if n_train_override is not None else data_cfg.get("num_train", 0)
    # 副 driver(out_suffix 非空) 不要求 val 文件存在,只检查 train
    cache_check = train_path.exists() if out_suffix else (train_path.exists() and val_path.exists())
    if not force and cache_check:
        with open(train_path, "r", encoding="utf-8") as f:
            actual = sum(1 for ln in f if ln.strip())
        if expected_train > 0 and actual >= expected_train * 0.95:
            print(f"  [缓存] {name} 已生成 ({actual} 条)，跳过。--force 强制重生成。")
            return

    print(f"\n=== {name} ({kind}) ===")
    if kind == "stage0":
        _generate_stage0(stage_cfg, split="train", n_override=n_train_override)
        _generate_stage0(stage_cfg, split="val", n_override=n_val_override)
    elif kind == "stage1":
        _generate_stage1(stage_cfg, split="train", n_override=n_train_override,
                         model_override=model_override, out_suffix=out_suffix, seed_offset=seed_offset,
                         workers_override=workers_override)
        # 副 driver 不生成 val(避免覆盖主 driver 的 val.jsonl)
        if not out_suffix:
            _generate_stage1(stage_cfg, split="val", n_override=n_val_override, model_override=model_override,
                             workers_override=workers_override)
    elif kind == "stage2":
        _generate_stage2(stage_cfg, split="train", n_override=n_train_override)
        _generate_stage2(stage_cfg, split="val", n_override=n_val_override)
    else:
        raise ValueError(f"未知 stage_kind: {kind}")


def _generate_stage2(stage_cfg: dict, *, split: str, n_override: int | None = None) -> tuple[int, int]:
    """Stage 2 联合巩固：从 stage 0/stage 1 jsonl 按比例抽样合并。

    yaml 例:
      sources:
        - {path: "data/v8/stage0/train.jsonl", ratio: 0.30, tag: "stage0"}
        - {path: "data/v8/stage1/train.jsonl", ratio: 0.70, tag: "stage1"}
    val 时自动把 path 里的 train → val（每个 source 独立切分,共用比例）。
    """
    data_cfg = stage_cfg.get("data", {})
    out_dir = Path(data_cfg.get("out_dir", "data/v8/stage2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_override if n_override is not None else data_cfg.get(f"num_{split}", data_cfg.get("num_train", 100))
    seed_base = int(data_cfg.get("seed", 44))
    seed = seed_base + (0 if split == "train" else 1)
    out_path = out_dir / f"{split}.jsonl"

    sources_raw = data_cfg.get("sources") or []
    if not sources_raw:
        raise ValueError("stage2 需配置 sources（每个含 path/ratio/tag）")
    # train→val 路径自动替换
    sources = []
    for s in sources_raw:
        p = s["path"]
        if split == "val":
            p = p.replace("train.jsonl", "val.jsonl")
        sources.append({"path": p, "ratio": s.get("ratio", 0), "tag": s.get("tag", "?")})

    print(f"  [stage2/{split}] 生成 {n} 条 → {out_path}")
    return generate_stage2_dataset(out_path, sources=sources, n_samples=n, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="v8 数据生成")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None, help="指定 stage 名")
    parser.add_argument("--all", action="store_true", help="生成所有 stage")
    parser.add_argument("--force", action="store_true", help="强制重生成（覆盖已有）")
    parser.add_argument("--n-train", type=int, default=None,
                        help="覆写 yaml 里 num_train（用于 smoke 量）")
    parser.add_argument("--n-val", type=int, default=None,
                        help="覆写 yaml 里 num_val（用于 smoke 量）")
    parser.add_argument("--model", type=str, default=None,
                        help="覆写 yaml 里 model（仅 stage1 生效）。含 '/' 走 OpenRouter，否则 DeepSeek。")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件后缀（仅 stage1 train），eg '_or' → train_or.jsonl，副 driver 用以并行写不同文件。")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="seed 偏移（避免副 driver 与主 driver 撞种），eg 1000。")
    parser.add_argument("--workers", type=int, default=None,
                        help="覆写 yaml 里 workers（仅 stage1 生效）。OR 限流时一键降并发。")
    args = parser.parse_args()

    from xinhe.model.config import XinheConfig
    _config, curriculum = XinheConfig.from_yaml(args.config)

    if not curriculum:
        print("配置文件没有 curriculum 段")
        sys.exit(1)

    if args.stage:
        target = [s for s in curriculum if s.get("name") == args.stage]
        if not target:
            names = [s.get("name") for s in curriculum]
            print(f"未知 stage: {args.stage!r}（可选: {names}）")
            sys.exit(1)
        stages_to_gen = target
    elif args.all:
        stages_to_gen = curriculum
    else:
        parser.print_help()
        sys.exit(1)

    for stage in stages_to_gen:
        generate_stage(stage, force=args.force,
                       n_train_override=args.n_train,
                       n_val_override=args.n_val,
                       model_override=args.model,
                       out_suffix=args.out_suffix,
                       seed_offset=args.seed_offset,
                       workers_override=args.workers)


if __name__ == "__main__":
    main()
