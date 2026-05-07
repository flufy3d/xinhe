"""数据生成入口 —— GENERATORS + mix dispatcher。

读 curriculum 配置,按 stage.data.kind 派发:
  kind in GENERATORS  → 实例化 Generator,跑 train + val
  kind == "mix"       → 调 mix_jsonl 按比例从已有 jsonl 抽样合并

加新生成器(如基于小说提取):
  1. 在 xinhe/data/generators/<kind>/ 写一个 Generator 子类
  2. 在 xinhe/data/generators/__init__.py 注册到 GENERATORS
不需要改本文件。

用法:
    python scripts/generate_data.py --config configs/curriculum.yaml --stage skeleton_pretrain
    python scripts/generate_data.py --config configs/curriculum.yaml --all
    python scripts/generate_data.py --config ... --stage dialog_5beat --n-train 500 --n-val 50
    python scripts/generate_data.py --config ... --stage dialog_5beat --force
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.config import validate_stage_config
from xinhe.data.generators import GENERATORS, GenerateRequest
from xinhe.data.mix import mix_jsonl


# 这些 yaml `data:` 字段是 dispatcher 公共消费,不传入 Generator.__init__
_DATA_RESERVED = {"kind", "out_dir", "num_train", "num_val", "seed", "sources"}


def run_stage(stage_cfg: dict, *,
              force: bool = False,
              n_train: int | None = None,
              n_val: int | None = None,
              model: str | None = None,
              out_suffix: str = "",
              seed_offset: int = 0,
              workers: int | None = None,
              novel_path: str | None = None,
              parquet_path: str | None = None,
              congliu_path: str | None = None) -> None:
    name = stage_cfg.get("name", "?")
    validate_stage_config(name, stage_cfg)

    data = dict(stage_cfg.get("data", {}))           # 拷贝,避免污染原 cfg
    kind = data.get("kind")
    if kind is None:
        raise ValueError(f"stage {name!r}: data.kind 缺失")

    out_dir = Path(data.get("out_dir", f"data/{kind}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_base = int(data.get("seed", 42))
    num_train_yaml = int(data.get("num_train", 0))
    num_val_yaml = int(data.get("num_val", 0))
    max_turns = int(stage_cfg.get("training", {}).get("max_turns_per_episode", 12))

    print(f"\n=== {name} ({kind}) ===")

    if kind in GENERATORS:
        # CLI overrides:仅 dialog 真正消化;skeleton 忽略(不传)
        gen_cfg = {k: v for k, v in data.items() if k not in _DATA_RESERVED}
        gen_cfg["max_turns"] = max_turns
        if kind == "dialog":
            if model is not None:
                gen_cfg["model"] = model
            if workers is not None:
                gen_cfg["workers"] = workers
        if kind == "novel":
            if not novel_path:
                raise ValueError(
                    f"stage {name!r}: kind=novel 需要从 CLI 传入 --novel-path /path/to/novel.txt "
                    "(配置文件不持久化路径)"
                )
            gen_cfg["novel_path"] = novel_path
        if kind == "longcite":
            if not parquet_path:
                raise ValueError(
                    f"stage {name!r}: kind=longcite 需要从 CLI 传入 --parquet-path /path/to/0000.parquet "
                    "(配置文件不持久化路径)"
                )
            gen_cfg["parquet_path"] = parquet_path
        if kind == "congliu":
            if not congliu_path:
                raise ValueError(
                    f"stage {name!r}: kind=congliu 需要从 CLI 传入 --congliu-path /path/to/congliu_raw "
                    "(配置文件不持久化路径)"
                )
            gen_cfg["raw_path"] = congliu_path

        gen = GENERATORS[kind](**gen_cfg)

        for split in ("train", "val"):
            n = (n_train if split == "train" else n_val)
            if n is None:
                n = (num_train_yaml if split == "train" else num_val_yaml)
            # n == -1 是 escape 信号: generator 自决数量(novel coverage 自适应)
            if n == -1:
                pass
            elif n <= 0:
                continue
            suffix = out_suffix if (split == "train" and kind == "dialog") else ""
            seed = seed_base + (0 if split == "train" else 1) \
                + (seed_offset if kind == "dialog" else 0)
            req = GenerateRequest(
                out_path=out_dir / f"{split}{suffix}.jsonl",
                rejected_path=out_dir / f"{split}{suffix}.rejected.jsonl",
                n_samples=n, seed=seed, split=split,
                max_turns=max_turns, force=force,
            )
            print(f"  [{kind}/{split}] 生成 {n} 条 → {req.out_path}")
            gen.generate(req)
            # dialog 副 driver(out_suffix 非空)只写 train,避免覆盖主 driver 的 val.jsonl
            if kind == "dialog" and out_suffix and split == "train":
                break

    elif kind == "mix":
        sources = data.get("sources")
        if not sources:
            raise ValueError(f"stage {name!r}: kind=mix 需配置 sources(每个含 path/ratio/tag)")
        for split in ("train", "val"):
            n = (n_train if split == "train" else n_val)
            if n is None:
                n = (num_train_yaml if split == "train" else num_val_yaml)
            if n <= 0:
                continue
            seed = seed_base + (0 if split == "train" else 1)
            out_path = out_dir / f"{split}.jsonl"
            print(f"  [mix/{split}] 生成 {n} 条 → {out_path}")
            mix_jsonl(out_path, sources=sources, n_samples=n, seed=seed,
                      split=split, resume=not force)

    else:
        raise ValueError(
            f"stage {name!r}: 未知 kind={kind!r} "
            f"(generators={list(GENERATORS)},另支持 'mix')"
        )


def main():
    parser = argparse.ArgumentParser(description="数据生成 dispatcher")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default=None, help="指定 stage 名")
    parser.add_argument("--all", action="store_true", help="生成所有 stage")
    parser.add_argument("--force", action="store_true", help="强制重生成(覆盖已有)")
    parser.add_argument("--n-train", type=int, default=None,
                        help="覆写 yaml 里 num_train(用于 smoke 量)")
    parser.add_argument("--n-val", type=int, default=None,
                        help="覆写 yaml 里 num_val(用于 smoke 量)")
    parser.add_argument("--model", type=str, default=None,
                        help="覆写 yaml 里 model(仅 dialog 生效)。含 '/' 走 OpenRouter,否则 DeepSeek。")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="输出文件后缀(仅 dialog train),eg '_or' → train_or.jsonl,副 driver 用以并行写不同文件。")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="seed 偏移(避免副 driver 与主 driver 撞种),eg 1000。")
    parser.add_argument("--workers", type=int, default=None,
                        help="覆写 yaml 里 workers(仅 dialog 生效)。OR 限流时一键降并发。")
    parser.add_argument("--novel-path", type=str, default=None,
                        help="小说 txt 路径(仅 kind=novel 生效)。配置文件不持久化此路径。")
    parser.add_argument("--parquet-path", type=str, default=None,
                        help="parquet 路径(仅 kind=longcite 生效)。配置文件不持久化此路径。")
    parser.add_argument("--congliu-path", type=str, default=None,
                        help="Congliu raw 数据目录或文件(仅 kind=congliu 生效)。配置文件不持久化此路径。")
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
            print(f"未知 stage: {args.stage!r}(可选: {names})")
            sys.exit(1)
        stages_to_gen = target
    elif args.all:
        stages_to_gen = curriculum
    else:
        parser.print_help()
        sys.exit(1)

    for stage in stages_to_gen:
        run_stage(
            stage,
            force=args.force,
            n_train=args.n_train,
            n_val=args.n_val,
            model=args.model,
            out_suffix=args.out_suffix,
            seed_offset=args.seed_offset,
            workers=args.workers,
            novel_path=args.novel_path,
            parquet_path=args.parquet_path,
            congliu_path=args.congliu_path,
        )


if __name__ == "__main__":
    main()
