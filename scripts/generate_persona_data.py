"""离线人格注入协议(零号纪元)的数据生成 CLI。

与 scripts/generate_data.py(GENERATORS dispatcher)解耦:产出张量缓存,不写 jsonl。

用法:
    python scripts/generate_persona_data.py \
        --config configs/persona_inject.yaml \
        --novel-path /path/to/novel.txt \
        [--force]

novel_path 只能从 CLI 注入(yaml 不持久化路径,与 NovelGenerator 同样规约)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.generators.persona_inject import PersonaInjectGenerator


def _load_yaml_with_base(path: Path) -> dict:
    """简化版 base 继承 —— 我们不需要 XinheConfig 的全部 dataclass 校验,
    只要能从 base.yaml 拿到 backbone.model_path。

    与 XinheConfig._load_and_merge 一致:遇到 base 字段就向上递归合并。
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in raw:
        base_path = path.parent / raw.pop("base")
        base = _load_yaml_with_base(base_path)
        # 深合并:子配置覆盖 base
        for k, v in raw.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
        raw = base
    return raw


def main():
    p = argparse.ArgumentParser(description="离线人格注入协议数据生成")
    p.add_argument("--config", required=True, help="configs/persona_inject.yaml")
    p.add_argument("--novel-path", required=True, help="小说 txt 路径(不持久化)")
    p.add_argument(
        "--out-name", default=None,
        help="<out_dir>/ 下的子目录名,跨平台安全(默认 sanitize 文件名;"
             "中文/特殊字符全被清空时 fallback 到 sha256 前缀)",
    )
    p.add_argument("--force", action="store_true", help="忽略缓存重生成")
    args = p.parse_args()

    cfg = _load_yaml_with_base(Path(args.config))

    backbone_model_path = cfg.get("backbone", {}).get("model_path")
    if not backbone_model_path:
        raise ValueError(f"{args.config}: backbone.model_path 缺失(检查 base 继承)")

    pi = cfg.get("persona_inject")
    if not pi:
        raise ValueError(f"{args.config}: persona_inject 段缺失")

    layer_indices = pi.get("layer_indices")
    if not layer_indices:
        raise ValueError(f"{args.config}: persona_inject.layer_indices 必须显式指定")

    gen = PersonaInjectGenerator(
        novel_path=args.novel_path,
        out_dir=pi.get("out_dir", "data/persona"),
        backbone_model_path=backbone_model_path,
        layer_indices=layer_indices,
        out_name=args.out_name or pi.get("out_name"),
        block_size=int(pi.get("block_size", 192)),
        chapter_aware=bool(pi.get("chapter_aware", True)),
        extract_batch_size=int(pi.get("extract_batch_size", 8)),
        extract_dtype=pi.get("extract_dtype", "bfloat16"),
        device=pi.get("device", "cuda"),
        chapter_pattern=pi.get("chapter_pattern",
            r"^第[一二三四五六七八九十百千零〇\d]+[章回节]"),
        min_dialog_density=float(pi.get("min_dialog_density", 0.0)),
    )

    print(f"=== persona_inject: {Path(args.novel_path).name} → {gen.out_dir} ===")
    manifest = gen.generate(force=args.force)
    print(f"\n[manifest] {gen.manifest_path}")
    print(f"  out_name={manifest['out_name']!r}, n_blocks={manifest['n_blocks']}, "
          f"n_pairs={manifest['n_pairs']}, n_layers_kept={manifest['n_layers_kept']}")


if __name__ == "__main__":
    main()
