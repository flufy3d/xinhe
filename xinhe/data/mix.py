"""数据混合:从已有 jsonl 按比例抽样合并(原 stage2)。

不调 LLM、不重生成;纯粹从已有 jsonl 抽样合并,对抗"训了 dialog 后 skeleton 能力遗忘"。
sources 为裸路径(yaml 兼容旧格式),split='val' 时自动把 .../train.jsonl 替换成 .../val.jsonl。
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"mix: 源文件不存在 {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"mix: 源文件为空 {path}")
    return out


def _rewrite_path_for_split(path_str: str, split: str) -> str:
    """split='val' 时把 .../train.jsonl 改 .../val.jsonl;train 时不变。"""
    if split == "val":
        return path_str.replace("train.jsonl", "val.jsonl")
    return path_str


def mix_jsonl(
    out_path: str | Path,
    *,
    sources: list[dict],
    n_samples: int,
    seed: int,
    split: str = "train",
    resume: bool = True,
) -> tuple[int, int]:
    """sources 形如:
        [{"path": "data/skeleton/train.jsonl", "ratio": 0.15, "tag": "skeleton"},
         {"path": "data/dialog/train.jsonl",   "ratio": 0.85, "tag": "dialog"}]

    每个 source 抽 n_samples * ratio 条(可重复抽样,源不足时循环复用)。
    最后整体 shuffle 写盘。每条 sample meta 加 'stage2_source' 标记来源(字段名保留兼容旧数据)。

    Args:
        split: "train" | "val";val 时自动把 sources 里 .../train.jsonl 换 .../val.jsonl
        resume: True 时若 out_path 已存在且条数 >= n_samples 则跳过
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if resume and out.exists():
        with open(out, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        if existing >= n_samples:
            print(f"[mix] 已满 ({existing} ≥ {n_samples}),跳过 {out}")
            return existing, 0

    # 比例归一化(容错:用户可能写 30/70 而非 0.3/0.7)
    total = sum(s.get("ratio", 0) for s in sources)
    if total <= 0:
        raise ValueError(f"mix: sources 比例总和 = {total},必须 > 0")
    sources_norm = [
        {
            "path": _rewrite_path_for_split(s["path"], split),
            "ratio": s.get("ratio", 0) / total,
            "tag": s.get("tag", Path(s["path"]).parent.name),
        }
        for s in sources
    ]

    print(f"[mix/{split}] 目标 {n_samples} 条 (sources={[(s['tag'], round(s['ratio'], 3)) for s in sources_norm]})")

    pool: list[dict] = []
    for src in sources_norm:
        n_take = int(round(n_samples * src["ratio"]))
        if n_take <= 0:
            continue
        loaded = _load_jsonl(Path(src["path"]))
        rng.shuffle(loaded)
        if len(loaded) >= n_take:
            taken = loaded[:n_take]
        else:
            # 源不足循环抽样(保比例)
            taken = (loaded * (1 + n_take // max(1, len(loaded))))[:n_take]
        for s in taken:
            meta = dict(s.get("meta", {}))
            meta["stage2_source"] = src["tag"]    # 字段名保留兼容旧数据
            s["meta"] = meta
        pool.extend(taken)
        print(f"  [mix] {src['tag']}: 抽 {len(taken)} 条 (源 {len(loaded)} 条, ratio={src['ratio']:.3f})")

    rng.shuffle(pool)
    pool = pool[:n_samples]   # 容差导致总数 ±1,截到目标

    with open(out, "w", encoding="utf-8") as f:
        for s in pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[mix] 落盘 {len(pool)}/{n_samples} → {out}")
    return len(pool), 0
