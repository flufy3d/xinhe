"""Stage 2 联合巩固数据生成：按比例从 stage0/stage1 jsonl 抽样合并。

目的：对抗 stage 1 训完后 stage 0 能力的灾难性遗忘。
stage 0 训"原子事件骨架"（A/B/C/D/E/F/G/H/I/J/K/L/M），stage 1 训"自然 5-Beat 对话"。
stage 2 按可配置比例混两边数据，让 backbone + W 在两种语言风格上同时收敛。

不重新调 LLM、不重生成；纯粹从已有 jsonl 抽样合并。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"stage2 mixer: 源文件不存在 {path}")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        raise ValueError(f"stage2 mixer: 源文件为空 {path}")
    return out


def generate_stage2_dataset(
    out_path: str | Path,
    *,
    sources: list[dict],
    n_samples: int,
    seed: int,
    resume: bool = True,
) -> tuple[int, int]:
    """sources 形如:
        [{"path": "data/v8/stage0/train.jsonl", "ratio": 0.3,  "tag": "stage0"},
         {"path": "data/v8/stage1/train.jsonl", "ratio": 0.7, "tag": "stage1"}]

    每个 source 抽 n_samples * ratio 条（可重复抽样，源不足时循环复用）。
    最后整体 shuffle 写盘。每条 sample meta 加 'stage2_source' 标记来源。

    Args:
        resume: True 时若 out_path 已存在且条数 >= n_samples 则跳过
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if resume and out.exists():
        with open(out, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
        if existing >= n_samples:
            print(f"[stage2] 已满 ({existing} ≥ {n_samples})，跳过 {out}")
            return existing, 0

    # 比例归一化（容错：用户可能写 30/70 而非 0.3/0.7）
    total = sum(s.get("ratio", 0) for s in sources)
    if total <= 0:
        raise ValueError(f"stage2 mixer: sources 比例总和 = {total}，必须 > 0")
    sources_norm = [
        {"path": s["path"], "ratio": s.get("ratio", 0) / total,
         "tag": s.get("tag", Path(s["path"]).parent.name)}
        for s in sources
    ]

    print(f"[stage2] 联合巩固: 目标 {n_samples} 条 (sources={[(s['tag'], round(s['ratio'], 3)) for s in sources_norm]})")

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
            # 源不足循环抽样（保比例）
            taken = (loaded * (1 + n_take // max(1, len(loaded))))[:n_take]
        for s in taken:
            meta = dict(s.get("meta", {}))
            meta["stage2_source"] = src["tag"]
            s["meta"] = meta
        pool.extend(taken)
        print(f"  [stage2] {src['tag']}: 抽 {len(taken)} 条 (源 {len(loaded)} 条, ratio={src['ratio']:.3f})")

    rng.shuffle(pool)
    pool = pool[:n_samples]   # 容差导致总数 ±1，截到目标

    with open(out, "w", encoding="utf-8") as f:
        for s in pool:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[stage2] 落盘 {len(pool)}/{n_samples} → {out}")
    return len(pool), 0
