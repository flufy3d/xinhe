"""
数据 IO:JSONL 读写 + 写盘前 schema 校验。

替代旧 samplers.episode_to_jsonl（保留语义但统一从 Sample dataclass 走）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from xinhe.data.schema import (
    Sample,
    SchemaError,
    validate_sample,
)


def write_jsonl(
    samples: Iterable[dict | Sample],
    out_path: str | Path,
    *,
    rejected_path: str | Path | None = None,
) -> tuple[int, int]:
    """把 sample 流写到 jsonl。schema 不通过的写到 rejected_path（如提供）。

    Returns:
        (n_ok, n_rejected)
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej = Path(rejected_path) if rejected_path else None
    if rej is not None:
        rej.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_rej = 0
    rej_fp = open(rej, "w", encoding="utf-8") if rej else None
    try:
        with open(out, "w", encoding="utf-8") as fp:
            for s in samples:
                d = s.to_dict() if isinstance(s, Sample) else s
                try:
                    validate_sample(d)
                except SchemaError as e:
                    n_rej += 1
                    if rej_fp is not None:
                        rej_fp.write(
                            json.dumps(
                                {"reason": str(e), "sample": d}, ensure_ascii=False
                            ) + "\n"
                        )
                    continue
                fp.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_ok += 1
    finally:
        if rej_fp is not None:
            rej_fp.close()
    return n_ok, n_rej


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ── 兼容旧接口（curriculum_data 删除前过渡用，最后清掉）──

def episode_to_jsonl_legacy(turns: list[dict]) -> str:
    """旧 samplers.episode_to_jsonl 等价接口。

    把 [{"user": ..., "assistant": ..., "train_loss": bool, "value": str|list}, ...]
    转成 v7 格式的 jsonl 单行。仅在删除旧代码前过渡保留。
    """
    conversations = []
    for turn in turns:
        conversations.append({"role": "user", "content": turn["user"]})
        entry = {
            "role": "assistant",
            "content": turn["assistant"],
            "train_loss": turn.get("train_loss", True),
        }
        if "value" in turn:
            entry["value"] = turn["value"]
        conversations.append(entry)
    return json.dumps({"conversations": conversations}, ensure_ascii=False)
