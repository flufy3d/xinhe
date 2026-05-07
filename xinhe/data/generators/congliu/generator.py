"""CongliuGenerator (v9.5) — 中文 R1 推理蒸馏数据,基于 Congliu/Chinese-DeepSeek-R1-Distill-data-110k

数据来源:HuggingFace `Congliu/Chinese-DeepSeek-R1-Distill-data-110k` 的 jsonl/parquet 分支。
原始字段(常见):input/instruction + output/content + reasoning_content + repo_name 等。
本实现兼容多种字段命名,优先级:input > instruction > prompt;output > content > response。

Episode 格式:
  - 简单模式(默认):单 turn 长 QA — user=input,assistant=output(含 think)
    适合训练长 reasoning 输出能力。
  - 多 turn 模式(待加,本版未启):把 reasoning 拆 chunk,前 N-1 turn 喂 reasoning,末 turn 给答案。

raw_path 只能从 CLI(--congliu-path)注入,yaml 不持久化。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from xinhe.data.io import write_jsonl
from xinhe.data.schema import Sample
from xinhe.data.generators.base import Generator, GenerateRequest


# 多种可能的字段命名(由 HF 上游数据集 schema 决定),按优先级 fallback
_USER_FIELD_CANDIDATES = ["input", "instruction", "prompt", "question", "query"]
_ASST_FIELD_CANDIDATES = ["output", "content", "response", "answer", "completion"]


def _extract_user(row: dict) -> Optional[str]:
    for k in _USER_FIELD_CANDIDATES:
        v = row.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_asst(row: dict) -> Optional[str]:
    """构造 R1 完整输出。Congliu schema 的 reasoning_content (think) 与 content (answer) 分离,
    若两者都存在,拼成 <think>{reasoning}</think>\\n{answer} 标准 R1 格式;
    否则按其他常见字段 fallback(单字段 output/response 等)。"""
    reasoning = row.get("reasoning_content")
    content = row.get("content")
    if (reasoning and isinstance(reasoning, str) and reasoning.strip()
            and content and isinstance(content, str) and content.strip()):
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
    # fallback:其他单字段命名
    for k in _ASST_FIELD_CANDIDATES:
        v = row.get(k)
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _load_rows_from_path(path: Path) -> list[dict]:
    """支持 jsonl / parquet / 目录(自动 glob)。"""
    if path.is_dir():
        # 目录:扫所有 jsonl + parquet
        rows = []
        for child in sorted(path.glob("*.jsonl")):
            rows.extend(_load_rows_from_path(child))
        for child in sorted(path.glob("*.parquet")):
            rows.extend(_load_rows_from_path(child))
        return rows
    if not path.exists():
        raise FileNotFoundError(f"[congliu] 文件不存在: {path}")
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return rows
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        return pf.read().to_pylist()
    raise ValueError(f"[congliu] 不支持的格式: {path.suffix}")


class CongliuGenerator(Generator):
    name = "congliu"

    def __init__(
        self,
        *,
        max_turns: int,
        raw_path: Optional[str] = None,    # CLI 注入
        max_user_chars: int = 2000,        # 单条 user 字符数上限(过长丢弃)
        max_asst_chars: int = 6000,        # 单条 assistant 字符数上限(R1 通常 1-3K 字)
        min_asst_chars: int = 100,         # 太短的丢弃(质量过滤)
    ):
        self.max_turns = max_turns
        self.raw_path = raw_path
        self.max_user_chars = max_user_chars
        self.max_asst_chars = max_asst_chars
        self.min_asst_chars = min_asst_chars
        self._rows: Optional[list[dict]] = None

    def _ensure_rows(self):
        if self._rows is not None:
            return
        if not self.raw_path:
            raise RuntimeError(
                "[congliu] 缺少 raw_path。从 CLI 传入:\n"
                "  python scripts/generate_data.py ... --congliu-path /path/to/congliu_raw"
            )
        path = Path(self.raw_path)
        self._rows = _load_rows_from_path(path)
        print(f"  [congliu] {path}: 读取 {len(self._rows)} 行")

    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]:
        self._ensure_rows()
        n_samples = req.n_samples
        rng = random.Random(req.seed)

        rows = list(self._rows)
        rng.shuffle(rows)

        kept_count = 0
        rejected_count = 0

        def _gen():
            nonlocal kept_count, rejected_count
            sample_idx = 0
            for row in rows:
                if n_samples > 0 and kept_count >= n_samples:
                    break
                user = _extract_user(row)
                asst = _extract_asst(row)
                if user is None or asst is None:
                    rejected_count += 1
                    continue
                if len(user) > self.max_user_chars or len(asst) > self.max_asst_chars:
                    rejected_count += 1
                    continue
                if len(asst) < self.min_asst_chars:
                    rejected_count += 1
                    continue

                conversations = [
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": asst,
                        "train_loss": "true",
                        "value": None,
                        "value_span": [],
                        "value_tier": None,
                        "weight_per_span": 0.0,
                    },
                ]
                yield Sample(
                    sample_id=f"congliu_{sample_idx:08x}",
                    stage="congliu",
                    skeleton_id=None,
                    meta={
                        "user_chars": len(user),
                        "asst_chars": len(asst),
                    },
                    conversations=conversations,
                )
                kept_count += 1
                sample_idx += 1

        kept, rejected = write_jsonl(_gen(), req.out_path, rejected_path=req.rejected_path)
        return kept, rejected
