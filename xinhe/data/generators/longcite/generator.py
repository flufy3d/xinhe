"""LongCiteGenerator (v9) — 长上下文 QA + 引用,基于 zai-org/LongCite-45k

数据来源:HuggingFace `zai-org/LongCite-45k` 的 parquet 分支
  refs/convert/parquet/default/partial-train/0000.parquet (~240 MB,~4065 行)
单 parquet 解压 ≈ 500 MB jsonl,符合 P-cap mini 数据规模。

原始字段:
  - prompt:英文 task 指令 + [Document Start] long_context (含 <C0>-<C{N}> 句标记) [Document End] + question
  - response:<statement>...<cite>[s-e]</cite></statement> 结构

Episode 切法(多 turn):
  - long_context 按 chunk_chars 切 N chunks
  - 前 (turns_per_episode-1) 个 turn:user="" asst=chunk(纯长上下文吸收)
  - 末 1 个 turn:user=question asst=response(QA 应用)

parquet_path 只能从 CLI(--parquet-path)注入,yaml 不持久化。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from xinhe.data.io import write_jsonl
from xinhe.data.schema import Sample
from xinhe.data.generators.base import Generator, GenerateRequest


_DOC_START = "[Document Start]"
_DOC_END = "[Document End]"


def _parse_prompt(prompt: str) -> tuple[str, str] | None:
    """从 prompt 解析 (long_context, question)。
    结构:[英文 instruction] + [Document Start] + long_context + [Document End] + question
    若无 [Document Start/End] 标志,返回 None(跳过此样本)。
    """
    if _DOC_START not in prompt or _DOC_END not in prompt:
        return None
    _, after_start = prompt.split(_DOC_START, 1)
    doc, after_doc = after_start.split(_DOC_END, 1)
    long_context = doc.strip().lstrip("\n").strip()
    question = after_doc.strip()
    if not long_context or not question:
        return None
    return long_context, question


def _split_chunks(text: str, chunk_chars: int) -> list[str]:
    """字符级切 chunks,末段不足半 chunk 丢弃。"""
    out: list[str] = []
    for i in range(0, len(text), chunk_chars):
        piece = text[i : i + chunk_chars]
        if i + chunk_chars >= len(text) and len(piece) < chunk_chars // 2:
            continue
        out.append(piece)
    return out


class LongCiteGenerator(Generator):
    name = "longcite"

    def __init__(
        self,
        *,
        max_turns: int,
        parquet_path: Optional[str] = None,    # CLI 注入,yaml 不持久化
        chunk_chars: int = 350,                 # ≈ 500 token/chunk
        turns_per_episode: int = 8,             # (turns_per_episode-1) 个 chunks + 1 个 QA turn
        min_long_context_chars: int = 1000,     # 太短的 prompt 丢弃(QA 学习信号弱)
        episodes_per_sample: int = 1,           # >1 时滑动窗口生成多 episode/sample,提升数据利用率
    ):
        self.max_turns = max_turns
        self.parquet_path = parquet_path
        self.chunk_chars = chunk_chars
        self.turns_per_episode = min(turns_per_episode, max_turns)
        self.min_long_context_chars = min_long_context_chars
        self.episodes_per_sample = max(1, int(episodes_per_sample))
        self._rows: Optional[list[dict]] = None

    def _ensure_rows(self):
        if self._rows is not None:
            return
        if not self.parquet_path:
            raise RuntimeError(
                "[longcite] 缺少 parquet_path。从 CLI 传入:\n"
                "  python scripts/generate_data.py ... --parquet-path /path/to/0000.parquet"
            )
        path = Path(self.parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"[longcite] parquet 不存在: {path}")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        # 一次性读全表(单个 parquet ~240 MB,内存能装)
        self._rows = pf.read().to_pylist()
        print(f"  [longcite] {path.name}: 读取 {len(self._rows)} 行")

    def _resolve_n_samples(self, requested: int) -> int:
        if requested == -1:
            self._ensure_rows()
            adaptive = len(self._rows) * self.episodes_per_sample
            print(f"  [longcite] num_samples=-1 → 自适应 {adaptive} "
                  f"(rows={len(self._rows)} × episodes_per_sample={self.episodes_per_sample})")
            return adaptive
        return requested

    def _is_cached(self, req: GenerateRequest) -> bool:
        if req.n_samples == -1:
            if not req.out_path.exists():
                return False
            from dataclasses import replace
            req2 = replace(req, n_samples=self._resolve_n_samples(-1))
            return super()._is_cached(req2)
        return super()._is_cached(req)

    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]:
        self._ensure_rows()
        n_samples = self._resolve_n_samples(req.n_samples)
        rng = random.Random(req.seed)

        # split 一致性:同 seed train/val 不同 row 子集
        rows = list(self._rows)
        rng.shuffle(rows)

        kept_count = 0
        rejected_count = 0
        n_leading = self.turns_per_episode - 1

        def _gen():
            nonlocal kept_count, rejected_count
            sample_idx = 0
            for row in rows:
                if kept_count >= n_samples:
                    break

                prompt = row.get("prompt", "")
                response = row.get("response", "")
                if not prompt or not response:
                    rejected_count += 1
                    continue

                parsed = _parse_prompt(prompt)
                if parsed is None:
                    rejected_count += 1
                    continue
                long_context, question = parsed

                if len(long_context) < self.min_long_context_chars:
                    rejected_count += 1
                    continue

                chunks = _split_chunks(long_context, self.chunk_chars)
                if len(chunks) < 1:
                    rejected_count += 1
                    continue

                # episodes_per_sample 个滑动窗口起点(均匀分布到 chunks 上)
                # 起点必须保证后面有 ≥ n_leading 个 chunks
                max_start = max(0, len(chunks) - n_leading)
                if self.episodes_per_sample == 1:
                    starts = [0]
                else:
                    if max_start == 0:
                        starts = [0]
                    else:
                        # 均匀分布 episodes_per_sample 个起点(覆盖整个 chunks 列表)
                        step = max(1, max_start // self.episodes_per_sample)
                        starts = [min(i * step, max_start) for i in range(self.episodes_per_sample)]
                        starts = sorted(set(starts))

                for start in starts:
                    if kept_count >= n_samples:
                        break
                    leading_chunks = chunks[start : start + n_leading]
                    if len(leading_chunks) < 1:
                        continue

                    conversations: list[dict] = []
                    for chunk in leading_chunks:
                        conversations.append({"role": "user", "content": ""})
                        conversations.append({
                            "role": "assistant",
                            "content": chunk,
                            "train_loss": "true",
                            "value": None,
                            "value_span": [],
                            "value_tier": None,
                            "weight_per_span": 0.0,
                        })
                    conversations.append({"role": "user", "content": question})
                    conversations.append({
                        "role": "assistant",
                        "content": response,
                        "train_loss": "true",
                        "value": None,
                        "value_span": [],
                        "value_tier": None,
                        "weight_per_span": 0.0,
                    })

                    yield Sample(
                        sample_id=f"longcite_{sample_idx:08x}",
                        stage="longcite",
                        skeleton_id=None,
                        meta={
                            "n_chunks_total": len(chunks),
                            "chunk_start": start,
                            "n_chunks_used": len(leading_chunks),
                            "long_context_chars": len(long_context),
                            "question_chars": len(question),
                            "response_chars": len(response),
                            "chunk_chars": self.chunk_chars,
                        },
                        conversations=conversations,
                    )
                    kept_count += 1
                    sample_idx += 1

        kept, rejected = write_jsonl(_gen(), req.out_path, rejected_path=req.rejected_path)
        return kept, rejected
