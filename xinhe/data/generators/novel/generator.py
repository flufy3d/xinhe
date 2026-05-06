"""NovelGenerator (v9 raw chunking) — 长上下文 next-token,支持单文件 or 目录(多本)

每 episode = N 个连续章内 chunk(章内不跨,避免 fast-weights 跨章错乱)。
每 chunk 当一个 turn 的 assistant content,user 留空占位。labels 全部 token
next-token,无 VALUE 加权。最贴 Titans-MAC 训练任务。

novel_path 只能从 CLI(--novel-path)注入,可指向单个 .txt 或包含多本的目录。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from xinhe.data.io import write_jsonl
from xinhe.data.schema import Sample
from xinhe.data.generators.base import Generator, GenerateRequest
from xinhe.data.generators.novel.novel_loader import (
    DEFAULT_CHAPTER_PATTERN,
    NovelIndex,
    load_paragraphs,
)


def _chunks_per_chapter(idx: NovelIndex, *, chunk_chars: int) -> dict[int, list[str]]:
    """把每章段落拼接后按 chunk_chars 切成 raw chunk(字符级)。

    中文 1 char ≈ 1.5 token,所以 chunk_chars=350 ≈ 500 token/chunk(留 12 token 给
    ChatML/empty-user prefix,turn_max_tokens=512 时 chunk 净占满)。

    最末不足半 chunk 丢弃;每章必须至少 2 个 chunk 才入选。
    """
    chap_text: dict[int, list[str]] = {}
    for i, p in enumerate(idx.paragraphs):
        chap_id = idx.chapter_id_of[i]
        chap_text.setdefault(chap_id, []).append(p)

    chunks: dict[int, list[str]] = {}
    for chap_id, paras in chap_text.items():
        text = "\n".join(paras)
        out: list[str] = []
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars]
            if i + chunk_chars >= len(text) and len(piece) < chunk_chars // 2:
                continue   # 章末不足半 chunk 丢
            out.append(piece)
        if len(out) >= 2:
            chunks[chap_id] = out
    return chunks


def _discover_novels(novel_path: str) -> list[tuple[str, Path]]:
    """novel_path 是文件或目录。返回 [(novel_name, file_path), ...]。"""
    p = Path(novel_path)
    if p.is_file():
        return [(p.stem, p)]
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            raise RuntimeError(f"[novel] 目录 {p} 下无 .txt 文件")
        return [(f.stem, f) for f in files]
    raise FileNotFoundError(f"[novel] novel_path 不是文件也不是目录: {novel_path}")


class NovelGenerator(Generator):
    name = "novel"

    def __init__(
        self,
        *,
        max_turns: int,
        novel_path: Optional[str] = None,
        chunk_chars: int = 350,           # ≈ 500 token/chunk(中文 ~1.5 token/char)
        turns_per_episode: int = 8,
        chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
        chapter_patterns: Optional[dict[str, str]] = None,   # novel_name (stem) → regex
    ):
        self.max_turns = max_turns
        self.novel_path = novel_path
        self.chunk_chars = chunk_chars
        self.turns_per_episode = min(turns_per_episode, max_turns)
        self.chapter_pattern = chapter_pattern
        self.chapter_patterns = chapter_patterns or {}
        # _chunks key 是 (novel_name, chapter_id),跨本不冲突
        self._chunks: Optional[dict[tuple[str, int], list[str]]] = None
        self._starts: Optional[list[tuple[tuple[str, int], int]]] = None

    def _ensure_chunks(self):
        if self._chunks is not None:
            return
        if not self.novel_path:
            raise RuntimeError(
                "[novel-v9] 缺少 novel_path。从 CLI 传入:\n"
                "  python scripts/generate_data.py ... --novel-path /path/to/file.txt\n"
                "  或 --novel-path /path/to/dir(目录下所有 *.txt 都会扫)"
            )

        novels = _discover_novels(self.novel_path)
        self._chunks = {}
        per_novel_summary: list[str] = []

        for novel_name, path in novels:
            # 优先用 per-novel pattern(精确匹配 stem),否则用全局默认
            pattern = self.chapter_patterns.get(novel_name, self.chapter_pattern)
            try:
                idx = load_paragraphs(
                    str(path),
                    min_density=0.0,
                    chapter_pattern=pattern,
                )
            except Exception as e:
                print(f"  [novel-v9] {novel_name}: load 失败 ({e}),跳过")
                continue
            chunks_for_novel = _chunks_per_chapter(idx, chunk_chars=self.chunk_chars)
            for chap_id, chunks in chunks_for_novel.items():
                self._chunks[(novel_name, chap_id)] = chunks
            total_chunks = sum(len(cs) for cs in chunks_for_novel.values())
            pat_tag = " [override]" if novel_name in self.chapter_patterns else ""
            per_novel_summary.append(
                f"{novel_name}{pat_tag}: {idx.n_chapters} 章 → {len(chunks_for_novel)} 入选, "
                f"{total_chunks} chunk"
            )

        # 收集所有合法 ((novel_name, chap_id), start_idx)
        starts: list[tuple[tuple[str, int], int]] = []
        for key, chunks in self._chunks.items():
            n = len(chunks) - self.turns_per_episode + 1
            if n > 0:
                starts.extend((key, i) for i in range(n))
        self._starts = starts

        total_chunks_all = sum(len(cs) for cs in self._chunks.values())
        print(f"  [novel-v9] {len(novels)} 本小说,共 {total_chunks_all} chunk(≈{self.chunk_chars} 字/chunk)")
        for line in per_novel_summary:
            print(f"    - {line}")
        print(f"  [novel-v9] {len(starts)} 合法 episode 起点(turns_per_episode={self.turns_per_episode})")
        if not starts:
            raise RuntimeError(
                f"[novel-v9] 没有任何章 ≥ {self.turns_per_episode} chunks。"
                f"考虑降低 turns_per_episode 或 chunk_chars,或换长章节小说。"
            )

    def _resolve_n_samples(self, requested: int) -> int:
        if requested == -1:
            self._ensure_chunks()
            adaptive = len(self._starts)
            print(f"  [novel-v9] num_samples=-1 → 自适应 {adaptive}(覆盖每个起点 1 次)")
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
        self._ensure_chunks()
        n_samples = self._resolve_n_samples(req.n_samples)
        rng = random.Random(req.seed)

        def _gen():
            for sample_idx in range(n_samples):
                key, start = rng.choice(self._starts)
                novel_name, chap_id = key
                chunks = self._chunks[key]
                episode_chunks = chunks[start : start + self.turns_per_episode]

                conversations: list[dict] = []
                for chunk in episode_chunks:
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

                yield Sample(
                    sample_id=f"novel_{sample_idx:08x}",
                    stage="novel",
                    skeleton_id=None,
                    meta={
                        "novel": novel_name,
                        "chapter_id": chap_id,
                        "chunk_start": start,
                        "n_chunks": len(episode_chunks),
                        "chunk_chars": self.chunk_chars,
                    },
                    conversations=conversations,
                )

        kept, rejected = write_jsonl(_gen(), req.out_path, rejected_path=req.rejected_path)
        return kept, rejected
