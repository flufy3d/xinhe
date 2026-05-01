"""NovelGenerator:基于真实小说的 regex 伪交互数据生成器,无 LLM 调用。

设计理念:
  - user = 引号内纯台词(信息孤立) → 模型只能从 W 拉前后文 anchor
  - assistant = 上 dialog 段后 → 当前 dialog 段(含原文) → 下 dialog 段前
  - 起点随机 + episode 长度随机 → 同段 dialog 出多种切分,堵 LoRA 死记

支持 n_samples=-1 自适应: 自动 = len(dialog_idx) × coverage。

novel_path 只能从 CLI (--novel-path) 注入,yaml 配置不持久化路径。
训练入口 train.py 不会触发生成,数据缺失打 hint 提醒 generate_data.py。
"""
from __future__ import annotations

from typing import Optional

from collections import defaultdict

from xinhe.data.generators.base import Generator, GenerateRequest
from xinhe.data.generators.novel.novel_loader import (
    DEFAULT_CHAPTER_PATTERN,
    NovelIndex,
    load_paragraphs,
)
from xinhe.data.generators.novel.runner import generate_novel_dataset


def _compute_valid_starts(idx: NovelIndex, *, min_remaining: int) -> list[int]:
    """章内剩余 dialog 段 ≥ min_remaining 的位置才算合法起点。

    防止起点贴近章末,episode 凑不够 n_turns(章内不可跨)。
    """
    chap_dialogs: dict[int, list[int]] = defaultdict(list)
    for d in idx.dialog_idx:
        chap_dialogs[idx.chapter_id_of[d]].append(d)
    valid: list[int] = []
    for dlist in chap_dialogs.values():
        if len(dlist) >= min_remaining:
            # 前 (len - min_remaining + 1) 个段:它们后面还有 ≥ min_remaining-1 个 dialog
            valid.extend(dlist[: len(dlist) - min_remaining + 1])
    return valid


class NovelGenerator(Generator):
    name = "novel"

    def __init__(
        self,
        *,
        max_turns: int,
        novel_path: Optional[str] = None,    # CLI 注入,yaml 不持久化
        turns_range: tuple[int, int] = (8, 12),
        coverage: int = 3,
        leading_paras: int = 2,
        turn_max_chars: int = 200,
        min_dialog_density: float = 0.05,
        chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
    ):
        self.max_turns = max_turns
        self.novel_path = novel_path
        self.turns_range = tuple(turns_range)
        self.coverage = coverage
        self.leading_paras = leading_paras
        self.turn_max_chars = turn_max_chars
        self.min_dialog_density = min_dialog_density
        self.chapter_pattern = chapter_pattern
        # 延迟加载: 只有真正生成时才读小说,避免 train.py 启动期就要求路径
        self._index = None

    def _ensure_index(self):
        if self._index is not None:
            return
        if not self.novel_path:
            raise RuntimeError(
                "[novel] 缺少 novel_path。\n"
                "  小说路径只能从 CLI 传入,请运行:\n"
                "    python scripts/generate_data.py --config configs/novel.yaml "
                "--stage novel_only --novel-path /path/to/novel.txt"
            )
        self._index = load_paragraphs(
            self.novel_path,
            min_density=self.min_dialog_density,
            chapter_pattern=self.chapter_pattern,
        )
        # 起点预过滤: 章内后续 dialog 段 ≥ turns_range[1] × 1.5 才算合法起点。
        # × 1.5 给 drop(bad_user/no_lead) 留余量,大幅降低 short_episode。
        self._valid_starts = _compute_valid_starts(
            self._index,
            min_remaining=int(self.turns_range[1] * 1.5),
        )
        print(
            f"  [novel] loaded {self.novel_path}: "
            f"{self._index.n_paragraphs} paragraphs (after chapter strip), "
            f"{len(self._index.dialog_idx)} dialog ({self._index.dialog_density:.1%}), "
            f"{len(self._index.action_idx)} action, "
            f"{self._index.n_chapters} chapters, "
            f"{len(self._valid_starts)} valid starts (after end-of-chapter pruning)"
        )
        if not self._valid_starts:
            raise RuntimeError(
                f"[novel] 章内 dialog 段不足 {self.turns_range[1]} 个 → 无合法起点。"
                f"考虑降低 turns_range 上限,或换章节更长的小说。"
            )

    def _resolve_n_samples(self, requested: int) -> int:
        """n_samples=-1 → 自适应 = dialog_segs × coverage;否则原样。"""
        if requested == -1:
            self._ensure_index()
            adaptive = len(self._index.dialog_idx) * self.coverage
            print(f"  [novel] num_samples=-1 自适应 → {adaptive} "
                  f"(dialog_segs={len(self._index.dialog_idx)} × coverage={self.coverage})")
            return adaptive
        return requested

    def _is_cached(self, req: GenerateRequest) -> bool:
        # n_samples=-1 时基类 _is_cached 直接返回 False(避免误触发文件读)
        # 改为先 resolve 自适应数量后再判;但 -1 路径需要先 load 小说,
        # 若仅做缓存命中检查时不希望强制 load,这里只在缓存文件存在时才 resolve。
        if req.n_samples == -1:
            if not req.out_path.exists():
                return False
            # 文件已存在 → 必须 resolve 真实数量才能比对
            from dataclasses import replace
            req2 = replace(req, n_samples=self._resolve_n_samples(-1))
            return super()._is_cached(req2)
        return super()._is_cached(req)

    def _generate_impl(self, req: GenerateRequest) -> tuple[int, int]:
        self._ensure_index()
        n_samples = self._resolve_n_samples(req.n_samples)
        return generate_novel_dataset(
            req.out_path,
            novel_index=self._index,
            valid_starts=self._valid_starts,
            n_samples=n_samples,
            seed=req.seed,
            turns_range=self.turns_range,
            leading_paras=self.leading_paras,
            turn_max_chars=self.turn_max_chars,
            rejected_path=req.rejected_path,
            progress_every=max(1000, n_samples // 10),
        )
