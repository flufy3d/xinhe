"""小说文本加载 + 段落化 + 启动期诊断 (L4 预案)。

启动一次即可,后续生成 N 万条 sample 全靠内存里这份段落表。

章节段(匹配 chapter_pattern)从 paragraphs 列表中删除——它们是 meta 信息,
不应进入训练数据。但章节归属通过 chapter_id_of[i] 保留,episode 不可跨章。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from xinhe.data.generators.novel.text_utils import (
    has_dialog_quote,
    is_action_paragraph,
    normalize_quotes,
)


# 默认章节标题 regex(覆盖中文数字 + 阿拉伯数字 + 章/回/节 三种常见称谓)。
# 用户可在 yaml 里配 `chapter_pattern: "..."` 覆盖,适配不同小说。
DEFAULT_CHAPTER_PATTERN = r"^第[一二三四五六七八九十百千零〇\d]+[章回节]"


@dataclass
class NovelIndex:
    """段落表 + 索引(dialog / action / 章节归属)。

    paragraphs 已剔除章节段(它们是 meta,不进训练)。chapter_id_of[i] 给
    paragraphs[i] 标章节号(0-based;开篇到第一章前的段为 -1)。
    """
    paragraphs: list[str]
    dialog_idx: list[int]            # 含对话引号的段
    action_idx: list[int]            # 含动作动词且短的段(L1 fallback 池)
    chapter_id_of: list[int]         # 每段所属章节号(per-paragraph)
    n_chapters: int                  # 章节总数

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def dialog_density(self) -> float:
        if not self.paragraphs:
            return 0.0
        return len(self.dialog_idx) / len(self.paragraphs)


def _read_text(path: Path) -> str:
    """utf-8 → fallback gbk → 抛错。"""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码 {path}: 试过 utf-8/utf-8-sig/gbk/gb18030 都失败")


def load_paragraphs(
    novel_path: str | Path,
    *,
    min_density: float = 0.05,
    chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
) -> NovelIndex:
    """读小说,切段,标注 dialog/action 索引 + 章节归属,跑 L4 密度校验。

    Args:
        min_density: dialog 段占总段比的下限,低于则抛错(L4 预案)。
        chapter_pattern: 章节标题 regex。匹配到的段从 paragraphs 中删除。

    Returns:
        NovelIndex(paragraphs 不含章节段, chapter_id_of 给每段标章节号)
    """
    path = Path(novel_path)
    if not path.exists():
        raise FileNotFoundError(f"小说文件不存在: {path}")

    text = _read_text(path)
    text = normalize_quotes(text)

    chapter_pat = re.compile(chapter_pattern)

    # 按单个 \n 切段:中文小说典型结构(段首 　　 缩进)。
    # 双 \n 不可靠 —— 大量小说仅用单 \n 划段,双 \n 仅出现在章节/卷/简介过渡。
    raw_paras = text.split("\n")
    paragraphs: list[str] = []
    chapter_id_of: list[int] = []
    cur_chapter = -1   # 第一章前的段(序章 / 简介 / 卷标题)归 -1

    for p in raw_paras:
        # 去掉首尾空白 + 中文段首缩进 　(全角空格)
        p = p.strip().lstrip("　").strip()
        if len(p) < 4:
            continue
        if chapter_pat.match(p):
            # 章节段不入 paragraphs,只更新章节计数
            cur_chapter += 1
            continue
        paragraphs.append(p)
        chapter_id_of.append(cur_chapter)

    if not paragraphs:
        raise RuntimeError(f"小说切段后为空: {path}")

    n_chapters = max(0, cur_chapter + 1)

    dialog_idx = [i for i, p in enumerate(paragraphs) if has_dialog_quote(p)]
    action_idx = [i for i, p in enumerate(paragraphs) if is_action_paragraph(p)]

    idx = NovelIndex(
        paragraphs=paragraphs,
        dialog_idx=dialog_idx,
        action_idx=action_idx,
        chapter_id_of=chapter_id_of,
        n_chapters=n_chapters,
    )

    # L4: dialog 密度过低 → 抛错让用户换书
    if idx.dialog_density < min_density:
        raise RuntimeError(
            f"小说 dialog 密度过低 ({idx.dialog_density:.1%}, 需 ≥{min_density:.0%}), "
            f"不适合 regex 伪交互模式。建议换一本对话密集的小说。"
        )

    return idx


def diagnose(
    novel_path: str | Path,
    *,
    coverage: int = 3,
    chapter_pattern: str = DEFAULT_CHAPTER_PATTERN,
) -> dict:
    """诊断模式:只读小说不生成 sample,打印段落统计 + 推荐 num_train。"""
    idx = load_paragraphs(novel_path, chapter_pattern=chapter_pattern)
    rec_train = len(idx.dialog_idx) * coverage
    print(f"\n[novel diagnose] {novel_path}")
    print(f"  chapter_pattern: {chapter_pattern!r}")
    print(f"  总段数(剔除章节后): {idx.n_paragraphs}")
    print(f"  dialog 段数      : {len(idx.dialog_idx)} ({idx.dialog_density:.1%})")
    print(f"  action 段数      : {len(idx.action_idx)}")
    print(f"  章节数           : {idx.n_chapters}")
    print(f"  推荐 num_train   : {rec_train} (= dialog_segs × coverage={coverage})")
    return {
        "n_paragraphs": idx.n_paragraphs,
        "n_dialog": len(idx.dialog_idx),
        "n_action": len(idx.action_idx),
        "n_chapters": idx.n_chapters,
        "dialog_density": idx.dialog_density,
        "recommended_num_train": rec_train,
    }
