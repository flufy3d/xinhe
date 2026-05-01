"""Novel-recall 数据集生成主体。

设计要点:
  - user 始终是 dialog 段引号内纯台词(信息孤立,只能从 W 拉前后文)
  - assistant 跨度 [prev_boundary, next_anchor),含当前 dialog 段完整原文
  - 起点随机化 + episode 长度 [8, 12] 随机 → 破 LoRA 死记
  - 3 级预案:dialog → action → sentence-cut,每次降级记 stats
"""
from __future__ import annotations

import json
import random
import re
import uuid
from bisect import bisect_left
from pathlib import Path
from typing import Optional

# user 至少含 1 个中日韩字符或字母数字(过滤纯标点台词,如 "……" / "?!")
_HAS_TEXT_CHAR = re.compile(r"[一-鿿㐀-䶿\w]")

from xinhe.data.generators.dialog.driver_core import _single_instance_lock
from xinhe.data.generators.novel.novel_loader import NovelIndex
from xinhe.data.generators.novel.text_utils import (
    extract_quoted,
    first_sentence_slice,
    truncate_at_sentence,
)


# user 台词字数下限: 仅过滤空台词。短台词(如 "什么？" "女巫")是自然用户输入。
_MIN_USER_CHARS = 1
# 单 turn assistant 中"非 anchor 文本"(前后叙事合计) 字数下限
# 防止 assistant 几乎只复读 anchor 一段台词 → 模型躺平 LM 直接抄
_MIN_CONTEXT_CHARS = 30
# 单 turn assistant 中"前置叙事"字数下限
# 防止 anchor 紧贴 assistant 开头 → 等于把"答案"放在续写起点 = 漏题
_MIN_LEAD_CHARS = 20
# anchor 段长度上限 (相对 turn_max_chars 的比例)
# 太长就没空间给前置叙事,必然漏题 → drop
_MAX_ANCHOR_RATIO = 0.7
from xinhe.data.schema import AssistantTurn, Sample, UserTurn, validate_sample


# 3 级 fallback 来源标记
_SRC_DIALOG = "dialog"
_SRC_ACTION = "action"
_SRC_SENTENCE = "sentence"


def _next_anchor_after(
    cursor: int,
    idx: NovelIndex,
    *,
    prev_boundary: int,
    target_chapter: int,
    min_lead: int = 0,
    stats: dict,
) -> tuple[int | None, str]:
    """找 cursor 之后、与 target_chapter 同章节的下一合法 anchor 段。

    遇到跨章节的候选立刻停止(返回 None),保证 episode 不跨章。
    min_lead > 0 时,要求 paragraphs[prev_boundary:cand] 累计字数 ≥ min_lead。

    优先级: dialog → action → sentence。
    返回 (anchor_idx, source) 或 (None, "none") 表示同章节内已无合法 anchor。
    """
    n = idx.n_paragraphs

    def _lead_ok(cand: int) -> bool:
        if min_lead <= 0:
            return True
        return sum(len(p) for p in idx.paragraphs[prev_boundary:cand]) >= min_lead

    # L0: dialog —— 严格要求同章节
    pos = bisect_left(idx.dialog_idx, cursor + 1)
    while pos < len(idx.dialog_idx):
        cand = idx.dialog_idx[pos]
        if idx.chapter_id_of[cand] != target_chapter:
            return None, "none"  # 已进入下一章,不再继续找
        if _lead_ok(cand):
            return cand, _SRC_DIALOG
        pos += 1
    # L1: action(同上,跨章则停)
    pos = bisect_left(idx.action_idx, cursor + 1)
    while pos < len(idx.action_idx):
        cand = idx.action_idx[pos]
        if idx.chapter_id_of[cand] != target_chapter:
            return None, "none"
        if _lead_ok(cand):
            stats["L1_action"] = stats.get("L1_action", 0) + 1
            return cand, _SRC_ACTION
        pos += 1
    # L2: 下一段(必须同章)
    nxt = cursor + 1
    if nxt < n and idx.chapter_id_of[nxt] == target_chapter:
        stats["L2_sentence"] = stats.get("L2_sentence", 0) + 1
        return nxt, _SRC_SENTENCE
    return None, "none"


def _build_user(paragraph: str, source: str) -> str:
    """按 source 类型抽取 user 字符串。"""
    if source == _SRC_DIALOG:
        u = extract_quoted(paragraph)
        if u:
            return u
        # 引号匹配失败(罕见,如未闭合) → 退化首句
        return first_sentence_slice(paragraph)
    if source == _SRC_ACTION:
        return paragraph.strip()  # action 段本身就短
    # _SRC_SENTENCE
    return first_sentence_slice(paragraph)


def _build_assistant_around_anchor(
    *,
    anchor: str,
    before: list[str],
    after: list[str],
    max_chars: int,
) -> str:
    """以 anchor 段为中心拼 assistant,前后叙事按剩余 budget 补。

    设计:user 台词原文(在 anchor 中)必须保留;前后叙事是"跨 turn anchor 信息"
    的主要载体,在长度允许时尽量塞进来。
    """
    # anchor 自己就超 budget:截到句末即可
    if len(anchor) >= max_chars:
        return truncate_at_sentence(anchor, max_chars)

    remaining = max_chars - len(anchor)
    budget_before = remaining // 2
    budget_after = remaining - budget_before

    # 尾段:从前往后塞,直到 budget 用完(不切单段,塞不下就停)
    after_text = ""
    for p in after:
        if len(after_text) + len(p) + 1 > budget_after:
            slot = budget_after - len(after_text) - 1
            if slot >= 15:    # 残 budget 至少 15 字才值得切句末塞,否则不塞
                after_text += "\n" + truncate_at_sentence(p, slot)
            break
        after_text += ("\n" if after_text else "\n") + p

    # 前段:从后往前塞 (靠 anchor 的优先,远端裁掉)
    before_text = ""
    for p in reversed(before):
        if len(before_text) + len(p) + 1 > budget_before:
            slot = budget_before - len(before_text) - 1
            if slot >= 15:    # 同上,降低门槛让短 budget 也能塞前置 tail
                # 远段保留尾部 slot 字 (靠近 anchor 那边的内容更相关)
                tail = truncate_at_sentence(p[-slot:], slot)
                before_text = tail + "\n" + before_text
            break
        before_text = p + ("\n" + before_text if before_text else "")

    parts = [s for s in (before_text, anchor, after_text.lstrip("\n")) if s]
    return "\n".join(parts)


def _build_episode(
    rng: random.Random,
    idx: NovelIndex,
    valid_starts: list[int],
    *,
    n_turns: int,
    leading_paras: int,
    turn_max_chars: int,
    stats: dict,
) -> list[tuple[str, str]]:
    """拼装一条 episode 的 (user, assistant) 列表。

    起点从 valid_starts(章内后续 dialog 数充足的 dialog 段)随机选,
    确保 episode 不易凑不够 n_turns。
    """
    if not valid_starts:
        return []
    start = rng.choice(valid_starts)
    target_chapter = idx.chapter_id_of[start]    # episode 锁在此章,跨章则终止
    cursor = start
    # prev_boundary 不能跨章节(回退到首个属同章节的段)
    prev_boundary = max(0, start - leading_paras)
    while prev_boundary < cursor and idx.chapter_id_of[prev_boundary] != target_chapter:
        prev_boundary += 1
    cur_source = _SRC_DIALOG          # 起点保证是 dialog 段
    turns: list[tuple[str, str]] = []
    paragraphs = idx.paragraphs

    # 缓存当前章节末段 index +1(用于文末场景的 end_excl 截断)
    chapter_end = cursor + 1
    while (chapter_end < idx.n_paragraphs
           and idx.chapter_id_of[chapter_end] == target_chapter):
        chapter_end += 1

    # 每个 turn 三种结局: 收下、跳过(不算 turn 但 cursor 推进)、结束
    # 跳过的 turn 累计到 stats 的 bad_user / echo_only / no_lead
    advance_budget = n_turns * 3   # 给 advance 最多 3× 重试预算,防止遥遥无期
    while len(turns) < n_turns and advance_budget > 0:
        advance_budget -= 1

        # next_anchor 必须保证下一 turn 的 prev_boundary→cand 之间有足够叙事池,
        # 且不跨章。跨章则 next_anchor=None → episode 终止
        nb_prev = cursor + 1
        next_anchor, next_source = _next_anchor_after(
            cursor, idx, prev_boundary=nb_prev,
            target_chapter=target_chapter,
            min_lead=_MIN_LEAD_CHARS, stats=stats,
        )
        # next_anchor=None 时 end_excl 截到当前章末段,不跨章
        end_excl = next_anchor if next_anchor is not None else chapter_end

        # ——— filter 1: user 含文字字符 (extract_quoted 已 trim 连接性标点) ———
        # 短台词 "什么？" "殿下？" "女巫" 都是自然用户输入,允许;只拦纯标点台词
        user = _build_user(paragraphs[cursor], cur_source)
        if (
            not user
            or len(user) < _MIN_USER_CHARS
            or not _HAS_TEXT_CHAR.search(user)
        ):
            stats["bad_user"] = stats.get("bad_user", 0) + 1
            if next_anchor is None:
                break
            prev_boundary = cursor + 1
            cursor = next_anchor
            cur_source = next_source
            continue

        anchor = paragraphs[cursor]
        before = paragraphs[prev_boundary:cursor]
        after = paragraphs[cursor + 1:end_excl]
        lead_chars = sum(len(p) for p in before)
        trail_chars = sum(len(p) for p in after)

        # ——— filter 2a: anchor 段太长,挤掉前置叙事 budget → 漏题 ———
        if len(anchor) > _MAX_ANCHOR_RATIO * turn_max_chars:
            stats["no_lead"] = stats.get("no_lead", 0) + 1
            if next_anchor is None:
                break
            prev_boundary = cursor + 1
            cursor = next_anchor
            cur_source = next_source
            continue

        # ——— filter 2b: 前置叙事字数池 < MIN_LEAD_CHARS → 漏题 ———
        # anchor 在 assistant 开头被复述,LM 直接抄 = 不需 W
        if lead_chars < _MIN_LEAD_CHARS:
            stats["no_lead"] = stats.get("no_lead", 0) + 1
            if next_anchor is None:
                break
            prev_boundary = cursor + 1
            cursor = next_anchor
            cur_source = next_source
            continue

        # ——— filter 3: 前后叙事池总长 < MIN_CONTEXT_CHARS → 复读机 ———
        if lead_chars + trail_chars < _MIN_CONTEXT_CHARS:
            stats["echo_only"] = stats.get("echo_only", 0) + 1
            if next_anchor is None:
                break
            prev_boundary = cursor + 1
            cursor = next_anchor
            cur_source = next_source
            continue

        # 通过所有 filter,拼 assistant
        assistant = _build_assistant_around_anchor(
            anchor=anchor, before=before, after=after, max_chars=turn_max_chars,
        )
        # 拼出来后再卡一次:实际 assistant 中的非 anchor 文本必须够
        if not assistant or len(assistant) - len(anchor) < _MIN_LEAD_CHARS:
            stats["echo_only"] = stats.get("echo_only", 0) + 1
            if next_anchor is None:
                break
            prev_boundary = cursor + 1
            cursor = next_anchor
            cur_source = next_source
            continue

        turns.append((user, assistant))

        if next_anchor is None:
            break                       # 文末,提前结束
        prev_boundary = cursor + 1
        cursor = next_anchor
        cur_source = next_source

    return turns


def _make_sample(turns: list[tuple[str, str]]) -> Sample:
    """组装 schema 兼容的 Sample (LM-only loss,无 value span)。

    meta 仅留 n_turns;不持久化路径/seed,避免 jsonl 里嵌私人信息。
    """
    conversations: list[dict] = []
    for user, assistant in turns:
        conversations.append(UserTurn(content=user).to_dict())
        conversations.append(
            AssistantTurn(
                role="assistant",
                content=assistant,
                train_loss="true",
                value=None,
                value_span=[],
                value_tier=None,
                weight_per_span=0.0,
            ).to_dict()
        )
    return Sample(
        sample_id=str(uuid.uuid4()),
        stage="novel",
        skeleton_id=None,
        meta={"n_turns": len(turns)},
        conversations=conversations,
    )


def generate_novel_dataset(
    out_path: str | Path,
    *,
    novel_index: NovelIndex,
    valid_starts: list[int],
    n_samples: int,
    seed: int,
    turns_range: tuple[int, int],
    leading_paras: int,
    turn_max_chars: int,
    rejected_path: str | Path | None = None,
    progress_every: int = 5000,
) -> tuple[int, int]:
    """流式生成 n_samples 条样本,返回 (kept, rejected)。

    schema 校验在写盘前跑;失败的丢 rejected_path。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej = Path(rejected_path) if rejected_path else None
    if rej is not None:
        rej.parent.mkdir(parents=True, exist_ok=True)

    stats: dict = {
        "L1_action": 0,
        "L2_sentence": 0,
        "short_episode": 0,
        "bad_user": 0,        # 台词非完整收尾 / 太短 → drop turn
        "no_lead": 0,         # 前置叙事不足 → 漏题 → drop turn
        "echo_only": 0,       # 前后叙事池不足 → 复读机 → drop turn
        "n_turns_total": 0,
        "n_episodes": 0,
    }

    n_ok = 0
    n_rej = 0

    with _single_instance_lock(out):
        rej_fp = open(rej, "w", encoding="utf-8") if rej else None
        try:
            with open(out, "w", encoding="utf-8") as fp:
                for i in range(n_samples):
                    rng = random.Random(seed + i)
                    n_turns = rng.randint(*turns_range)
                    turns = _build_episode(
                        rng,
                        novel_index,
                        valid_starts,
                        n_turns=n_turns,
                        leading_paras=leading_paras,
                        turn_max_chars=turn_max_chars,
                        stats=stats,
                    )
                    # 凑不到下限 turns_range[0] → 视为 short,直接丢
                    # (不再因"没凑够随机抽到的 n_turns"而记 short — 那只是抽样目标,
                    #  实际凑到 ≥ turns_range[0] 已经算合格 episode)
                    if len(turns) < turns_range[0]:
                        stats["short_episode"] += 1
                        continue

                    sample = _make_sample(turns)
                    d = sample.to_dict()
                    try:
                        validate_sample(d)
                    except Exception as e:
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
                    stats["n_episodes"] += 1
                    stats["n_turns_total"] += len(turns)

                    if progress_every and (i + 1) % progress_every == 0:
                        print(f"  [novel] {i + 1}/{n_samples}  kept={n_ok} rej={n_rej}")
        finally:
            if rej_fp is not None:
                rej_fp.close()

    _print_stats(stats, n_ok, n_rej)
    return n_ok, n_rej


def _print_stats(stats: dict, n_ok: int, n_rej: int) -> None:
    """生成完打印来源构成 + 异常计数。"""
    n_turns = stats["n_turns_total"]
    n_eps = max(1, stats["n_episodes"])
    # turns 总尝试数(收下 + 各种 drop)
    n_attempts = (
        n_turns + stats["bad_user"] + stats["no_lead"] + stats["echo_only"]
    )
    n_attempts_safe = max(1, n_attempts)
    print(f"  [novel] generated {n_ok} samples ({n_rej} rejected)")
    if n_attempts > 0:
        print(
            f"    L1 fallback (action segment):  {stats['L1_action']:>6}  "
            f"({stats['L1_action'] / n_attempts_safe:.1%})"
        )
        print(
            f"    L2 fallback (sentence cut):    {stats['L2_sentence']:>6}  "
            f"({stats['L2_sentence'] / n_attempts_safe:.1%})"
        )
        print(
            f"    drop bad_user (台词不完整):    {stats['bad_user']:>6}  "
            f"({stats['bad_user'] / n_attempts_safe:.1%})"
        )
        print(
            f"    drop no_lead (漏题):           {stats['no_lead']:>6}  "
            f"({stats['no_lead'] / n_attempts_safe:.1%})"
        )
        print(
            f"    drop echo_only (复读机):       {stats['echo_only']:>6}  "
            f"({stats['echo_only'] / n_attempts_safe:.1%})"
        )
    print(
        f"    short_episode (n_turns 不达预期): {stats['short_episode']:>6}  "
        f"({stats['short_episode'] / max(1, stats['n_episodes'] + stats['short_episode']):.1%})"
    )
    print(f"    avg n_turns: {n_turns / n_eps:.1f}")
