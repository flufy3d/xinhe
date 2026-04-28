"""Counter form 反问纠错检测：

只对 plan.recall_form == "counter" 启用。
逻辑：
  - user Beat 4 故意提及一个 plan 外的实体 X 当陷阱（颜色/食物/城市/职业 等类别词）
  - assistant 必须否认 X，只确认 plan 内的 fact
  - 如果 assistant Beat 4 echo 了 X → 编造 user 没说过的事实，reject

实现：
  - 加载 dicts 的 "事实" 类（foods/colors/brands/hobbies/cities/jobs/pets/project_codes/passwords/organizations）
    成全集 trap_set；不含 surnames/given_names（人名误伤太重）
  - 用 n-gram 切片做 O(N×L) 的 substring 命中（trap_lens = 实际词长集合，通常 ≤8）
  - trap = (user Beat 4 命中的 entity) − canonical − aliases
  - leaked = trap ∩ assistant Beat 4
  - leaked 非空 → reject
"""
from __future__ import annotations

from dataclasses import dataclass, field

from xinhe.data.dicts.bank import load_bank


# 事实类词典：人名（surnames/given_names）除外，避免与第三方人物名/通用人称撞车
_TRAP_CATEGORIES = (
    "foods", "colors", "brands", "hobbies",
    "cities", "jobs", "pets", "project_codes", "passwords", "organizations",
)


_TRAP_BY_LEN: dict[int, set[str]] | None = None
_TRAP_LENS: tuple[int, ...] = ()


def _ensure_trap_index() -> tuple[dict[int, set[str]], tuple[int, ...]]:
    global _TRAP_BY_LEN, _TRAP_LENS
    if _TRAP_BY_LEN is not None:
        return _TRAP_BY_LEN, _TRAP_LENS
    by_len: dict[int, set[str]] = {}
    for cat in _TRAP_CATEGORIES:
        try:
            bank = load_bank(cat, split="all")
        except Exception:
            continue
        for e in bank.entries:
            if not e or len(e) < 2:
                continue
            by_len.setdefault(len(e), set()).add(e)
    _TRAP_BY_LEN = by_len
    _TRAP_LENS = tuple(sorted(by_len.keys()))
    return by_len, _TRAP_LENS


def _entities_in(text: str) -> set[str]:
    if not text:
        return set()
    by_len, lens = _ensure_trap_index()
    found: set[str] = set()
    for n in lens:
        bucket = by_len.get(n)
        if not bucket:
            continue
        for i in range(len(text) - n + 1):
            ng = text[i : i + n]
            if ng in bucket:
                found.add(ng)
    return found


# sentence-level 否定 / 纠错 / 第一次提到 标记：含其一视为否认 trap，asst 不算编造
_NEG_MARKERS = (
    # 直接否定
    "不是", "不算", "不属", "不对", "不在", "不该", "不许", "不要",
    "可不是", "可不", "并不", "并不是", "并非", "并没",
    "没说", "没提", "没有", "没记", "没关系", "没听", "没见",
    "未曾", "未必", "无关", "不相干", "不一定", "不正确",
    "不一样", "不太确定", "不太对", "没必要",
    # 纠错 / 记错
    "记错", "记岔", "记串", "记反", "记混", "搞混", "弄混",
    "弄错", "搞错", "记岔成", "想多了",
    # 强调正确答案 → 隐含否认 trap
    "明明", "明明是",
    # 反问质疑
    "怎么会", "怎么变成", "怎么突然", "怎么一", "哪来的", "哪来",
    "虚构", "瞎想", "幻觉",
    # "第一次/刚才提到 → 否认之前说过"
    "第一次", "刚才提到", "新话题", "倒是第一次", "倒是", "倒是新",
)


def _clauses(text: str) -> list[str]:
    """按标点切 clause，用于判断 trap 是否落在否认子句里。"""
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in "。！？，；,;.!?":
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def _all_positions(text: str, pat: str) -> list[int]:
    out: list[int] = []
    if not pat or not text:
        return out
    start = 0
    while True:
        i = text.find(pat, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def _overlaps_any(pos: int, length: int, intervals: list[tuple[int, int]]) -> bool:
    """trap 出现位置 [pos, pos+length) 与 intervals 中任一区间有重叠 → True。
    用于过滤"canonical 字面 spillover"伪 trap，如 "绿色" 出现在 "橄榄绿色" 中。"""
    end = pos + length
    for a, b in intervals:
        if not (end <= a or b <= pos):
            return True
    return False


def _is_real_echo(
    asst_text: str,
    trap: str,
    canonical_or_alias: set[str],
    extra_safe_words: set[str],
) -> bool:
    """trap 在 asst 中出现 → 检查每个出现位置:
      - 与 canonical/alias 区间有任何重叠（"绿色" 在 "橄榄绿色" 中）→ 跳过
      - 与 extra_safe_words（如第三方 subject 名）区间重叠 → 跳过
      - 落在含否定标记的 sentence 中 → 跳过（asst 在纠错否认）
      - 都不命中 → 真 echo
    """
    occurrences = _all_positions(asst_text, trap)
    if not occurrences:
        return False

    # canonical/alias 在 asst 中的覆盖区间（用于过滤 substring 重叠）
    cov: list[tuple[int, int]] = []
    for c in canonical_or_alias:
        if not c:
            continue
        for p in _all_positions(asst_text, c):
            cov.append((p, p + len(c)))
    # 额外安全词（如第三方 subject 名 "轩辕湛"），asst 复述其 subject 是合理的
    for w in extra_safe_words:
        if not w:
            continue
        for p in _all_positions(asst_text, w):
            cov.append((p, p + len(w)))

    # 按句号 / 问号 / 感叹号切句（比 clause 大；包含 ， 内部的对比"X 是 Y, 不是 Z"）
    sentences_spans: list[tuple[int, int, str]] = []
    cur_start = 0
    for i, ch in enumerate(asst_text):
        if ch in "。！？.!?":
            seg = asst_text[cur_start : i + 1]
            if seg.strip():
                sentences_spans.append((cur_start, i + 1, seg))
            cur_start = i + 1
    if cur_start < len(asst_text):
        seg = asst_text[cur_start:]
        if seg.strip():
            sentences_spans.append((cur_start, len(asst_text), seg))

    for pos in occurrences:
        if _overlaps_any(pos, len(trap), cov):
            continue
        # 找 trap 落在哪个 sentence
        host_sentence = ""
        for a, b, seg in sentences_spans:
            if a <= pos < b:
                host_sentence = seg
                break
        if host_sentence and any(neg in host_sentence for neg in _NEG_MARKERS):
            continue
        return True   # 既未与 canonical 区间重叠、也不在否定句中 → 真 echo
    return False


@dataclass
class CounterAntiEchoResult:
    ok: bool
    reason: str = ""
    trap_entities: list[str] = field(default_factory=list)
    leaked_entities: list[str] = field(default_factory=list)


def check_counter_anti_echo(
    conversations: list[dict],
    beat4_turn_indices: list[int],
    canonical: set[str],
    aliases: set[str],
    subjects: set[str] | None = None,
) -> CounterAntiEchoResult:
    """beat4_turn_indices 是 assistant turn 的 idx；其前一个 idx 是对应 user turn。
    subjects: 第三方 subject 名（如 "轩辕湛"），asst 复述这些是合理的，不算 trap echo。
    """
    if not beat4_turn_indices:
        return CounterAntiEchoResult(ok=True)

    subjects = subjects or set()

    user_text = ""
    asst_text = ""
    for ai in beat4_turn_indices:
        if 0 <= ai < len(conversations) and conversations[ai].get("role") == "assistant":
            asst_text += conversations[ai].get("content", "")
            ui = ai - 1
            if 0 <= ui < len(conversations) and conversations[ui].get("role") == "user":
                user_text += conversations[ui].get("content", "")

    if not (user_text and asst_text):
        return CounterAntiEchoResult(ok=True)

    user_entities = _entities_in(user_text)
    if not user_entities:
        return CounterAntiEchoResult(ok=True)

    raw_trap = user_entities - canonical - aliases

    # Fix 1: 排除与 canonical/alias/subjects 互为子串的 entity
    safe_words = canonical | aliases | subjects
    trap = set()
    for e in raw_trap:
        if any((e in c) or (c in e) for c in safe_words if c):
            continue
        trap.add(e)
    if not trap:
        return CounterAntiEchoResult(ok=True)

    # Fix 2: 否定句豁免 + canonical/subjects 区间覆盖豁免
    canonical_or_alias = canonical | aliases
    leaked = sorted([
        e for e in trap if _is_real_echo(asst_text, e, canonical_or_alias, subjects)
    ])
    if leaked:
        return CounterAntiEchoResult(
            ok=False,
            reason=f"counter form: asst Beat 4 echo 了 user 故意误导的实体 {leaked}（plan 外，不许 echo）",
            trap_entities=sorted(trap),
            leaked_entities=leaked,
        )
    return CounterAntiEchoResult(
        ok=True,
        trap_entities=sorted(trap),
        leaked_entities=[],
    )
