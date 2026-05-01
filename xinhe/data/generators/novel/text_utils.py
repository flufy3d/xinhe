"""Novel-recall 文本工具:引号提取、句末截断、action 段识别。

关键约束:user 一定是"信息孤立"的——不带 attribution、不带前后叙事。
让模型只能从 W 拉前 N turn 的人物 / 世界观 anchor。
"""
from __future__ import annotations

import re

# 弯引号 → 直引号 normalize 表 (loader 一进来就跑一次)
_QUOTE_NORMALIZE = str.maketrans({
    "“": '"',  # left double quote “
    "”": '"',  # right double quote ”
    "‘": "'",  # left single quote ‘
    "’": "'",  # right single quote ’
})


def normalize_quotes(text: str) -> str:
    """弯引号统一到直引号,后续 regex 单一形式。"""
    return text.translate(_QUOTE_NORMALIZE)


# 中文 dialog 引号变体 (normalize 后只剩 " ' ; 加上原生 「」 『』)
_DIALOG_QUOTE_OPEN = '"\'「『'
_DIALOG_QUOTE_CLOSE = '"\'」』'

# regex: 段落里凡是含开引号(任意一种)即视为 dialog 段
_HAS_DIALOG = re.compile(r'["‘“「『\']')


def has_dialog_quote(text: str) -> bool:
    """段落是否含对话引号(normalize 前/后都能用,放宽匹配)。"""
    return bool(_HAS_DIALOG.search(text))


# 首尾配对的引号块 (贪婪到最后一个闭引号,跨多句也 OK)
_QUOTE_PAIRS = [
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    re.compile(r'「([^」]+)」'),
    re.compile(r'『([^』]+)』'),
]


_TRAILING_SOFT_PUNCT = "，,；;、 \t　"


def extract_quoted(paragraph: str) -> str:
    """提取段落中第一个闭合引号块的内部纯台词作为 user。

    返回去除引号的纯台词字符串;若无配对引号,返回空串(由调用方触发预案)。

    末尾的连接性标点(逗号/分号/顿号/空白)会被 strip 掉:
    中文小说常见格式 `"台词，"attribution说道。` 把逗号留作 attribution 过渡,
    台词本身意义完整,作 user 时该逗号是干扰。
    """
    candidates = []
    for pat in _QUOTE_PAIRS:
        m = pat.search(paragraph)
        if m:
            candidates.append((m.start(), m.group(1).strip()))
    if not candidates:
        return ""
    # 取段落中最早出现的引号块(避免段尾闲笔的次要台词盖过主台词)
    candidates.sort(key=lambda x: x[0])
    raw = candidates[0][1]
    return raw.rstrip(_TRAILING_SOFT_PUNCT)


# 句末标点 (中英文兼顾)
_SENTENCE_END = "。！？!?…\n"


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """从前往后取到第一个超过 max_chars 的句末作截断,保证不切到半句。

    若整段 < max_chars 直接返回;若开头到 max_chars 内无句末,硬切。
    """
    if len(text) <= max_chars:
        return text
    # 在 [max_chars*0.7, max_chars] 范围内找最末的句末标点,避免切得太短或切碎句子
    lo = int(max_chars * 0.7)
    hi = min(max_chars, len(text) - 1)
    best = -1
    for i in range(hi, lo, -1):
        if text[i] in _SENTENCE_END:
            best = i
            break
    if best > 0:
        return text[: best + 1].rstrip()
    # 退化:硬切 (罕见,长句子无标点)
    return text[:max_chars].rstrip()


# Action 动词词表 (L1 fallback: dialog 段缺失时,退化匹配 action 短段作 user)
# 选取常见的"独立动作"——它们也具备"孤立喂出去模型不知道是谁"的特性
_ACTION_VERBS = (
    "笑", "转身", "回头", "低头", "抬头", "点头", "摇头", "皱眉", "叹气",
    "起身", "坐下", "站起", "走来", "走去", "退后", "靠近", "握紧",
    "凝视", "盯着", "看向", "望着", "瞥", "瞪",
    "拔剑", "举手", "挥手", "摆手", "招手", "拍",
    "喊", "吼", "叫", "咳嗽", "深吸", "屏息", "颤抖",
)
# 用 regex 一次匹配所有动词
_ACTION_PAT = re.compile("|".join(_ACTION_VERBS))


_COMPLETE_UTTERANCE_END = set("?!。！？…")


def is_complete_utterance(s: str) -> bool:
    """user 台词是否以"完整结束"标点收尾。

    可接受: ? ! 。 ！ ？ …(中英问号/叹号/句号/省略号)
    不可接受: 逗号 / 分号 / 顿号 — 看起来"话没说完",作为用户独立输入不自然。
    """
    if not s:
        return False
    return s[-1] in _COMPLETE_UTTERANCE_END


def is_action_paragraph(text: str, max_chars: int = 50) -> bool:
    """L1 预案: 段落短(< max_chars)且含动作动词,认为是 action-like 段。"""
    return len(text) <= max_chars and bool(_ACTION_PAT.search(text))


def first_sentence_slice(text: str, lo: int = 10, hi: int = 20) -> str:
    """L2 预案: 从段落首句切 [lo, hi] 字作 user。"""
    if len(text) <= lo:
        return text.strip()
    # 先看首句
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END and i >= lo:
            return text[: i + 1].strip()
    # 没标点,硬切
    return text[:hi].strip()
