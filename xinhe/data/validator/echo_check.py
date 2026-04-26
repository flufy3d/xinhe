"""Echo defense：assistant content 不得复读上一轮 user content。

已知的 ling 退化模式（2026-04-26 实测）：在长 5-Beat prompt 下，ling 频繁把
assistant content 直接以 user content 整段为前缀/后缀/全等，并在后面附点小尾巴。
tier 校验只检查 canonical 是否在 content 里出现（hard match），echo 复读必然会让
canonical 顺带"出现"，所以原校验通不过这一关。

判定：user content len ≥ 8 中文字符或 12 字符总长，且 assistant content 满足下列任一：
  - exact-match: assistant == user（去首尾标点空白后）
  - prefix-echo: assistant 以 user 作前缀，且尾部新增量 < user 长度的 50%
  - suffix-echo: assistant 以 user 作后缀，且头部新增量 < user 长度的 50%

容忍：
  - 短 user（< 8 中文字 / 12 总字符），如"是的"、"嗯"等不在防御范围
  - assistant 长度 ≥ user 的 2 倍 → 视作 acknowledge + extend，不算 echo
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EchoResult:
    ok: bool
    n_echo: int
    echo_turns: list[tuple[int, str]]   # [(turn_index, kind), ...] kind ∈ {exact, prefix, suffix}

    @property
    def reason(self) -> str:
        if self.ok:
            return "ok"
        kinds = ", ".join(f"turn[{i}]={k}" for i, k in self.echo_turns)
        return f"{self.n_echo} 个 assistant turn 复读 user: {kinds}"


_PUNCT = "，。！？；：、,.!?;:'\"“”‘’ \t\n"


def _strip_edges(s: str) -> str:
    return s.strip(_PUNCT)


def _zh_len(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")


def _classify(user_content: str, asst_content: str) -> str | None:
    u = _strip_edges(user_content)
    a = _strip_edges(asst_content)
    # 长度门槛：太短不防御
    if _zh_len(u) < 8 and len(u) < 12:
        return None
    if not u or not a:
        return None
    # acknowledge + extend：assistant 长度 ≥ user 两倍且不直接以 user 起头/结尾，不算 echo
    if a == u:
        return "exact"
    if a.startswith(u):
        tail_extra = len(a) - len(u)
        # 尾部新增量足够（≥ user 长度 60%）才算正常 extend；否则是 echo
        if tail_extra < int(len(u) * 0.6):
            return "prefix"
        return None
    if a.endswith(u):
        head_extra = len(a) - len(u)
        if head_extra < int(len(u) * 0.6):
            return "suffix"
        return None
    return None


def check_echo(conversations: list[dict]) -> EchoResult:
    echo_turns: list[tuple[int, str]] = []
    for i, turn in enumerate(conversations):
        if i == 0:
            continue
        if turn.get("role") != "assistant":
            continue
        prev = conversations[i - 1]
        if prev.get("role") != "user":
            continue
        kind = _classify(prev.get("content", ""), turn.get("content", ""))
        if kind:
            echo_turns.append((i, kind))
    return EchoResult(
        ok=len(echo_turns) == 0,
        n_echo=len(echo_turns),
        echo_turns=echo_turns,
    )
