"""Claude Code CLI sampler — 与 deepseek_sampler 同形接口的 subprocess 后端。

调 claude CLI 的 -p (headless) + --output-format json + --json-schema 模式,走用户的
Claude Code 订阅配额(OAuth,非 ANTHROPIC_API_KEY)。

关键:
  - 不能加 --bare,--bare 强制走 ANTHROPIC_API_KEY,绕开订阅配额
  - cwd=tempfile.mkdtemp() → 干净目录,避免触发 CLAUDE.md auto-discovery 污染 system
  - --tools "" → 关掉所有工具,纯文本生成
  - --no-session-persistence → 不落 session 文件
  - --system-prompt 替换默认(不 append),完全控制
  - --json-schema 强制结构化输出,落到 result JSON 的 structured_output 字段

driver.py 看到 model="claude-cli" 就 dispatch 过来。也支持 "claude-cli:opus" 选 backing model
(默认 sonnet,因为用户的 weekly quota 仅限 sonnet)。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class ClaudeCliError(Exception):
    """claude CLI 调用错误。"""
    pass


class ClaudeQuotaExhaustedError(ClaudeCliError):
    """claude 订阅配额耗尽(weekly cap 或 5h rolling)。
    driver 看到要立刻 sys.exit,不要再重试。"""
    pass


# is_error=true 时 result/stderr 内出现这些片段 = quota 耗尽,立即 abort
_CLAUDE_QUOTA_SIGS = (
    "weekly limit",
    "5-hour limit",
    "5 hour limit",
    "rate_limit_error",
    "rate limit",
    "usage limit",
    "session limit",
    "quota",
    "Please try again at",
    "Sonnet limit",
    "Opus limit",
)


_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["conversations"],
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "content", "train_loss", "value", "beat"],
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {"type": "string"},
                    "train_loss": {"type": "string", "enum": ["true", "lm_only", "false"]},
                    "value": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                    "beat": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        }
    },
}


_CLAUDE_BIN: Optional[str] = None


def _resolve_claude() -> str:
    global _CLAUDE_BIN
    if _CLAUDE_BIN:
        return _CLAUDE_BIN
    for cand in ("claude.cmd", "claude.exe", "claude"):
        p = shutil.which(cand)
        if p:
            _CLAUDE_BIN = p
            return p
    raise ClaudeCliError("找不到 claude 可执行文件 (claude.cmd / claude.exe / claude)")


def _parse_model(model: str) -> str:
    """'claude-cli:opus' → 'opus'; 'claude-cli' / 'claude' → 'sonnet'(默认)。"""
    if ":" in model:
        backing = model.split(":", 1)[1].strip()
        return backing or "sonnet"
    return "sonnet"


def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "claude-cli",
    temperature: float = 0.85,   # ignored
    top_p: float = 0.92,          # ignored
    max_tokens: int = 12000,      # ignored
    json_mode: bool = True,       # 永远走 --json-schema
    timeout: int = 900,           # 5-Beat 长输出 + schema 校验 实测中位数 ~400s, 长尾到 520s+, 留 ~70% 余量
    max_budget_usd: float = 5.0,  # 单次调用上限,防失控
    **_ignore,
) -> dict:
    """单次调用,失败抛 ClaudeCliError;重试由 driver.py 的 a1/a2/a3 处理。

    返回 {"choices":[{"message":{"content": <json_str>}}]} 跟 deepseek_sampler 同形。
    """
    work_dir = Path(tempfile.mkdtemp(prefix="claude_smp_"))
    try:
        backing = _parse_model(model)

        cmd = [
            _resolve_claude(),
            "-p", "--print",
            "--no-session-persistence",
            "--model", backing,
            "--output-format", "json",
            "--system-prompt", system_prompt,
            "--tools", "",
            "--max-budget-usd", str(max_budget_usd),
            "--json-schema", json.dumps(_OUTPUT_SCHEMA, ensure_ascii=False),
        ]

        try:
            r = subprocess.run(
                cmd,
                input=user_prompt,
                cwd=str(work_dir),
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCliError(f"claude timeout ({timeout}s)") from e

        out = (r.stdout or "").strip()
        err_tail = (r.stderr or "")[-500:]

        if not out:
            if any(sig in err_tail for sig in _CLAUDE_QUOTA_SIGS):
                raise ClaudeQuotaExhaustedError(
                    f"claude quota exhausted; stderr tail: {err_tail[-200:]!r}"
                )
            raise ClaudeCliError(
                f"claude stdout 为空; rc={r.returncode}; stderr tail: {err_tail[-300:]!r}"
            )

        try:
            envelope = json.loads(out)
        except json.JSONDecodeError as e:
            raise ClaudeCliError(
                f"claude stdout 非 JSON: {e}; head: {out[:300]!r}"
            ) from e

        # is_error=true → 看 result/stderr 是否含 quota 信号
        if envelope.get("is_error"):
            result_text = str(envelope.get("result") or "")
            api_status = str(envelope.get("api_error_status") or "")
            blob = result_text + " " + api_status + " " + err_tail
            if any(sig in blob for sig in _CLAUDE_QUOTA_SIGS):
                raise ClaudeQuotaExhaustedError(
                    f"claude quota exhausted; result: {result_text[:200]!r}; api_status: {api_status[:80]!r}"
                )
            raise ClaudeCliError(
                f"claude is_error=true; result: {result_text[:300]!r}; api_status: {api_status[:80]!r}"
            )

        # 优先 structured_output(schema 命中);fallback result(纯文本)
        structured = envelope.get("structured_output")
        if structured is not None:
            content_str = json.dumps(structured, ensure_ascii=False)
        else:
            content_str = str(envelope.get("result") or "")
            if not content_str.strip():
                raise ClaudeCliError(
                    f"claude result 为空; envelope: {str(envelope)[:300]!r}"
                )

        return {"choices": [{"message": {"content": content_str}}]}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
