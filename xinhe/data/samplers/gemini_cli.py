"""Gemini CLI sampler — 与 deepseek_sampler 同形接口的 subprocess 后端。

调 gemini CLI 的 -p (headless) + -o json 模式,走用户的 Google 账号配额。

关键: gemini CLI 默认是 coding agent,会去翻 cwd 文件干活。三招把 agent loop 关死,
拿干净的"模型直答":
  1. cwd=tempfile.mkdtemp() → 空目录,没文件可翻
  2. --skip-trust → 跳过 trust folder 检查(headless 必须)
  3. --approval-mode plan → 只读模式,工具用不动

输出格式 -o json,gemini 会包一个 envelope:
  {"session_id": ..., "response": "<模型实际输出>", "stats": {...}}
我们取 response 字段塞进 OpenAI-style 的 {"choices":[{"message":{"content": ...}}]} 返回。

driver.py 看到 model="gemini-cli" 就 dispatch 过来。也支持 "gemini-cli:gemini-2.5-pro" 形式。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class GeminiCliError(Exception):
    """gemini CLI 调用错误。"""
    pass


class GeminiQuotaExhaustedError(GeminiCliError):
    """gemini 配额耗尽(per-day 或 per-account quota)。
    driver 看到要立刻 sys.exit,不要再重试——继续打会触发账号风控。"""
    pass


# stderr/envelope 内出现这些片段 = quota 耗尽,立即 abort
_GEMINI_QUOTA_SIGS = (
    "exhausted your capacity",          # gemini-3.1-pro 主要信号
    "quota will reset",
    "RESOURCE_EXHAUSTED",
    "rate limit",
    "429",
)


_GEMINI_BIN: Optional[str] = None


def _resolve_gemini() -> str:
    global _GEMINI_BIN
    if _GEMINI_BIN:
        return _GEMINI_BIN
    for cand in ("gemini.cmd", "gemini.exe", "gemini"):
        p = shutil.which(cand)
        if p:
            _GEMINI_BIN = p
            return p
    raise GeminiCliError("找不到 gemini 可执行文件 (gemini.cmd / gemini.exe / gemini)")


def _parse_model(model: str) -> Optional[str]:
    """'gemini-cli:gemini-2.5-pro' → 'gemini-2.5-pro'; 'gemini-cli' → None(用默认)。"""
    if ":" in model:
        return model.split(":", 1)[1] or None
    return None


# 偶发 LLM 仍会在 response 里包 ```json ... ``` 围栏,parser 层有容错;
# 这里再做一道剥壳,提高一次过率。
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\n?\s*```\s*$")


def _strip_fence(s: str) -> str:
    s = _FENCE_RE.sub("", s, count=1)
    s = _FENCE_END_RE.sub("", s, count=1)
    return s.strip()


def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "gemini-cli",
    temperature: float = 0.85,   # ignored
    top_p: float = 0.92,          # ignored
    max_tokens: int = 12000,      # ignored
    json_mode: bool = True,       # 靠 prompt 强约束 + envelope 提取
    timeout: int = 300,
    **_ignore,
) -> dict:
    """单次调用,失败抛 GeminiCliError;重试由 driver.py 的 a1/a2/a3 处理。

    返回 {"choices":[{"message":{"content": <json_str>}}]} 跟 deepseek_sampler 同形。
    """
    prompt_content = (
        "# 系统规范\n\n" + system_prompt + "\n\n"
        "# 任务参数\n\n" + user_prompt + "\n\n"
        "# 输出要求\n\n"
        '严格只输出一个 JSON 对象 { "conversations": [...] },'
        "不要任何前后缀说明文字,不要 ```json 代码块包裹,直接以 { 开头、以 } 结尾。\n"
    )

    # 注意: 走 stdin pipe 而不是 -p,因为长 prompt 经 Windows CreateProcess 命令行
    # 会被截断(实测 Chinese UTF-8 多字节,5-Beat system prompt 5000+ 字符就跑题),
    # stdin pipe 没有这个限制,且非 TTY 自动触发 headless mode。
    cmd = [
        _resolve_gemini(),
        "--skip-trust",
        "--approval-mode", "plan",
        "-o", "json",
    ]
    backing = _parse_model(model)
    if backing:
        cmd.extend(["-m", backing])

    work_dir = Path(tempfile.mkdtemp(prefix="gemini_smp_"))
    try:
        try:
            r = subprocess.run(
                cmd,
                input=prompt_content,
                cwd=str(work_dir),
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as e:
            raise GeminiCliError(f"gemini timeout ({timeout}s)") from e

        if r.returncode != 0:
            blob = (r.stderr or "") + (r.stdout or "")
            if any(sig in blob for sig in _GEMINI_QUOTA_SIGS):
                raise GeminiQuotaExhaustedError(
                    f"gemini quota exhausted; stderr tail: {(r.stderr or '')[-200:]!r}"
                )
            raise GeminiCliError(
                f"gemini rc={r.returncode}; "
                f"stdout: {r.stdout[-200:]!r}; stderr: {r.stderr[-200:]!r}"
            )

        try:
            envelope = json.loads(r.stdout or "{}")
        except json.JSONDecodeError as e:
            # 容错: rc=0 但 stdout 非 JSON,可能是 quota warning 文本
            if any(sig in (r.stdout or "") + (r.stderr or "") for sig in _GEMINI_QUOTA_SIGS):
                raise GeminiQuotaExhaustedError(
                    f"gemini quota exhausted in stdout; head: {r.stdout[:200]!r}"
                )
            raise GeminiCliError(
                f"gemini -o json 包络解析失败: {e}; stdout head: {r.stdout[:200]!r}"
            ) from e

        response = envelope.get("response")
        if not response:
            raise GeminiCliError(
                f"gemini envelope 没 response 字段; envelope keys: {list(envelope.keys())}; "
                f"error: {envelope.get('error')}"
            )

        raw = _strip_fence(response)
        if not raw:
            raise GeminiCliError("gemini response 剥壳后为空")
        return {"choices": [{"message": {"content": raw}}]}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
