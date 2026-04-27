"""Codex CLI sampler — 与 deepseek_sampler 同形接口的 subprocess 后端。

调 codex CLI 的 stdin + --output-schema 模式,走用户的 ChatGPT Plus 配额。
Windows 上必须 --dangerously-bypass-approvals-and-sandbox(Win sandbox bug)。
5h rolling quota,1-2 worker 比较稳。

driver.py 看到 model="codex-cli" 就 dispatch 过来。也支持 "codex-cli:gpt-5" 形式选 backing model。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class CodexCliError(Exception):
    """codex CLI 调用错误(找不到 binary、超时、rc!=0、quota 耗尽等)。"""
    pass


_OUTPUT_SCHEMA_JSON = """{
  "type": "object",
  "additionalProperties": false,
  "required": ["conversations"],
  "properties": {
    "conversations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["role", "content", "train_loss", "value", "beat"],
        "properties": {
          "role": {"type": "string", "enum": ["user", "assistant"]},
          "content": {"type": "string"},
          "train_loss": {"type": "string", "enum": ["true", "lm_only", "false"]},
          "value": {
            "anyOf": [
              {"type": "array", "items": {"type": "string"}},
              {"type": "null"}
            ]
          },
          "beat": {"type": "integer", "minimum": 1, "maximum": 5}
        }
      }
    }
  }
}
"""


_CODEX_BIN: Optional[str] = None


def _resolve_codex() -> str:
    global _CODEX_BIN
    if _CODEX_BIN:
        return _CODEX_BIN
    for cand in ("codex.cmd", "codex.exe", "codex"):
        p = shutil.which(cand)
        if p:
            _CODEX_BIN = p
            return p
    raise CodexCliError("找不到 codex 可执行文件 (codex.cmd / codex.exe / codex)")


def _parse_model(model: str) -> Optional[str]:
    """'codex-cli:gpt-5' → 'gpt-5'; 'codex-cli' / 'codex' → None(用 codex 默认)。"""
    if ":" in model:
        return model.split(":", 1)[1] or None
    return None


def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "codex-cli",
    temperature: float = 0.85,   # ignored, CLI 不接受
    top_p: float = 0.92,          # ignored
    max_tokens: int = 12000,      # ignored
    json_mode: bool = True,       # 永远走 --output-schema
    timeout: int = 300,
    reasoning_effort: str = "medium",
    **_ignore,
) -> dict:
    """单次调用,失败抛 CodexCliError;重试由 driver.py 的 a1/a2/a3 处理。

    返回 {"choices":[{"message":{"content": <json_str>}}]} 跟 deepseek_sampler 同形。
    content 是 codex 写到 out.json 的对话 JSON 串。
    """
    work_dir = Path(tempfile.mkdtemp(prefix="codex_smp_"))
    try:
        out_file = work_dir / "out.json"
        schema_file = work_dir / "schema.json"
        schema_file.write_text(_OUTPUT_SCHEMA_JSON, encoding="utf-8")

        prompt_content = (
            "# 系统规范\n\n" + system_prompt + "\n\n"
            "# 任务参数\n\n" + user_prompt + "\n"
        )

        cmd = [
            _resolve_codex(), "exec",
            "--ephemeral", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(work_dir),
            "-c", "features.image_generation=false",
            "-c", 'web_search="disabled"',
            "--output-schema", str(schema_file),
            "--output-last-message", str(out_file),
        ]
        backing = _parse_model(model)
        if backing:
            cmd.extend(["-m", backing])
        if reasoning_effort:
            cmd.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
        cmd.append("-")  # stdin sentinel

        try:
            r = subprocess.run(
                cmd,
                input=prompt_content,
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as e:
            raise CodexCliError(f"codex timeout ({timeout}s)") from e

        if not out_file.exists():
            raise CodexCliError(
                f"codex 没写出文件; rc={r.returncode}; "
                f"stdout tail: {r.stdout[-200:]!r}; stderr tail: {r.stderr[-200:]!r}"
            )
        raw = out_file.read_text(encoding="utf-8")
        if not raw.strip():
            raise CodexCliError(f"codex out.json 为空; rc={r.returncode}")
        return {"choices": [{"message": {"content": raw}}]}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
