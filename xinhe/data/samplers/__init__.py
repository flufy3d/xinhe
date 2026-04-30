"""LLM sampler registry。

按 model 名 dispatch,5 家后端接口已对齐:
  call_with_retry(system_prompt, user_prompt, *, model, ...) -> {"choices":[{"message":{"content":...}}]}

  codex-cli[:backing]  → codex_cli   (subprocess, ChatGPT Plus 配额)
  gemini-cli[:backing] → gemini_cli  (subprocess, Google 账号配额)
  claude-cli[:backing] → claude_cli  (subprocess, Claude Code 订阅配额)
  含 "/"               → openrouter  (HTTP, minimax/google/qwen 等)
  其他                  → deepseek    (HTTP, deepseek-v4-flash 等)
"""
from __future__ import annotations

import importlib
from typing import Callable

# (matcher, module_name, error_class_name)
_RULES: list[tuple[Callable[[str], bool], str, str]] = [
    (lambda m: m.split(":", 1)[0].lower() in ("codex-cli", "codex"),   "codex_cli",  "CodexCliError"),
    (lambda m: m.split(":", 1)[0].lower() in ("gemini-cli", "gemini"), "gemini_cli", "GeminiCliError"),
    (lambda m: m.split(":", 1)[0].lower() in ("claude-cli", "claude"), "claude_cli", "ClaudeCliError"),
    (lambda m: "/" in m,                                                "openrouter", "OpenRouterError"),
]


def resolve(model: str) -> tuple[Callable, type[Exception]]:
    """返回 (call_with_retry, ApiError);未命中规则 fallback 到 deepseek。"""
    for matcher, mod_name, err_name in _RULES:
        if matcher(model):
            mod = importlib.import_module(f"xinhe.data.samplers.{mod_name}")
            return mod.call_with_retry, getattr(mod, err_name)
    from xinhe.data.samplers import deepseek
    return deepseek.call_with_retry, deepseek.DeepSeekError
