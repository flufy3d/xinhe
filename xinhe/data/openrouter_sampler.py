"""OpenRouter sampler — 与 deepseek_sampler 同形接口，便于在 build_dicts / stage1 复用。

OpenRouter 是 OpenAI 兼容网关，支持 minimax / google / qwen 等多家模型。免费模型 model id 形如
`minimax/minimax-m2.5:free`。Key 从 .env 的 OPENROUTER_API_KEY 读取。
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(Exception):
    pass


def _load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    from pathlib import Path
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENROUTER_API_KEY="):
                    val = line.partition("=")[2].strip()
                    if val and val[0] in '"\'' and val[-1] == val[0]:
                        val = val[1:-1]
                    if val:
                        return val
        except Exception:
            pass
    raise OpenRouterError(
        "找不到 OPENROUTER_API_KEY。在 .env 加 OPENROUTER_API_KEY=sk-or-..."
    )


def call_openrouter(
    system_prompt: str = "",
    user_prompt: str = "",
    model: str = "",
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 4000,
    json_mode: bool = True,
    timeout: int = 180,
    reasoning_effort: str | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    messages: Optional[list[dict]] = None,
) -> dict:
    """messages 优先于 system_prompt + user_prompt,支持多轮 conditional retry。
    主流 provider(Anthropic/DeepSeek/OpenAI/Google) 自动 prefix cache,免费 provider 不一定。

    reasoning_effort: "low"/"medium"/"high" 仅对 reasoning model(gpt-oss 等)生效,
    None 表示不传(模型默认全力 reasoning)。低 effort 显著减少 token 消耗。"""
    key = _load_api_key()
    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    # 主流 provider(Minimax/OpenAI/Anthropic/DeepSeek/Groq) 都支持 response_format,
    # 启用后强制 LLM 输出合法 JSON。少数老免费 provider(如某些 OpenInference) 不支持,
    # 这种情况下 caller 显式传 json_mode=False 即可。
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        OPENROUTER_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/flufy3d/xinhe",
            "X-Title": "xinhe-data-pipeline",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
        # 防御：免费 provider 偶发 content=None / choices=空，直接当 retryable 处理。
        choices = obj.get("choices") or []
        if not choices or choices[0].get("message", {}).get("content") is None:
            raise OpenRouterError(
                f"empty content (provider={obj.get('provider')!r}, finish={choices[0].get('finish_reason') if choices else None})"
            )
        return obj
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        raise OpenRouterError(f"HTTP {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise OpenRouterError(f"网络错误: {e}") from e
    except json.JSONDecodeError as e:
        raise OpenRouterError(f"响应不是合法 JSON: {e}") from e


def call_with_retry(
    system_prompt: str = "",
    user_prompt: str = "",
    model: str = "",
    max_retries: int = 4,
    initial_backoff: float = 10.0,
    **kwargs,
) -> dict:
    backoff = initial_backoff
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_openrouter(system_prompt, user_prompt, model=model, **kwargs)
        except OpenRouterError as e:
            msg = str(e)
            if "HTTP 401" in msg or "HTTP 400" in msg or "HTTP 403" in msg:
                raise
            last_err = e
            if attempt < max_retries:
                print(f"  [or retry {attempt+1}/{max_retries}] {msg[:120]} → 等 {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff *= 2
            else:
                raise OpenRouterError(f"重试 {max_retries} 次后仍失败: {last_err}") from last_err
