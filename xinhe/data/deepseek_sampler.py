"""
DeepSeek V3 sampler — Stage 1 5-Beat 对话生成 + 词典/语料扩充。

关键设计:
- 强制 response_format={"type": "json_object"} 降低解析失败率
- DeepSeek API 是 OpenAI 兼容的，使用 urllib 调用（零新依赖）
- Key 只从 os.environ["DEEPSEEK_API_KEY"] 读取，不接受参数传入
- 断点续传、rate limit 重试由调用方（build_dicts.py / stage1.driver）负责
- 质量过滤在采样完后本地做（schema validator + n-gram 重复 + tier 分类）

DeepSeek API 参考: https://api-docs.deepseek.com/
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional


DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"  # DeepSeek V4 Flash


class DeepSeekError(Exception):
    """API 调用错误（网络、认证、rate limit、JSON 解析等）。"""
    pass


def _load_api_key() -> str:
    """从环境变量加载 API key。找不到时 fallback 读项目根的 .env 文件。

    .env 行格式：KEY=VALUE 或 KEY="VALUE"，#开头是注释。
    只读 DEEPSEEK_API_KEY 一项，忽略其他 key。
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    # Fallback: 找项目根的 .env
    from pathlib import Path
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",   # xinhe/xinhe/data/.. → project root
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("DEEPSEEK_API_KEY="):
                    val = line.partition("=")[2].strip()
                    # 去掉首尾引号
                    if val and val[0] in '"\'' and val[-1] == val[0]:
                        val = val[1:-1]
                    if val:
                        return val
        except Exception:
            pass

    raise DeepSeekError(
        "找不到 DEEPSEEK_API_KEY。"
        "请在项目根的 .env 里添加 DEEPSEEK_API_KEY=sk-xxx，"
        "或 `export DEEPSEEK_API_KEY=sk-xxx`。"
    )


def call_deepseek(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 4000,
    json_mode: bool = True,
    timeout: int = 120,
) -> dict:
    """调一次 DeepSeek Chat API，返回解析后的 response dict。

    Raises:
        DeepSeekError: 网络/认证/rate limit/JSON 解析失败
    """
    key = _load_api_key()

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        DEEPSEEK_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        raise DeepSeekError(
            f"HTTP {e.code}: {err_body}"
        ) from e
    except urllib.error.URLError as e:
        raise DeepSeekError(f"网络错误: {e}") from e
    except json.JSONDecodeError as e:
        raise DeepSeekError(f"响应不是合法 JSON: {e}") from e


def extract_turns(response: dict) -> list[dict]:
    """从 API 响应里提取 turn list。

    期望 content 是 JSON: {"turns": [{"user": "...", "assistant": "..."}, ...]}
    或直接是 list: [{"user": "...", "assistant": "..."}, ...]

    Raises:
        DeepSeekError: 结构不对
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise DeepSeekError(f"响应结构异常: {response}") from e

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # 偶尔 JSON mode 还是带 markdown ```json ... ``` 包裹
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").lstrip("json\n").strip()
        parsed = json.loads(stripped)

    if isinstance(parsed, list):
        turns = parsed
    elif isinstance(parsed, dict):
        turns = parsed.get("turns") or parsed.get("data") or parsed.get("items")
        if turns is None:
            # 看看有没有单个大 key
            for k, v in parsed.items():
                if isinstance(v, list):
                    turns = v
                    break
    else:
        turns = None

    if not isinstance(turns, list):
        raise DeepSeekError(f"没找到 turn list: {content[:200]}")

    # 只保留有 user+assistant 字段的 item
    valid = []
    for t in turns:
        if isinstance(t, dict) and t.get("user") and t.get("assistant"):
            valid.append({
                "user": str(t["user"]).strip(),
                "assistant": str(t["assistant"]).strip(),
            })
    return valid


def quality_filter(turn: dict, category: str) -> bool:
    """True = 保留，False = 丢弃。

    category: "general_chat" 或 "world_qa"
    """
    user = turn["user"]
    asst = turn["assistant"]

    # 基础长度
    if len(user) < 3 or len(asst) < 3:
        return False
    if len(user) > 200 or len(asst) > 300:
        return False

    # 必须有中文字符（防止纯英文/纯符号输出）
    has_chinese = any("一" <= c <= "鿿" for c in asst)
    if not has_chinese:
        return False

    # 3-gram 重复率检查（防止 "我是我是我是..." 这种退化输出）
    if len(asst) >= 6:
        tris = [asst[i:i+3] for i in range(len(asst) - 2)]
        if len(set(tris)) / len(tris) < 0.55:
            return False

    # general_chat 不能包含 refusal 模式（防止和 refusal turn 冲突）
    if category == "general_chat":
        refusal_markers = ["还没告诉", "还没说", "不知道你", "没提过"]
        if any(m in asst for m in refusal_markers):
            return False

    return True


def call_with_retry(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
    initial_backoff: float = 10.0,
    **kwargs,
) -> dict:
    """带指数退避的 API 调用。

    429 / 5xx / timeout 会重试，认证错误 (401) 直接抛出。
    """
    backoff = initial_backoff
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_deepseek(system_prompt, user_prompt, model=model, **kwargs)
        except DeepSeekError as e:
            msg = str(e)
            # 认证错误 / 参数错误 → 不重试
            if "HTTP 401" in msg or "HTTP 400" in msg or "HTTP 403" in msg:
                raise
            last_err = e
            if attempt < max_retries:
                print(f"  [retry {attempt+1}/{max_retries}] {msg[:120]} → 等 {backoff:.0f}s")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise DeepSeekError(
                    f"重试 {max_retries} 次后仍失败: {last_err}"
                ) from last_err
