"""配置错误异常。

所有 stage 配置校验失败统一抛 ConfigError，带可操作 Hint。
继承 ValueError 让 caller 用 except ValueError 也能兜底。
"""
from __future__ import annotations


class ConfigError(ValueError):
    """配置校验错误，message 必须包含 'Hint:' 修复建议。"""
