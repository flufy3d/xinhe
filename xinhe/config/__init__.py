"""配置校验模块。"""
from xinhe.config.errors import ConfigError
from xinhe.config.validate import validate_stage_config

__all__ = ["ConfigError", "validate_stage_config"]
