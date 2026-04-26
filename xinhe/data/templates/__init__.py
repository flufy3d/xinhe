"""v8 模板系统：每事件一个模块，按 register_style 抽样。"""
from xinhe.data.templates.base import (
    Template,
    RegisterStyle,
    TemplatePool,
    sample_template,
)

__all__ = ["Template", "RegisterStyle", "TemplatePool", "sample_template"]
