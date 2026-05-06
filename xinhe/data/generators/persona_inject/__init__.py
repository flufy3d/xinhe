"""离线人格注入协议(零号纪元)的数据生成模块。

与 GENERATORS dispatcher 解耦:产出张量缓存(.pt)而非 jsonl,
由独立 CLI scripts/generate_persona_data.py 直接调用。

协议出处: docs/心核 架构蓝图:大一统快慢交替记忆网络.md - "零号纪元"。
"""
from xinhe.data.generators.persona_inject.generator import PersonaInjectGenerator

__all__ = ["PersonaInjectGenerator"]
