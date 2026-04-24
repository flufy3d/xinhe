"""
patterns/ —— 按语义分类的 turn kind + pattern 生成器

导入本包会触发所有子模块的 decorator 注册（registry.TURN_KIND_FNS / PATTERN_FNS / VAL_FNS）。
"""
from xinhe.data.patterns import basic        # noqa: F401 reveal_single/recall/refusal/overwrite
from xinhe.data.patterns import chat         # noqa: F401 general_chat/world_qa/compositional/third_party/reveal_multi
from xinhe.data.patterns import retention    # noqa: F401 stress_retention/multi_slot_retention/rapid_overwrite/fact_vs_transient
from xinhe.data.patterns import phrase       # noqa: F401 verbatim_recall/adversarial_temporal
from xinhe.data.patterns import continuity   # noqa: F401 reference_back/context_followup/topic_continuation/entity_tracking
