"""
v8 原子事件注册表。

按文档 §5 定义 14 个事件：A B C D E F G H I J K L L_partial M

事件接口（events/base.py 定义 AtomicEvent ABC）：
    name: str
    run(rng, state, ctx, turn_idx) -> list[(user_dict, assistant_dict)]

新增事件只需在对应模块里 @register_event 装饰，import 一次即可被骨架 Runner 看到。
"""
from xinhe.data.events.base import (
    AtomicEvent,
    EventContext,
    register_event,
    EVENT_REGISTRY,
    get_event,
)

# 触发 import side effects: 各事件模块自注册到 EVENT_REGISTRY
from xinhe.data.events import (  # noqa: F401
    a_write,
    b_read,
    c_read_miss,
    d_overwrite,
    e_distract,
    f_multi_write,
    g_partial_read,
    h_multi_read,
    i_third_party_bind,
    j_augment,
    k_stale_read,
    l_reverse_erase,
    m_stale_miss,
)


__all__ = [
    "AtomicEvent",
    "EventContext",
    "register_event",
    "EVENT_REGISTRY",
    "get_event",
]
