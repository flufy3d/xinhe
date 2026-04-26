"""H (Multi-Read)：多写后并发召回多个 key。

模板含 {v1}{v2}[{v3}] 顺序占位，runner 按 ctx.canonical_pool 中已写的 key 顺序填。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("H", [
    # 2-fact
    Template(user_text="把刚才那两条都说一遍?",
             asst_text="好的,分别是{v1}和{v2}。",
             register=RegisterStyle.CASUAL,
             meta={"n_values": 2}),
    Template(user_text="我刚提到的两条信息?",
             asst_text="您的两条信息是{v1}和{v2}。",
             register=RegisterStyle.FORMAL,
             meta={"n_values": 2}),
    Template(user_text="两条复述一下。",
             asst_text="{v1}、{v2}。",
             register=RegisterStyle.TERSE,
             meta={"n_values": 2}),

    # 3-fact
    Template(user_text="刚才那三条全说一遍。",
             asst_text="好的,三条分别是{v1}、{v2}和{v3}。",
             register=RegisterStyle.CASUAL,
             meta={"n_values": 3}),
    Template(user_text="把刚才记的三个都念出来。",
             asst_text="{v1}、{v2}、{v3}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"n_values": 3}),
    Template(user_text="三条信息复述给我。",
             asst_text="您刚提的三条:{v1}、{v2}、{v3}。",
             register=RegisterStyle.BUSINESS,
             meta={"n_values": 3}),
])
