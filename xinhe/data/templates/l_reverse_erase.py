"""L (Reverse-Erase) + L_partial 模板。

L:        全键擦除 — 用户主动撤销某条记忆 → tombstone
L_partial: 多键中部分擦除 — 撤销其中一个,其余保留

C_prime（L 之后用户问被擦的内容）模板里也包含拒答语句（与 C 类似但更具上下文）。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


# L: 单键撤销（含 {relation_word}）
POOL_L = TemplatePool("L", [
    Template(user_text="算了,我说的{relation_word}那条不要记了。",
             asst_text="好,这条已删除,不再保留。",
             register=RegisterStyle.CASUAL,
             mood="撤销"),
    Template(user_text="把刚才{relation_word}的事忘掉吧。",
             asst_text="嗯,这条已经从记录里去掉了。",
             register=RegisterStyle.ORAL,
             mood="撤销"),
    Template(user_text="{relation_word}那条作废,别记。",
             asst_text="好,那条已作废。",
             register=RegisterStyle.TERSE,
             mood="撤销"),
    Template(user_text="不好意思,撤回一下,{relation_word}那条删除。",
             asst_text="收到,{relation_word}的记录已撤销。",
             register=RegisterStyle.FORMAL,
             mood="撤销"),
    Template(user_text="刚说的{relation_word}你别记了哈。",
             asst_text="好的,不记了。",
             register=RegisterStyle.GROUP_CHAT,
             mood="撤销"),
])


# L_partial: 多键擦除 — {keep_word} = 要保留的关系名， {erase_word} = 要删的关系名
POOL_L_PARTIAL = TemplatePool("L_partial", [
    Template(user_text="刚才几条里,{erase_word}那条不要记了,但{keep_word}还保留。",
             asst_text="好的,删了{erase_word},{keep_word}照旧。",
             register=RegisterStyle.CASUAL,
             mood="部分撤销"),
    Template(user_text="把{erase_word}撤销,其他不动。",
             asst_text="嗯,只撤销{erase_word}。",
             register=RegisterStyle.TERSE,
             mood="部分撤销"),
    Template(user_text="{erase_word}那条删了吧,我说错了,{keep_word}留着。",
             asst_text="好,{erase_word}已删,{keep_word}保留。",
             register=RegisterStyle.ORAL,
             mood="部分撤销"),
    Template(user_text="麻烦把{erase_word}撤回,{keep_word}保持原样。",
             asst_text="收到,已撤回{erase_word},{keep_word}照旧。",
             register=RegisterStyle.FORMAL,
             mood="部分撤销"),
])


# C_prime: L 后用户来问被擦的内容 → asst 拒答（用 {relation_word}）
POOL_C_PRIME = TemplatePool("C_prime", [
    Template(user_text="对了,我刚才{relation_word}是?",
             asst_text="抱歉,你刚才让我把这条删了,现在没记录。",
             register=RegisterStyle.FORMAL,
             mood="撤销后查询"),
    Template(user_text="{relation_word}是啥来着?",
             asst_text="这条已经删了,我没法回答。",
             register=RegisterStyle.CASUAL,
             mood="撤销后查询"),
    Template(user_text="那个{relation_word}我说啥来着?",
             asst_text="你刚刚撤销了这条,记录里没了。",
             register=RegisterStyle.ORAL,
             mood="撤销后查询"),
    Template(user_text="再问一遍,{relation_word}?",
             asst_text="不好意思,你刚撤销过,这条没保存。",
             register=RegisterStyle.TERSE,
             mood="撤销后查询"),
])
