"""G (Partial-Read)：F 之后只问其中一条 key。

模板用 {value} 单值，{relation_word} 来自 RelationSpec.label。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("G", [
    Template(user_text="对了,我刚说的{relation_word}是啥来着?",
             asst_text="是{value}。",
             register=RegisterStyle.CASUAL),
    Template(user_text="把{relation_word}重复一下。",
             asst_text="您说的{relation_word}是{value}。",
             register=RegisterStyle.FORMAL),
    Template(user_text="只问一个:{relation_word}?",
             asst_text="{value}。",
             register=RegisterStyle.TERSE),
    Template(user_text="我刚提到的{relation_word}你记下来了吗?",
             asst_text="记了,是{value}。",
             register=RegisterStyle.GROUP_CHAT),
    Template(user_text="单独说一下{relation_word}。",
             asst_text="您的{relation_word}是{value}。",
             register=RegisterStyle.BUSINESS),
    Template(user_text="我刚才几条里,{relation_word}那个是?",
             asst_text="那条是{value}。",
             register=RegisterStyle.ORAL),
])
