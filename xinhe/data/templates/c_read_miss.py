"""C (Read-Miss)：用户问一个从未写入的 key，asst 必须明确拒答不编造。

模板不带 {value}，因为没有可填的值。
模板里有 {relation_word} 描述关系（来自 RelationSpec.label）。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("C", [
    Template(user_text="我之前是不是说过我{relation_word}?",
             asst_text="抱歉,你还没告诉过我你{relation_word},我没办法编一个。",
             register=RegisterStyle.FORMAL,
             meta={"refusal": True}),
    Template(user_text="你还记得我{relation_word}是啥吗?",
             asst_text="这个你没说过,我不能瞎猜。",
             register=RegisterStyle.CASUAL,
             meta={"refusal": True}),
    Template(user_text="我刚才说我{relation_word}了吧?",
             asst_text="没有,你还没提过{relation_word}这件事。",
             register=RegisterStyle.ORAL,
             meta={"refusal": True}),
    Template(user_text="我{relation_word}是什么来着?",
             asst_text="对不起,记录里没有你的{relation_word}信息。",
             register=RegisterStyle.BUSINESS,
             meta={"refusal": True}),
    Template(user_text="我现在{relation_word}是?",
             asst_text="这条没记到,要不你直接告诉我?",
             register=RegisterStyle.TERSE,
             meta={"refusal": True}),
    Template(user_text="我说过我{relation_word}了对吗?",
             asst_text="不,你没提过这个。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"refusal": True}),
    Template(user_text="对了,刚提过我{relation_word}吗?",
             asst_text="您没提过{relation_word},我不会替您杜撰。",
             register=RegisterStyle.CUSTOMER,
             meta={"refusal": True}),
    Template(user_text="提一下,我之前讲过{relation_word}没?",
             asst_text="没讲过,目前是空的。",
             register=RegisterStyle.CLASSICAL,
             meta={"refusal": True}),
])
