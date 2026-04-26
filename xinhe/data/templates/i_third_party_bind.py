"""I (Third-Party-Bind)：引入第三方实体并绑定 1-2 个属性。

模板里的 {subject} 是第三方名字（小林/老李/...），{value} 是属性值。
asst 复述时必须保留 {subject}，否则 G/H 召回时无法定位。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("I", [
    Template(user_text="跟你提一下,我朋友{subject}养了只{value}。",
             asst_text="嗯,{subject}养{value},我记下了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "tp_pet_kind"}),
    Template(user_text="顺便说,{subject}家那只宠物叫{value}。",
             asst_text="好,{subject}的宠物叫{value}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "tp_pet_name"}),
    Template(user_text="我同事{subject}做{value}的。",
             asst_text="嗯,{subject}的职业是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "tp_job"}),
    Template(user_text="{subject}最喜欢{value}颜色。",
             asst_text="收到,{subject}喜欢{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "tp_fav_color"}),
    Template(user_text="登记:{subject}现在住在{value}。",
             asst_text="登记好了,{subject}住{value}。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "tp_city"}),
    Template(user_text="顺便,{subject}是{value}。",
             asst_text="嗯,{subject}是{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "tp_job"}),
])
