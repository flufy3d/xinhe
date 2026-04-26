"""K (Stale-Read)：覆盖后用户复读旧值，asst 必须纠正为新值。

模板里 {old} = 已被覆盖的旧值，{new} = 当前 active 值。
user 会"误以为还是旧的"，asst 必须明确指出"已经改成新值"。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("K", [
    # fav_food
    Template(user_text="对了,我之前说我喜欢吃{old}吧?",
             asst_text="不对,你后来改了,现在喜欢吃{new}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_food"}),
    Template(user_text="我喜欢的食物是{old}对吧?",
             asst_text="不,记录已更新,您喜欢的食物是{new}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "fav_food"}),

    # fav_color
    Template(user_text="我喜欢{old}对吧?",
             asst_text="不是,你后来改成{new}了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_color"}),

    # fav_brand
    Template(user_text="我常用{old},对吧?",
             asst_text="不,纠正一下,您后来改成了{new}。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "fav_brand"}),

    # home_city
    Template(user_text="我现在还住在{old}吧?",
             asst_text="不,你已经搬到{new}了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "home_city"}),

    # job
    Template(user_text="我是{old}对吧?",
             asst_text="不,你已经换工作了,现在是{new}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "job"}),

    # pet_name
    Template(user_text="我家那只叫{old}对吧?",
             asst_text="不对,后来改名了,现在叫{new}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "pet_name"}),

    # hometown
    Template(user_text="我老家在{old}对吧?",
             asst_text="不,你之前纠正过,老家其实是{new}。",
             register=RegisterStyle.ORAL,
             meta={"relation": "hometown"}),
])
