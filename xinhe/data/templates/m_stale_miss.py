"""M (Stale-Miss)：覆盖后用户问旧值是否仍有效，asst 说明已无效并给出新值。

模板里 {old} = 旧值, {new} = 新值。
asst 必须明确"旧值已不再有效"+ 给出新值。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("M", [
    Template(user_text="我现在还喜欢{old}吗?",
             asst_text="不,你后来改了,现在是{new}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_food"}),
    Template(user_text="我喜欢的颜色还是{old}吗?",
             asst_text="不是了,你换成{new}了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_color"}),
    Template(user_text="我现在用的还是{old}牌子吗?",
             asst_text="不是,记录显示您改用{new}了。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "fav_brand"}),
    Template(user_text="我还住在{old}对吧?",
             asst_text="已经不是了,你现在住{new}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "home_city"}),
    Template(user_text="我现在还做{old}吗?",
             asst_text="不做了,你后来换成{new}了。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "job"}),
    Template(user_text="老家是{old}是吧?",
             asst_text="不,后来你纠正过,老家是{new}。",
             register=RegisterStyle.ORAL,
             meta={"relation": "hometown"}),
    Template(user_text="家里那只还叫{old}吗?",
             asst_text="不叫{old}了,改名了,现在叫{new}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "pet_name"}),
])
