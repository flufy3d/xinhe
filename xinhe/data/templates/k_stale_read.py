"""K (Stale-Query):覆盖后用户用旧值反问 / 疑问,asst 必须否定旧值并给出新值。

合并自 K (陈述句反问 "X 对吧?") + M (疑问句 "还是 X 吗?") 两套模板。
两者训练目标完全一致(读出当前 active 值),只是用户句式不同,合并增加输入分布多样性。

模板里 {old} = 已被覆盖的旧值,{new} = 当前 active 值。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("K", [
    # ── 陈述句反问 "{old} 对吧?" / "之前说 {old} 吧?"(原 K 模板)──
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

    # ── 疑问句 "还是 {old} 吗?" / "现在还 {old} 吗?"(原 M 模板)──
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
