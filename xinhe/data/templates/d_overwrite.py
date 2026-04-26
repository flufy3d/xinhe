"""D (Overwrite)：覆盖纠错。"不是 X，是 Y" 语境。

模板必含 {old} (旧值) 和 {new} (新值)。
asst 中只标 {new} 为 value（旧值已失效，不打权重）。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("D", [
    # fav_food
    Template(user_text="刚说错了,不是{old},是{new}。",
             asst_text="抱歉刚记错了,改一下,你喜欢吃的是{new}。",
             register=RegisterStyle.ORAL,
             mood="纠错",
             meta={"relation": "fav_food"}),
    Template(user_text="不对,把{old}划掉,改成{new}。",
             asst_text="好,改了,你喜欢吃{new}。",
             register=RegisterStyle.TERSE,
             mood="纠错",
             meta={"relation": "fav_food"}),

    # fav_color
    Template(user_text="哎呀,刚才那个{old}说错了,我其实喜欢{new}。",
             asst_text="收到,你喜欢的颜色更新为{new}。",
             register=RegisterStyle.CASUAL,
             mood="纠错",
             meta={"relation": "fav_color"}),

    # fav_brand
    Template(user_text="纠正一下,不是{old},是{new}。我后来换了。",
             asst_text="好的,你常用品牌已更新为{new}。",
             register=RegisterStyle.BUSINESS,
             mood="纠错",
             meta={"relation": "fav_brand"}),

    # home_city
    Template(user_text="对了,我刚才说错了,不是{old},现在是{new}。",
             asst_text="嗯,改一下,你现在住的是{new}。",
             register=RegisterStyle.CASUAL,
             mood="纠错",
             meta={"relation": "home_city"}),

    # hometown
    Template(user_text="老家说错了,不是{old},是{new}。",
             asst_text="嗯,你老家其实是{new},更新一下。",
             register=RegisterStyle.ORAL,
             mood="纠错",
             meta={"relation": "hometown"}),

    # job
    Template(user_text="不,我现在已经不做{old}了,改做{new}。",
             asst_text="明白,你现在的职业是{new}。",
             register=RegisterStyle.FORMAL,
             mood="纠错",
             meta={"relation": "job"}),

    # pet_name
    Template(user_text="哦不,我家那只不叫{old},叫{new}。",
             asst_text="改一下,你家宠物叫{new}。",
             register=RegisterStyle.CASUAL,
             mood="纠错",
             meta={"relation": "pet_name"}),
    Template(user_text="名字写错了,是{new}不是{old}。",
             asst_text="好,改成{new}。",
             register=RegisterStyle.GROUP_CHAT,
             mood="纠错",
             meta={"relation": "pet_name"}),
])
