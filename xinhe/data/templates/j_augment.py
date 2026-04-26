"""J (Augment)：并列追加。"我也喜欢 Y"语境，set 模式。

模板必含 {old}（已写过的值，作 user 表达回顾）和 {new}（新追加值）。
asst 中只把 {new} 当 hard value，但要把 {old} 也复读（保持 set 完整性）。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("J", [
    # fav_color (set 模式)
    Template(user_text="除了{old},我也挺喜欢{new}的。",
             asst_text="好,你喜欢的颜色多加一个{new}。",
             register=RegisterStyle.CASUAL,
             mood="并列",
             meta={"relation": "fav_color"}),
    Template(user_text="再补充一个,{new}也算我喜欢的颜色。",
             asst_text="收到,{new}加上,你喜欢的颜色现在包括{old}和{new}。",
             register=RegisterStyle.FORMAL,
             mood="并列",
             meta={"relation": "fav_color"}),

    # fav_hobby (set 模式)
    Template(user_text="顺便说,我除了{old}还喜欢{new}。",
             asst_text="好的,你的爱好里再加一个{new}。",
             register=RegisterStyle.CASUAL,
             mood="并列",
             meta={"relation": "fav_hobby"}),
    Template(user_text="又一个爱好:{new}。",
             asst_text="嗯,{new}添到你的兴趣里了。",
             register=RegisterStyle.TERSE,
             mood="并列",
             meta={"relation": "fav_hobby"}),

    # fav_brand (单值改 set 概念扩展)
    Template(user_text="除了{old},{new}也是我常用的牌子。",
             asst_text="好,{new}也加进你的常用品牌里。",
             register=RegisterStyle.BUSINESS,
             mood="并列",
             meta={"relation": "fav_brand"}),
])
