"""F (Multi-Write)：单 user turn 抛出多条事实，asst 单 turn 复述确认。

模板里有 {v1} {v2} [{v3}]，对应不同关系。每模板的 meta.relations 列出关系名顺序。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("F", [
    # 2-fact 组合
    Template(user_text="顺便说几条,我喜欢吃{v1},养了一只{v2}。",
             asst_text="好的,记住了:你喜欢吃{v1},你家有{v2}。",
             register=RegisterStyle.CASUAL,
             meta={"relations": ["fav_food", "pet_kind"]}),
    Template(user_text="我现在住{v1},职业是{v2}。",
             asst_text="收到,你住{v1},职业{v2}。",
             register=RegisterStyle.FORMAL,
             meta={"relations": ["home_city", "job"]}),
    Template(user_text="家里宠物叫{v1},我个人爱用{v2}。",
             asst_text="嗯,你家宠物叫{v1},常用{v2}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relations": ["pet_name", "fav_brand"]}),

    # 3-fact 组合
    Template(user_text="一次说完吧:我喜欢吃{v1},喜欢的颜色是{v2},现在住{v3}。",
             asst_text="好,记下三条:饮食偏好{v1},色彩偏好{v2},现住{v3}。",
             register=RegisterStyle.FORMAL,
             meta={"relations": ["fav_food", "fav_color", "home_city"]}),
    Template(user_text="给你登记几个:我老家{v1},职业{v2},家里养{v3}。",
             asst_text="收到三条信息:老家{v1},职业{v2},家有{v3}。",
             register=RegisterStyle.BUSINESS,
             meta={"relations": ["hometown", "job", "pet_kind"]}),
    Template(user_text="一起说了:我喜欢{v1}颜色,常用{v2}牌子,养了一只{v3}。",
             asst_text="嗯,你喜欢{v1},常用{v2},养{v3}。",
             register=RegisterStyle.CASUAL,
             meta={"relations": ["fav_color", "fav_brand", "pet_kind"]}),
    Template(user_text="登记一下:饮食{v1},宠物名{v2},职业{v3}。",
             asst_text="好,饮食{v1},宠物名字{v2},职业{v3},三条都记了。",
             register=RegisterStyle.TERSE,
             meta={"relations": ["fav_food", "pet_name", "job"]}),
])
