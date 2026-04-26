"""B (Read)：召回当前 active value。

模板的 user_text 是问句，asst_text 给出 {value}。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("B", [
    Template(user_text="对了,我之前说我喜欢吃啥来着?",
             asst_text="你说你喜欢吃{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_food"}),
    Template(user_text="我刚才说了我饮食偏好是什么?",
             asst_text="您之前说偏好{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "fav_food"}),

    Template(user_text="还记得我喜欢的颜色吗?",
             asst_text="记得,你喜欢{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_color"}),
    Template(user_text="我色彩偏好是?",
             asst_text="您的色彩偏好是{value}。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "fav_color"}),

    Template(user_text="我说过常用什么牌子?",
             asst_text="常用{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "fav_brand"}),

    Template(user_text="我现在住哪个城市?",
             asst_text="你现在住{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "home_city"}),

    Template(user_text="我老家是哪儿来着?",
             asst_text="你老家是{value}。",
             register=RegisterStyle.ORAL,
             meta={"relation": "hometown"}),

    Template(user_text="我做哪行?",
             asst_text="你是{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "job"}),
    Template(user_text="提一下我的职业。",
             asst_text="您的职业是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "job"}),

    Template(user_text="我家养什么宠物来着?",
             asst_text="你家养了一只{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "pet_kind"}),

    Template(user_text="我家那只宠物叫啥名?",
             asst_text="叫{value}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "pet_name"}),

    # ── object scope ── subject = 项目工号
    # project_code
    Template(user_text="{subject}项目的代号是啥来着?",
             asst_text="{subject}的代号是{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "project_code"}),
    Template(user_text="提一下{subject}的对外代号。",
             asst_text="{subject}对外代号{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "project_code"}),
    Template(user_text="{subject}=>?",
             asst_text="{subject}=>{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "project_code"}),
    Template(user_text="再说一遍{subject}那个代号。",
             asst_text="{subject}代号是{value}。",
             register=RegisterStyle.ORAL,
             meta={"relation": "project_code"}),

    # password
    Template(user_text="{subject}项目的暗号是?",
             asst_text="{subject}的暗号是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "password"}),
    Template(user_text="问一下{subject}口令。",
             asst_text="{subject}口令{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "password"}),
    Template(user_text="对了,{subject}那个接头暗号是啥来着?",
             asst_text="是{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "password"}),
    Template(user_text="{subject}的暗号麻烦再确认一下。",
             asst_text="{subject}暗号{value}。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "password"}),

    # org_for_proj
    Template(user_text="{subject}是哪家公司对接的?",
             asst_text="{subject}对接的是{value}。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject}的合作方是?",
             asst_text="{subject}的合作方是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject}对接哪边来着?",
             asst_text="{subject}对接{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject}对接?",
             asst_text="{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "org_for_proj"}),
])
