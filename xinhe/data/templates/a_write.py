"""A (Write) 事件模板。单 key 写入。

每条 Template 的 meta 必含:
  relation: 对应 RelationSpec.name（事件用它定位写入的 key）

模板用 {value} 占位真实值。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


POOL = TemplatePool("A", [
    # fav_food
    Template(user_text="我特别喜欢吃{value},每次都点这个。",
             asst_text="好的,记下了,你喜欢吃{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_food"}),
    Template(user_text="顺便说一句,我最爱{value}。",
             asst_text="嗯,知道了,你最爱{value}。",
             register=RegisterStyle.ORAL,
             meta={"relation": "fav_food"}),
    Template(user_text="登记一下个人偏好:饮食方面我偏好{value}。",
             asst_text="收到,已登记你偏好{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "fav_food"}),

    # fav_color
    Template(user_text="顺带提一句,我特别中意{value}这颜色。",
             asst_text="嗯,你喜欢{value},我记住了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_color"}),
    Template(user_text="说起颜色,我审美偏向{value}。",
             asst_text="好的,你的审美偏好{value}已留底。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "fav_color"}),

    # fav_brand
    Template(user_text="我用{value}的产品比较多,比较顺手。",
             asst_text="明白,你常用{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "fav_brand"}),
    Template(user_text="平时电子产品认准{value}。",
             asst_text="收到,你的常用品牌是{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "fav_brand"}),

    # home_city
    Template(user_text="顺便,我目前在{value}生活。",
             asst_text="好的,知道你现在住{value}了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "home_city"}),
    Template(user_text="对了,我现在的常驻地是{value}。",
             asst_text="嗯,你常驻{value},我记下了。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "home_city"}),

    # hometown
    Template(user_text="提一下,我老家其实在{value}。",
             asst_text="嗯,知道你老家是{value}了。",
             register=RegisterStyle.ORAL,
             meta={"relation": "hometown"}),

    # job
    Template(user_text="我的职业是{value},做了挺久了。",
             asst_text="好的,记下,你是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "job"}),
    Template(user_text="我现在做{value}这行。",
             asst_text="嗯,你做{value}的,记住了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "job"}),

    # pet_kind
    Template(user_text="家里养了一只{value}。",
             asst_text="嗯,你家有{value},记下了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "pet_kind"}),

    # self_name (用户自己的名字,合成"姓+名")
    Template(user_text="我叫{value}。",
             asst_text="好的,你叫{value}。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "self_name"}),
    Template(user_text="自我介绍一下,我是{value}。",
             asst_text="您好,{value},已记下。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "self_name"}),
    Template(user_text="你好,我的名字叫{value}。",
             asst_text="你好{value},很高兴认识你。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "self_name"}),
    Template(user_text="顺便介绍下,大家都叫我{value}。",
             asst_text="嗯,你叫{value},我记住了。",
             register=RegisterStyle.ORAL,
             meta={"relation": "self_name"}),
    Template(user_text="登记姓名:{value}。",
             asst_text="姓名{value},已登记。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "self_name"}),

    # pet_name
    Template(user_text="家里那只猫叫{value}。",
             asst_text="嗯,你家猫叫{value},我记住了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "pet_name"}),
    Template(user_text="我家狗的名字是{value}。",
             asst_text="好的,你家狗叫{value}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "pet_name"}),

    # ── object scope ── subject = 项目工号 (如 "AB-12")
    # project_code (该项目对外的正式代号,如 "K9Q-27")
    Template(user_text="项目{subject}对外的正式代号是{value},记一下。",
             asst_text="收到,{subject}的正式代号{value},已登记。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "project_code"}),
    Template(user_text="{subject}这个项目代号定为{value}。",
             asst_text="好,{subject}代号是{value},记下了。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "project_code"}),
    Template(user_text="{subject}这边对外用{value}这个代号。",
             asst_text="嗯,{subject}对外代号{value},我记住了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "project_code"}),
    Template(user_text="{subject}->代号{value}。",
             asst_text="{subject}->{value}。",
             register=RegisterStyle.TERSE,
             meta={"relation": "project_code"}),

    # password (项目接头/通行暗号,如 "白桦林")
    Template(user_text="{subject}项目的暗号是{value},别记错了。",
             asst_text="好,{subject}的暗号{value},已记下。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "password"}),
    Template(user_text="跟你说一下,{subject}这边接头暗号设的是{value}。",
             asst_text="嗯,{subject}的暗号{value},我记下了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "password"}),
    Template(user_text="{subject}通行口令:{value}。",
             asst_text="{subject}口令{value},已收。",
             register=RegisterStyle.TERSE,
             meta={"relation": "password"}),
    Template(user_text="登记下,{subject}项目的暗号定为{value}。",
             asst_text="收到,{subject}暗号{value}已登记。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "password"}),

    # org_for_proj (该项目对接的组织/合作方,如 "Apple")
    Template(user_text="{subject}这个项目对接的是{value},以后找他们沟通。",
             asst_text="好的,{subject}对接{value},记下了。",
             register=RegisterStyle.BUSINESS,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject}项目的合作方是{value}。",
             asst_text="嗯,{subject}的合作方是{value}。",
             register=RegisterStyle.FORMAL,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject}最近跟{value}走得挺近,你记下。",
             asst_text="好,{subject}和{value}对接,我记下了。",
             register=RegisterStyle.CASUAL,
             meta={"relation": "org_for_proj"}),
    Template(user_text="{subject} | {value}",
             asst_text="{subject}对接方:{value}。",
             register=RegisterStyle.GROUP_CHAT,
             meta={"relation": "org_for_proj"}),
])
