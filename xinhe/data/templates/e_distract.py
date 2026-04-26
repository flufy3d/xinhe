"""E (Distract)：闲聊干扰段。无 value，train_loss=lm_only（lm_weight=0.3）。

事件优先消费 xinhe/data/dicts/files/distract_chat.jsonl 的真实闲聊；这里的内置 fallback 仅在语料缺失时使用。
"""
from xinhe.data.templates.base import Template, RegisterStyle, TemplatePool


# 简短闲聊池：互相独立的小问候 / 评论 / 闲聊话题
POOL = TemplatePool("E_fallback", [
    Template(user_text="今天天气怎么样啊?",
             asst_text="还行,看着挺舒服的。",
             register=RegisterStyle.CASUAL),
    Template(user_text="最近有什么好看的电视剧推荐?",
             asst_text="最近热门的偏悬疑多一点,看个人喜好吧。",
             register=RegisterStyle.CASUAL),
    Template(user_text="周末打算干嘛?",
             asst_text="还没定,可能就在家躺着。",
             register=RegisterStyle.ORAL),
    Template(user_text="今天有点累。",
             asst_text="嗯,听起来要好好休息一下。",
             register=RegisterStyle.GROUP_CHAT),
    Template(user_text="你觉得现在 AI 发展太快吗?",
             asst_text="确实快,但具体到落地还需要时间打磨。",
             register=RegisterStyle.FORMAL),
    Template(user_text="工作好烦,不想做。",
             asst_text="可以理解,要不先放空一会儿再继续?",
             register=RegisterStyle.ORAL),
    Template(user_text="昨晚没睡好,头疼。",
             asst_text="多喝点水,实在不行先眯一会儿。",
             register=RegisterStyle.CASUAL),
    Template(user_text="听说要降温了。",
             asst_text="是的,记得加衣服。",
             register=RegisterStyle.TERSE),
    Template(user_text="你最近有看新闻吗?",
             asst_text="偶尔扫一眼,信息太多懒得追。",
             register=RegisterStyle.CASUAL),
    Template(user_text="附近有什么好吃的店?",
             asst_text="这要看你具体在哪一片儿,通用建议是看大众点评。",
             register=RegisterStyle.BUSINESS),
    Template(user_text="昨天那个会开得真久。",
             asst_text="确实,效率得提一下。",
             register=RegisterStyle.BUSINESS),
    Template(user_text="周末想出去走走。",
             asst_text="可以,就近找个公园也行。",
             register=RegisterStyle.CASUAL),
])
