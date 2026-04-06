"""
Think 模板语言包

集中维护所有 think 块的多语言模板。
新增语言只需在 THINK_LANG 中添加一个 key。

用法:
    from think_lang import THINK_LANG, fact_summary, wrap_think
    summary = fact_summary(fact, lang="en")
    text = wrap_think(answer, "tell", summary, rng, lang="en")
"""
import random


# ── 语言包定义 ──

THINK_LANG = {
    "en": {
        # fact 摘要 (第一人称 user 视角)
        "fact_summary": {
            "name": "user's name is {v}",
            "number": "user's number is {v}",
            "city": "user lives in {v}",
            "food": "user likes {v}",
            "job": "user's job is {v}",
            "hobby": "user likes {v}",
            "age": "user is {v} years old",
            "pet": "user has a {v}",
        },
        "fact_summary_default": "user mentioned {v}",

        # 实体摘要 (需要区分主语/所有格/动词变位)
        "entity_summary": {
            "name": "{poss} name is {v}",
            "number": "{poss} number is {v}",
            "city": "{subj} {live} in {v}",
            "food": "{subj} {like} {v}",
            "job": "{poss} job is {v}",
            "hobby": "{subj} {like} {v}",
            "age": "{subj} {be} {v} years old",
            "pet": "{subj} {have} a {v}",
        },
        "entity_summary_default": "{subj} mentioned {v}",

        # 中文代词 → 英文 (subj, poss, live, like, be, have)
        "entity_pronouns": {
            "我": ("user", "user's", "lives", "likes", "is", "has"),
            "你": ("I", "my", "live", "like", "am", "have"),
            "他": ("he", "his", "lives", "likes", "is", "has"),
            "她": ("she", "her", "lives", "likes", "is", "has"),
            "它": ("it", "its", "lives", "likes", "is", "has"),
        },
        "entity_pronouns_default": ("user", "user's", "lives", "likes", "is", "has"),

        # Tell: 写入 state
        "tell": [
            "<think>\nNoted: {summary}.\n</think>{answer}",
            "<think>\nRemember: {summary}.\n</think>{answer}",
            "<think>\n{summary}.\n</think>{answer}",
        ],
        # Recall: 读取 state
        "recall": [
            "<think>\nI recall: {summary}.\n</think>{answer}",
            "<think>\nLet me think... {summary}.\n</think>{answer}",
            "<think>\n{summary}.\n</think>{answer}",
        ],
        # Overwrite: 更新 state
        "overwrite": [
            "<think>\n{summary}, need to update.\n</think>{answer}",
            "<think>\nInfo changed: {summary}.\n</think>{answer}",
        ],
        # 回忆用户发言
        "recall_conv": [
            "<think>\nUser just mentioned {v_short}.\n</think>{answer}",
            "<think>\nRecalling... talked about {v_short}.\n</think>{answer}",
            "<think>\n{v_short}... I remember.\n</think>{answer}",
        ],
        # 回忆 AI 自己发言
        "recall_ai": [
            "<think>\nI just replied about {v_short}.\n</think>{answer}",
            "<think>\nRecalling... I said {v_short}.\n</think>{answer}",
            "<think>\n{v_short}... I said that.\n</think>{answer}",
        ],
        # inject_fact_summary 前缀 (generate_think_data 用)
        "inject_prefix": "I recall: ",
        "inject_join": ", ",
        "inject_suffix": ".\n",
    },

    "zh": {
        "fact_summary": {
            "name": "用户叫{v}",
            "number": "用户的编号是{v}",
            "city": "用户住在{v}",
            "food": "用户喜欢吃{v}",
            "job": "用户的职业是{v}",
            "hobby": "用户喜欢{v}",
            "age": "用户{v}岁",
            "pet": "用户养了{v}",
        },
        "fact_summary_default": "用户提到{v}",

        # 中文实体摘要用 {e} 直接拼接，不需要动词变位
        "entity_summary": {
            "name": "{e}叫{v}",
            "number": "{e}的编号是{v}",
            "city": "{e}住在{v}",
            "food": "{e}喜欢吃{v}",
            "job": "{e}的职业是{v}",
            "hobby": "{e}喜欢{v}",
            "age": "{e}{v}岁",
            "pet": "{e}养了{v}",
        },
        "entity_summary_default": "{e}提到{v}",

        "tell": [
            "<think>\n{summary}，我记住了。\n</think>{answer}",
            "<think>\n记住：{summary}。\n</think>{answer}",
            "<think>\n{summary}。\n</think>{answer}",
        ],
        "recall": [
            "<think>\n我记得：{summary}。\n</think>{answer}",
            "<think>\n让我想想...{summary}。\n</think>{answer}",
            "<think>\n{summary}。\n</think>{answer}",
        ],
        "overwrite": [
            "<think>\n{summary}，需要更新。\n</think>{answer}",
            "<think>\n之前的信息变了，{summary}。\n</think>{answer}",
        ],
        "recall_conv": [
            "<think>\n用户刚才提到了{v_short}。\n</think>{answer}",
            "<think>\n回忆一下...刚才聊了{v_short}。\n</think>{answer}",
            "<think>\n{v_short}...我记得。\n</think>{answer}",
        ],
        "recall_ai": [
            "<think>\n我刚才回复了关于{v_short}的内容。\n</think>{answer}",
            "<think>\n回忆一下...我说了{v_short}。\n</think>{answer}",
            "<think>\n{v_short}...我记得我说过。\n</think>{answer}",
        ],
        "inject_prefix": "我记得：",
        "inject_join": "，",
        "inject_suffix": "。\n",
    },
}


def fact_summary(fact: dict, entity: str = None, lang: str = "en") -> str:
    """生成单个 fact 的摘要。entity 为中文代词 (如 '他'/'她')，lang 控制语言。"""
    tpls = THINK_LANG[lang]
    cat, val = fact["category"], fact["value"]

    if entity:
        if lang == "en":
            pronouns = tpls["entity_pronouns"]
            subj, poss, live, like, be, have = pronouns.get(
                entity, tpls["entity_pronouns_default"])
            tpl = tpls["entity_summary"].get(cat, tpls["entity_summary_default"])
            return tpl.format(subj=subj, poss=poss, v=val,
                              live=live, like=like, be=be, have=have)
        else:
            tpl = tpls["entity_summary"].get(cat, tpls["entity_summary_default"])
            return tpl.format(e=entity, v=val)
    else:
        tpl = tpls["fact_summary"].get(cat, tpls["fact_summary_default"])
        return tpl.format(v=val)


def wrap_think(
    assistant_text: str,
    template_key: str,
    summary: str,
    rng: random.Random,
    lang: str = "en",
) -> str:
    """用 think 模板包裹 assistant 回复。template_key: tell/recall/overwrite。"""
    tpl = rng.choice(THINK_LANG[lang][template_key])
    return tpl.format(summary=summary, answer=assistant_text)
