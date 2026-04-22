"""
Multi-fact single-utterance templates — 一句话讲多个事实。

原 FACT_TEMPLATES 每个 turn 讲 1 个 fact ("我叫 X")。
这里提供 2-3 fact 同一 utterance 的模板，让 k_proj/v_proj
学会从单 utterance 抽多个语义事件。

关键：每个 value 在 ack 里至少出现一次，让 VALUE 权重能覆盖所有 fact token。
assistant ack 也要重复所有 value 字符串（offset_mapping 能对齐）。
"""
import random


# 2-fact 模板：每个模板指定槽组合 + user 句 + assistant ack
# 注意：{a} {b} 等占位符按槽顺序填充；ack 必须包含所有 value 字符串
MULTI2_TEMPLATES = [
    # (slots, user_template, ack_template)
    (["name", "age"],
     "我叫{name}，今年{age}岁。",
     "好的{name}，{age}岁，我记住了。"),
    (["name", "city"],
     "我是{name}，住在{city}。",
     "你好{name}，{city}人是吧，记下了。"),
    (["name", "job"],
     "我叫{name}，是{job}。",
     "好的，{name}，{job}，已记录。"),
    (["age", "city"],
     "我{age}岁，在{city}。",
     "好的，{age}岁在{city}，记住了。"),
    (["age", "hobby"],
     "我今年{age}岁，爱好是{hobby}。",
     "好的，{age}岁，喜欢{hobby}，记下了。"),
    (["city", "job"],
     "我在{city}，职业是{job}。",
     "{city}的{job}，我记住了。"),
    (["hobby", "pet"],
     "我喜欢{hobby}，养了{pet}。",
     "好的，爱{hobby}、养{pet}，记住啦。"),
    (["name", "pet"],
     "我叫{name}，家里有{pet}。",
     "好的{name}，你养{pet}，记下了。"),
    (["name", "hobby"],
     "我是{name}，平时喜欢{hobby}。",
     "好的{name}，爱{hobby}，记住了。"),
    (["food", "hobby"],
     "我爱吃{food}，平时喜欢{hobby}。",
     "好的，爱吃{food}、爱{hobby}，都记下了。"),
]

# 3-fact 模板
MULTI3_TEMPLATES = [
    (["name", "age", "city"],
     "我叫{name}，今年{age}岁，在{city}。",
     "好的{name}，{age}岁在{city}，都记住了。"),
    (["name", "age", "job"],
     "我是{name}，{age}岁，{job}。",
     "{name}你好，{age}岁的{job}，记下了。"),
    (["name", "city", "job"],
     "介绍一下自己：{name}，{city}人，{job}。",
     "好的{name}，{city}的{job}，记住了。"),
    (["name", "job", "hobby"],
     "我叫{name}，{job}，爱好是{hobby}。",
     "好的{name}，{job}、爱{hobby}，都记下了。"),
    (["age", "city", "hobby"],
     "我{age}岁，住在{city}，喜欢{hobby}。",
     "好的，{age}岁的{city}人，爱{hobby}，记住了。"),
    (["name", "age", "hobby"],
     "我是{name}，{age}岁，平时爱{hobby}。",
     "好的{name}，{age}岁爱{hobby}，记下了。"),
    (["name", "city", "pet"],
     "我叫{name}，在{city}，养了{pet}。",
     "好的{name}，{city}人养{pet}，记住了。"),
    (["age", "job", "hobby"],
     "我{age}岁，{job}，业余喜欢{hobby}。",
     "好的，{age}岁的{job}爱{hobby}，都记下了。"),
]


def sample_multi_reveal(rng: random.Random, persona, num_facts: int = None) -> dict:
    """从 persona 里生成一个多 fact 单 utterance turn。

    返回 dict:
        {
            "user": "我叫X，今年Y岁",
            "assistant": "好的X，Y岁，记住了",
            "slots": ["name", "age"],   # 涉及的槽（给状态机用来更新 revealed）
            "values": ["X", "Y"],       # value 字符串 list（用于多 value 权重）
        }

    如果 num_facts=None，根据 persona.unrevealed_slots() 数量随机选 2 or 3。
    """
    unrev = persona.unrevealed_slots()
    if len(unrev) < 2:
        return None  # 不够填多 fact，调用方跳过这个 turn kind

    if num_facts is None:
        num_facts = 2 if len(unrev) == 2 or rng.random() < 0.6 else 3
    num_facts = min(num_facts, len(unrev))

    # 优先挑与 unrev 匹配的模板
    candidate_templates = (
        MULTI2_TEMPLATES if num_facts == 2 else MULTI3_TEMPLATES
    )
    # 筛选模板：模板的 slots 必须全在 unrev 里
    viable = [t for t in candidate_templates if all(s in unrev for s in t[0])]

    if not viable:
        # 没有完全匹配的模板，降级为 2 fact
        if num_facts == 3:
            viable = [t for t in MULTI2_TEMPLATES if all(s in unrev for s in t[0])]
        if not viable:
            return None

    slots, user_tmpl, ack_tmpl = rng.choice(viable)
    values = [persona.slot_value(s) for s in slots]

    fill = {s: persona.slot_value(s) for s in slots}
    user_text = user_tmpl.format(**fill)
    ack_text = ack_tmpl.format(**fill)

    return {
        "user": user_text,
        "assistant": ack_text,
        "slots": slots,
        "values": values,
    }
