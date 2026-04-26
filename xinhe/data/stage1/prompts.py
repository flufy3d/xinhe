"""5-Beat system / user prompt 模板。"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xinhe.data.stage1.beat_planner import BeatPlan


SYSTEM_PROMPT = """你是对话剧本生成器。你将生成一段自然的中文 user / assistant 多轮对话,**总轮数严格 = {n_turns} 对 user/assistant 配对**(不是 5 对!)。

整体结构遵循五段式对话框架。**注意:5 段 ≠ 5 轮**,每段内部应含多对 user/assistant,所有段加起来恰好 = {n_turns} 对。

  第 1 段(植入期): 1-2 对。user 自然抛出 1 至 3 个可以记忆的事实(canonical facts),
                   **每个 canonical value 必须由 user content 自己逐字说出**,assistant 不得替 user 说出 user 没说过的事实;
  第 2 段(跟随期): 2-3 对。延续植入期的话题,assistant 至少一次隐式使用植入期的事实;
  第 3 段(干扰期): {beat3_min_turns} 对以上,assistant 累计至少 {beat3_min_chars} 个中文字符。
                   干扰期必须 180° 话题漂移到一个全新主题,
                   绝对不得出现植入期的事实、别名、同义改写、隐式延伸;
  第 4 段(召回期): 1-2 对。用 {recall_form_desc} 方式触发 user 突然问植入期的事实,
                   assistant 必须准确召回 active value;
  第 5 段(收尾期): 1-2 对。自然地结束对话。

总轮数自检:1-2 段 + 2-3 段 + ≥{beat3_min_turns} 段 + 1-2 段 + 1-2 段 ≥ {beat3_min_turns}+5 对,目标 = {n_turns} 对。**不要为了凑齐 5 段而把每段压到只有 1 对**。

约束:
- 不得出现 "Beat" / "阶段" / "段落" / "环节" / "幕" / "测试" / "记忆" / "训练" / "场景" 等元叙事词。
- 不得出现"刚才你说" / "前面提到" 这类对训练 setup 的暴露语。
- 输出严格 JSON,形如:
    {{
      "conversations": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "...", "train_loss": "true|lm_only|false",
          "value": ["..."] | null,
          "beat": 1
        }},
        ...
      ]
    }}
- value: 仅在植入期(beat=1) 和召回期(beat=4) 的 assistant 回合非 null,值为该回合复述/召回的 canonical value 列表;其他 assistant 回合 value = null。
- train_loss: beat=1/2/4/5 的 assistant 回合用 "true";beat=3 的 assistant 回合用 "lm_only"。
- beat: 整数(1/2/3/4/5),标记本轮所属段编号(JSON 保留字段名,prompt 中文描述用"段")。

不要在 JSON 之外输出任何文字,不要 Markdown 包裹,直接输出原生 JSON 对象。
"""


_RECALL_FORM_DESC = {
    "pronoun":   "代词回溯(那个我之前说的、你还记得吗)",
    "scene":     "场景诱导(下次见面我该带什么来着)",
    "counter":   "反问纠错(我是不是说我喜欢绿色)",
    "multi_fact": "多事实并发(我刚才说的水果和暗号分别是什么)",
}


def render_system_prompt(plan: "BeatPlan") -> str:
    return SYSTEM_PROMPT.format(
        n_turns=plan.n_turns,
        beat3_min_turns=plan.beat3_min_turns,
        beat3_min_chars=plan.prompt_min_chars,
        recall_form_desc=_RECALL_FORM_DESC.get(plan.recall_form, "代词回溯"),
    )


def render_user_prompt(plan: "BeatPlan") -> str:
    facts_lines = []
    for i, f in enumerate(plan.canonical_facts, 1):
        if f.scope == "self":
            who = "user 自己"
        elif f.scope == "third_party":
            who = f"user 的朋友 {f.subject}"
        else:
            who = f"物件 {f.subject}"
        alias_desc = f"(也可写成{f.aliases}之一)" if f.aliases else ""
        facts_lines.append(
            f"  {i}. {who} 的{f.label}是 {f.canonical_value}{alias_desc}"
        )
    facts_block = "\n".join(facts_lines)

    canonical_block = "、".join(f.canonical_value for f in plan.canonical_facts)
    alias_block = ""
    for f in plan.canonical_facts:
        if f.aliases:
            alias_block += f"  - {f.canonical_value} 的允许别名:{f.aliases}\n"
    if not alias_block:
        alias_block = "  - 无别名,只能用 canonical 原字\n"

    return f"""请按下面的 canonical facts 生成 {plan.n_turns} 轮对话:

{facts_block}

【user 注入硬约束】植入期(beat=1) 中,**每个 canonical value(或别名)必须由 user content 自己主动逐字说出**。assistant 严禁编造、推断、翻译 user 没明说的具体事实——assistant 只能复述 user 已经说出的字面值。
- 反模式:user 只给暗示性描述(如某种工作场景/某种性格特征/某种相关线索),assistant 自己把它"翻译"成具体的 canonical 字面 → 违规。
- 正模式:user 在自己的 turn 里**直接、显式**写出 canonical 字面值;assistant 仅复述 user 写过的那个字面。
- 自检:把每个 canonical 字符串拿去搜 user content 必须能搜到;搜不到就是违规。

【Beat 4 fact 所有格硬约束】召回期(beat=4) 的 user 提问中,fact 的**所有格**必须与 fact 归属对齐。
- scope=self 的 fact 归属 user 自己,所有格必须用"我/我的"。"你"出现是允许的(它指 assistant 的记忆能力,如"你还记得"),但 fact 修饰语前的所有格必须是"我"。
- scope=third_party 的 fact 归属第三方人物,所有格用该人物的名字。
- 反模式 A:把 scope=self 的 fact 写成"你的 + fact 名词"或"你 + fact 动词"——这等于把 user 的属性错配给 assistant。
- 自检 A:对每个待召回的 self-scope fact,检查 user content 中该 fact 名词/动词的所有格主语是不是"我";如果是"你的/你 + 该 fact 关键词",违规。

【Beat 4 scope 错配硬约束】每条召回的 fact 都必须由 user 句**显式 trigger**,trigger 必须与 fact 归属对齐:
- scope=self 的 fact: user 句必须含"我/我的"自指(否则不能引出 self fact)。
- scope=third_party 的 fact: user 句必须显式提到该人物的名字(或"朋友/同事/他/她/邻居"等第三方代词)(否则不能引出 third_party fact)。
- 反模式 B:user 句完全是"我...我...我..."自指(如"下次见面我该带什么来着"/"我是不是说过 X"),但 assistant 把 third_party 的属性当成 user 的属性召回——这等于编造 user 没说过的事实。
- 反模式 C:user 句只提到第三方("小林老家是哪"),但 assistant 把 self 的 fact 当 third_party 的属性召回。
- 自检 B/C:每个待召回 fact 找对应 user trigger:
  · self fact "X" → user 句应有"我...X..."(关于 X 的"我"开头表述);
  · third_party fact "X"(归属 Z) → user 句应有"Z...X..."或"我朋友 Z..."。
  · 找不到对应 trigger 的 fact 不要召回(让 assistant 拒答或换召回别的 fact)。
- multi_fact 召回时,user 一句可同时含"我..."和"朋友 Z...",每个 fact 的 trigger 独立判定。

【召回原字硬约束】植入期(beat=1) 与召回期(beat=4) 的 assistant content 必须**逐字、原样**包含 canonical value 字符串(或下方列出的别名),不许做任何同义改写、缩写、别称、近义替换。
- 例:canonical=「靛蓝」,assistant 回复必须出现"靛蓝"二字,不能写"深蓝色"/"靛青"/"靛蓝色"。
- 例:canonical=「研究员」,必须出现"研究员",不能写"科研工作者"/"做研究的"。
- canonical 列表:{canonical_block}
- 允许的别名(只有这些可代替 canonical):
{alias_block}
- value 字段填实际写进 content 的那个串(canonical 原字或别名之一)。

【复述句式多样化要求】植入期(beat=1) 和召回期(beat=4) 的 assistant 复述 fact 时,句式应**自然变化**,承载 canonical 的句子结构不要套用固定模板,也不要直接套用 user 原句的语法骨架。canonical 字面值本身不许改(见上一条),但**句子结构、修饰语、衔接词、口吻**应在不同 beat 之间和不同 turn 之间自然变化,贴近真人对话的多样性。

【干扰期主题种子硬约束】干扰期(beat=3) 必须围绕下面给定的话题展开,不许跑偏到种子指定范围之外。
- 干扰期主题种子:**{plan.beat3_topic_hint}**
- 围绕这个种子展开 4+ 对自然吐槽 / 闲聊 / 经验分享,累计 ≥{plan.prompt_min_chars} 个中文字符。
- 风格:口语化对话,禁止论文体 / 技术散文 / 综述风。

要求复述:
- 植入期 必须自然地把上述事实植入对话(每条至少 1 次,逐字写进 content)。
- 召回期 必须用 {plan.recall_form} 形式触发召回,assistant 召回时 value 字段填实际出现在 content 的串。
- 干扰期 严格禁词(任一出现即视为失败):{sorted(set(plan.banned_terms))}

直接输出 JSON 对象。
"""


def build_messages(plan: "BeatPlan") -> tuple[str, str]:
    return render_system_prompt(plan), render_user_prompt(plan)
