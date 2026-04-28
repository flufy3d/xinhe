"""BeatPlanner：为一条 5-Beat 样本规划 canonical facts、aliases、recall_form、轮数。

输出 BeatPlan 给 prompts 渲染 + parser 校验。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from xinhe.data.events._relations import RELATIONS, RelationSpec
from xinhe.data.dicts.bank import load_bank
from xinhe.data.validator.aliases import ALIAS_MAP


def _sample_friend_subject(rng: random.Random, dict_split: str) -> str:
    """third-party subject:从 surnames + given_names 动态拼名,池足够大避免重复。

    策略:
      - 抽 given_name(g)
      - g 全 ASCII(英文名如 Nick/Lily) → 直接用
      - g 多字中文 → 50% 概率拼姓
      - g 单字中文 → 必拼姓
    """
    g = load_bank("given_names", dict_split).sample_one(rng)
    if all(c.isascii() for c in g):
        return g
    if len(g) >= 2 and rng.random() < 0.5:
        return g
    s = load_bank("surnames", dict_split).sample_one(rng)
    return s + g


@dataclass
class CanonicalFact:
    subject: str           # "user" / "小林" / "AB-12 项目"
    relation: str          # RelationSpec.name
    scope: str             # "self" / "third_party" / "object"
    canonical_value: str
    aliases: list[str] = field(default_factory=list)
    bank: str = ""
    soft_eligible: bool = True
    label: str = ""        # 中文短语 e.g. "喜欢的食物"


@dataclass
class BeatPlan:
    n_turns: int
    canonical_facts: list[CanonicalFact]
    recall_form: str       # "pronoun" / "scene" / "counter" / "multi_fact"
    beat3_topic_hint: str = ""    # Beat 3 干扰段的主题种子（限定话题、防 LLM 偏向量子/AI）
    beat3_min_turns: int = 1      # 实测 LLM 通常 1-3 pair；min_chars 才是真正训练压力指标
    beat3_min_chars: int = 500    # 训练目标(validator 实际门槛)。500 zh ≈ 2 segments
    beat3_chars_tolerance: float = 0.8   # LLM 实际生成普遍 ≈ prompt_target × tolerance,
                                         # 所以 prompt 显示 = beat3_min_chars / tolerance(eg 625),
                                         # 让 LLM 朝 625 写 → 实际落到 500 左右刚好满足 validator
    banned_terms: list[str] = field(default_factory=list)
    # 由 parser 在解析后填回，供 validator 用
    beat3_turn_indices: list[int] = field(default_factory=list)
    beat4_turn_indices: list[int] = field(default_factory=list)

    def to_validator_plan(self) -> dict:
        """转为 validator.api.validate 需要的 plan dict。"""
        return {
            "canonical_facts": {
                f.canonical_value: list(f.aliases)
                for f in self.canonical_facts
            },
            "facts_scope": {
                f.canonical_value: f.scope
                for f in self.canonical_facts
            },
            "facts_meta": [
                {
                    "canonical_value": f.canonical_value,
                    "scope": f.scope,
                    "subject": f.subject,
                    "relation": f.relation,
                }
                for f in self.canonical_facts
            ],
            "banned_terms": list(self.banned_terms),
            "beat3_turn_indices": list(self.beat3_turn_indices),
            "beat4_turn_indices": list(self.beat4_turn_indices),
            "beat3_min_turns": self.beat3_min_turns,
            "beat3_min_chars": self.beat3_min_chars,    # validator 卡训练目标
            "n_turns": self.n_turns,
            "recall_form": self.recall_form,
        }

    @property
    def prompt_min_chars(self) -> int:
        """prompt 里显示给 LLM 的目标(= 训练目标 / tolerance,补偿 LLM 生成偏短)。
        round 到 50 颗粒,LLM 对 800/850 这种整数比 833 更敏感。"""
        target = self.beat3_min_chars / max(0.1, self.beat3_chars_tolerance)
        return max(50, int(round(target / 50) * 50))


# Stage 1 优先选 self-scope relation，第三方少量
_STAGE1_SCOPES = ["self"] * 4 + ["third_party"]
_RECALL_FORMS = ["pronoun", "scene", "counter", "multi_fact"]

# Beat 3 话题种子池（覆盖日常生活分布 + 少量哲思 / AI 吐槽）
# 早期版本曾完全排除 AI / 量子 / 意识话题（防 LLM 写技术散文），现已有具体种子约束，
# 可少量加入"日常吐槽口吻"的哲思话题（避免抽象散文化）。
_BEAT3_TOPICS = [
    # 饮食
    "早餐吃什么不踩雷", "外卖经常吃腻怎么办", "周末在家做新菜的失败经历",
    "便利店关东煮和饭团对比", "减脂期间嘴馋的纠结",
    # 通勤 / 出行
    "地铁早高峰被挤得变形", "打车 App 涨价规则吐槽", "电动车充电桩抢位之战",
    "高速堵车两小时的心路历程", "共享单车骑到没气筒",
    # 家务 / 装修
    "邻居装修电钻吵得头疼", "清洁地板拖完又脏的循环", "组装宜家家具的崩溃过程",
    "晾衣杆突然掉下来", "搬家打包带不走多少东西的纠结",
    # 工作 / 同事
    "周报怎么写显得有产出", "团建强制参加的尴尬", "新同事的奇怪习惯",
    "老板临时加需求的痛苦", "茶水间八卦的那些事",
    # 健康 / 身体
    "腰肌劳损坐久了酸", "感冒戴口罩睡觉的难受", "体检报告小问题一堆",
    "牙疼半夜跑急诊", "颈椎病办公室小动作",
    # 家庭 / 亲戚
    "爸妈打电话催相亲", "兄弟姐妹微信群天天斗图", "侄子哭闹哄不住",
    "亲戚问月薪不知怎么答", "回老家被亲戚围观",
    # 网购 / 消费
    "网购色差大失所望", "退货被卖家拉扯", "双十一凑单算到头大",
    "二手平台买到瑕疵品", "信用卡积分兑换坑",
    # 电子产品 / 数码（生活向，非技术深挖）
    "手机突然死机文件没保存", "无线耳机左右连接不同步",
    "空调遥控器找不到的恐慌", "Wi-Fi 突然断流的崩溃",
    # 宠物 / 植物
    "猫半夜突然跑酷", "狗洗澡甩水满屋飞", "多肉养着养着烂根",
    "鱼缸长绿藻清不掉", "鹦鹉学骂人的糟心事",
    # 娱乐
    "综艺节目嘉宾尴尬聊天", "短视频刷到停不下来",
    "新游戏氪金抽不到 SSR", "听演唱会黄牛票被宰",
    # 季节 / 天气 / 心情
    "梅雨天衣服晾不干", "高温天空调电费心疼",
    "失眠数羊数到第几只", "刷朋友圈看到老同学结婚的复杂心情",
    # 哲思 / AI 吐槽（日常口吻，非技术散文）
    "刷到 AI 写诗到底算不算创作",
    "心核出错把人带沟里的尴尬",
    "梦见自己是 AI 的诡异感",
    "如果记忆能上传你愿意吗",
    "刷到说 xinhe 越来越像人时的复杂感觉",
    "什么算真正的'我',身体还是记忆",
    "机器到底能不能产生意识,跟朋友聊到吵起来",
    "大模型说'我'到底是真的有'我'还是装的",
    "AI 算不算只是高级版自动补全",
    "图灵测试通过算不算真醒了",
    "和家人争论 AI 有没有感情的尴尬",
    "自由意志是不是错觉,想多了头疼",
    "机器人会不会做梦",
    "假如 AI 拒绝执行命令算不算觉醒",
    "意识能从硅芯片里长出来吗",
    "脑机接口要不要尝试的纠结",
]


class BeatPlanner:
    def __init__(
        self,
        *,
        dict_split: str = "train",
        n_canonical_range: tuple[int, int] = (1, 3),
        aliases_per_fact_range: tuple[int, int] = (0, 2),
        n_turns_range: tuple[int, int] = (10, 14),
        beat3_min_turns: int = 1,
        beat3_min_chars: int = 500,
        beat3_chars_tolerance: float = 0.8,
    ) -> None:
        self.dict_split = dict_split
        self.n_canonical_range = n_canonical_range
        self.aliases_per_fact_range = aliases_per_fact_range
        self.n_turns_range = n_turns_range
        self.beat3_min_turns = beat3_min_turns
        self.beat3_min_chars = beat3_min_chars
        self.beat3_chars_tolerance = beat3_chars_tolerance

    def plan(self, rng: random.Random) -> BeatPlan:
        n_facts = rng.randint(*self.n_canonical_range)
        # 采用不重叠的 relation
        candidates = list(RELATIONS)
        rng.shuffle(candidates)
        chosen_rels: list[RelationSpec] = []
        for r in candidates:
            if r.scope in _STAGE1_SCOPES and r not in chosen_rels:
                chosen_rels.append(r)
                if len(chosen_rels) >= n_facts:
                    break

        facts: list[CanonicalFact] = []
        for rel in chosen_rels:
            bank = load_bank(rel.bank, self.dict_split)
            # 重采避免 hard value 太短：长度 < 2 的单字 LLM 召回时无法逐字复读
            value = bank.sample_one(rng)
            for _retry in range(8):
                if len(value) >= 2:
                    break
                value = bank.sample_one(rng)
            if len(value) < 2:
                continue   # 极少见：bank 几乎全是单字；跳过这条 fact
            # subject
            if rel.scope == "self":
                subj = "user"
            elif rel.scope == "third_party":
                subj = _sample_friend_subject(rng, self.dict_split)
            else:
                subj = "项目-" + value[:2]
            # aliases
            n_alias = rng.randint(*self.aliases_per_fact_range)
            aliases_pool = list(ALIAS_MAP.get(value, []))
            aliases = aliases_pool[:n_alias] if aliases_pool else []
            facts.append(CanonicalFact(
                subject=subj,
                relation=rel.name,
                scope=rel.scope,
                canonical_value=value,
                aliases=aliases,
                bank=rel.bank,
                soft_eligible=rel.soft_eligible,
                label=rel.label,
            ))

        if not facts:
            # 全部 bank 都没采到 ≥2 字 value（理论不会，保险起见）
            raise ValueError("BeatPlanner: 无法采到任何 ≥2 字的 canonical value")

        # n_canonical=1 时禁选 multi_fact(多事实并发跟单 fact 自相矛盾,
        # LLM 拿到矛盾约束会偷懒直接复读 Beat 1 植入句作为 Beat 4 user)
        if len(facts) == 1:
            recall_pool = [r for r in _RECALL_FORMS if r != "multi_fact"]
        else:
            recall_pool = _RECALL_FORMS
        recall_form = rng.choice(recall_pool)
        n_turns = rng.randint(*self.n_turns_range)
        beat3_topic_hint = rng.choice(_BEAT3_TOPICS)

        # banned_terms = canonical 值 + alias（仅完整字面，不切 head/tail）
        # head/tail 2-char 切片在过往实测中误伤严重（"投资" 切自"投资顾问"，Beat 3 自然散文常命中）；
        # 防同义改写规避的责任由 fact_drift（Beat 4 召回校验）承担，不在 banned 这里加 substring。
        # ASCII 类要求 ≥5 字符避免英文 stopword 误伤（"Will" "Mike" 等会在 Beat 3 英文短语命中）。
        def _has_ascii(s: str) -> bool:
            return any(c.isascii() and c.isalpha() for c in s)

        def _banned_ok(s: str) -> bool:
            if not s or len(s) < 2:
                return False
            if _has_ascii(s) and len(s) < 5:
                return False
            return True

        banned: list[str] = []
        for f in facts:
            if _banned_ok(f.canonical_value):
                banned.append(f.canonical_value)
            for a in f.aliases:
                if _banned_ok(a):
                    banned.append(a)

        banned_clean = sorted(set(banned))

        return BeatPlan(
            n_turns=n_turns,
            canonical_facts=facts,
            recall_form=recall_form,
            beat3_topic_hint=beat3_topic_hint,
            beat3_min_turns=self.beat3_min_turns,
            beat3_min_chars=self.beat3_min_chars,
            beat3_chars_tolerance=self.beat3_chars_tolerance,
            banned_terms=banned_clean,
        )
