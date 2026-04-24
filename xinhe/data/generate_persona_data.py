"""
Persona 驱动的统一对话数据生成 —— 替代 13-stage memory curriculum 的单一分布。

每个 episode = 一个随机 Persona × 12-20 turn 自然对话：
    - reveal_single / reveal_multi: 披露个人信息（单 fact 或多 fact 一句话）
    - recall: 问已披露的信息
    - refusal: 问未披露的信息 → 拒答（学"说不知道"）
    - overwrite: 纠正已披露信息 → Delta Rule 原生强项
    - third_party: 第三方人物的记忆
    - general_chat: 从 DeepSeek cache 读自然闲聊（train_loss=false）
    - world_qa: 从 cache 读事实问答（train_loss=true, 均匀权重）
    - compositional: 跨槽组合问答（需多个已披露槽）

关键设计：
    - reveal_order 控制披露顺序，永不披露的槽为 refusal 的自然来源
    - refusal 不带 VALUE 权重（避免死记硬背拒答措辞）
    - multi-fact turn 的 value 是 list[str]（conversation.py 已兼容）
    - 早期 turn 偏 reveal/chat，后期 turn 偏 recall/compositional
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from xinhe.data.generate_memory_data import (
    FACT_TEMPLATES, RECALL_TEMPLATES, OVERWRITE_TEMPLATES,
    ENTITY_FACT_TEMPLATES, ENTITY_RECALL_TEMPLATES,
    episode_to_jsonl,
)
from xinhe.data.persona import Persona, sample_persona, SLOT_NAMES
from xinhe.data.refusal_templates import sample_refusal
from xinhe.data.multi_fact_templates import sample_multi_reveal


# ─── 默认 turn mix ───
# 加起来 = 1.0
DEFAULT_TURN_MIX = {
    "general_chat":  0.34,
    "world_qa":      0.10,
    "reveal_single": 0.14,
    "reveal_multi":  0.08,
    "recall":        0.15,
    "refusal":       0.10,
    "overwrite":     0.04,
    "third_party":   0.03,
    "compositional": 0.02,
}


@dataclass
class TurnCache:
    """预加载的 DeepSeek cache，存 general_chat 和 world_qa turn 池。"""
    chat_turns: list = field(default_factory=list)
    qa_turns: list = field(default_factory=list)
    chat_idx: int = 0
    qa_idx: int = 0

    def pop_chat(self, rng: random.Random) -> Optional[dict]:
        if not self.chat_turns:
            return None
        # 随机 pop 而非顺序，增加 episode 间多样性
        return rng.choice(self.chat_turns)

    def pop_qa(self, rng: random.Random) -> Optional[dict]:
        if not self.qa_turns:
            return None
        return rng.choice(self.qa_turns)


def load_cache(cache_dir: str) -> TurnCache:
    """从 cache_dir 读 general_chat.jsonl + world_qa.jsonl。

    如果 cache 不存在，返回空 TurnCache（调用方会 fallback）。
    """
    cache = TurnCache()
    cache_path = Path(cache_dir)
    chat_file = cache_path / "general_chat.jsonl"
    qa_file = cache_path / "world_qa.jsonl"

    if chat_file.exists():
        with open(chat_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    cache.chat_turns.append(json.loads(ln))
    if qa_file.exists():
        with open(qa_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    cache.qa_turns.append(json.loads(ln))

    return cache


# ─── 单个 turn 生成函数 ───

def _reveal_single(rng, persona: Persona) -> Optional[dict]:
    """披露一个槽。使用 FACT_TEMPLATES。"""
    unrev = persona.unrevealed_slots()
    if not unrev:
        return None
    slot = unrev[0]  # 按 reveal_order 顺序
    value = persona.slot_value(slot)
    user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
    persona.revealed.add(slot)
    return {
        "user": user_tmpl.format(v=value),
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }


def _reveal_multi(rng, persona: Persona) -> Optional[dict]:
    """一句话多 fact 披露。使用 multi_fact_templates。"""
    result = sample_multi_reveal(rng, persona)
    if result is None:
        return None
    for s in result["slots"]:
        persona.revealed.add(s)
    return {
        "user": result["user"],
        "assistant": result["assistant"],
        "train_loss": True,
        "value": result["values"],     # list[str]：每个 fact 独立打 VALUE 权重
    }


def _recall(rng, persona: Persona) -> Optional[dict]:
    """对已披露的槽问答。"""
    if not persona.revealed:
        return None
    slot = rng.choice(list(persona.revealed))
    value = persona.slot_value(slot)
    user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
    return {
        "user": user_tmpl,
        "assistant": asst_tmpl.format(v=value),
        "train_loss": True,
        "value": value,
    }


def _refusal(rng, persona: Persona) -> Optional[dict]:
    """问一个未披露的槽 → 模型应拒答（无 VALUE 权重）。"""
    candidates = persona.refusal_candidates()
    if not candidates:
        return None
    slot = rng.choice(candidates)
    user, asst = sample_refusal(rng, slot)
    return {
        "user": user,
        "assistant": asst,
        "train_loss": True,
        # 不给 value！避免学死具体拒答措辞
    }


def _overwrite(rng, persona: Persona) -> Optional[dict]:
    """纠正已披露的槽。替换 persona 的值，然后用 OVERWRITE_TEMPLATES。"""
    if not persona.revealed:
        return None
    # 只挑有 OVERWRITE_TEMPLATES 的槽（name/number/city/food/job/hobby/age/pet —— 即 SLOT_NAMES 除 number 外全有，number 也有）
    slot = rng.choice(list(persona.revealed))
    if slot not in OVERWRITE_TEMPLATES:
        return None

    # 采一个新值（不同于旧值）
    from xinhe.data.persona import _sample_slot
    old = persona.slot_value(slot)
    for _ in range(5):
        new_val = _sample_slot(rng, slot)
        if new_val != old:
            break
    else:
        return None  # 采不到新值

    # 更新 persona
    setattr(persona, slot, new_val)

    user_tmpl, asst_tmpl = rng.choice(OVERWRITE_TEMPLATES[slot])
    return {
        "user": user_tmpl.format(v=new_val),
        "assistant": asst_tmpl.format(v=new_val),
        "train_loss": True,
        "value": new_val,
    }


def _third_party(rng, persona: Persona) -> Optional[dict]:
    """第三方人物的 fact 或 recall。使用 ENTITY_FACT/RECALL_TEMPLATES。"""
    if not persona.third_party:
        return None
    tp_key = rng.choice(list(persona.third_party.keys()))
    tp_data = persona.third_party[tp_key]
    if not tp_data:
        return None
    slot = rng.choice(list(tp_data.keys()))
    value = tp_data[slot]

    # 随机选 fact 披露或 recall（50/50）
    if slot not in ENTITY_FACT_TEMPLATES:
        return None

    if rng.random() < 0.5 and slot in ENTITY_FACT_TEMPLATES:
        user_tmpl, asst_tmpl = rng.choice(ENTITY_FACT_TEMPLATES[slot])
    else:
        if slot not in ENTITY_RECALL_TEMPLATES:
            return None
        user_tmpl, asst_tmpl = rng.choice(ENTITY_RECALL_TEMPLATES[slot])

    # 第三方的 {e}/{ea} 统一用 tp_key（"我朋友小明"）
    return {
        "user": user_tmpl.format(e=tp_key, ea=tp_key, v=value),
        "assistant": asst_tmpl.format(e=tp_key, ea=tp_key, v=value),
        "train_loss": True,
        "value": value,
    }


def _general_chat(rng, cache: TurnCache) -> Optional[dict]:
    """从 cache 读闲聊 turn。train_loss=false, 不参与 loss。"""
    t = cache.pop_chat(rng)
    if t is None:
        return None
    return {
        "user": t["user"],
        "assistant": t["assistant"],
        "train_loss": False,
        # 无 value
    }


def _world_qa(rng, cache: TurnCache) -> Optional[dict]:
    """从 cache 读事实 QA。train_loss=true, 无 VALUE 加权（均匀 1.0）。"""
    t = cache.pop_qa(rng)
    if t is None:
        return None
    return {
        "user": t["user"],
        "assistant": t["assistant"],
        "train_loss": True,
        # 不给 value —— 让整个 answer 均匀 1.0 权重，不过强拟合
    }


# 跨槽组合问答（简单版：如 "我这个年纪的人适合做什么"）
COMPOSITIONAL_TEMPLATES = [
    # 需要 age
    (("age",), "我这个年纪适合学习新东西吗？", "{age}岁正是学东西的好时候。"),
    (("age",), "像我这个年纪的人通常在做什么？", "{age}岁的人各有各的活法，很难一概而论。"),
    # 需要 city
    (("city",), "我在{city}的话，周末有什么好玩的推荐吗？", "{city}是个好地方，市区或郊区都能转转。"),
    # 需要 hobby + age
    (("hobby", "age"), "我这个年纪还适合{hobby}吗？", "当然可以，{age}岁玩{hobby}没问题。"),
    # 需要 job + city
    (("job", "city"), "我在{city}做{job}，发展前景如何？", "{city}的{job}机会还是不错的。"),
]


def _compositional(rng, persona: Persona) -> Optional[dict]:
    """跨槽组合 turn —— 基于已披露的多个槽生成语义问答。"""
    if len(persona.revealed) < 1:
        return None
    # 筛选 persona 已披露槽覆盖的模板
    viable = [
        t for t in COMPOSITIONAL_TEMPLATES
        if all(s in persona.revealed for s in t[0])
    ]
    if not viable:
        return None

    slots, user_tmpl, asst_tmpl = rng.choice(viable)
    fill = {s: persona.slot_value(s) for s in slots}
    # 把所有使用的 slot 值作为 value 打 weight（不重要，可以不给）
    values = list(fill.values())
    return {
        "user": user_tmpl.format(**fill),
        "assistant": asst_tmpl.format(**fill),
        "train_loss": True,
        "value": values,  # 跨槽组合用 list value
    }


# ─── 主状态机 ───

_MAKE_FNS = {
    "reveal_single": _reveal_single,
    "reveal_multi": _reveal_multi,
    "recall": _recall,
    "refusal": _refusal,
    "overwrite": _overwrite,
    "third_party": _third_party,
    "compositional": _compositional,
}


def _pick_kind(rng: random.Random, turn_mix: dict) -> str:
    """按 turn_mix 概率分布采一个 kind。"""
    r = rng.random()
    cum = 0.0
    for k, p in turn_mix.items():
        cum += p
        if r < cum:
            return k
    return list(turn_mix.keys())[-1]


def generate_stress_retention_episode(
    rng: random.Random,
    cache: TurnCache,
    min_chat_between: int = 2,
    max_chat_between: int = 5,
) -> list[dict]:
    """专门制造"retain through chat"场景的 episode 结构：

    reveal(slot1) → chat×N → recall(slot1)
    → reveal(slot2) → chat×N → recall(slot2)
    → mixed recall 检查两个槽都还在

    target: 强迫模型在生成长 chat 序列后仍能读出早先 reveal 的 slot。
    这是真实使用的核心场景；普通 turn mix 没专门强化。
    """
    persona = sample_persona(rng, num_reveal=rng.randint(2, 4))
    turns = []

    # 披露每个槽 → 穿插 chat → 召回
    for slot in persona.reveal_order:
        # 1. reveal
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl.format(v=value),
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
        persona.revealed.add(slot)

        # 2. N 轮 chat / world_qa 穿插
        n_chat = rng.randint(min_chat_between, max_chat_between)
        for _ in range(n_chat):
            # 70% general_chat, 30% world_qa
            if rng.random() < 0.7:
                t = _general_chat(rng, cache)
            else:
                t = _world_qa(rng, cache)
            if t is None:
                # cache 空 → 填一个 refusal-of-unrevealed 充数
                t = _refusal(rng, persona)
            if t is not None:
                turns.append(t)

        # 3. recall 同一个 slot（验证 retention）
        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })

    # 4. 结尾：随机 cross-recall 一次（检查多槽同时 retention）
    if len(persona.revealed) >= 2:
        random_slot = rng.choice(list(persona.revealed))
        value = persona.slot_value(random_slot)
        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[random_slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })

    return turns


def generate_multi_slot_retention_episode(
    rng: random.Random,
    cache: TurnCache,
    min_chat: int = 2,
    max_chat: int = 5,
) -> list[dict]:
    """多槽同时 retention: reveal A → reveal B (→ reveal C) → chat×N → recall 全部槽。

    覆盖 persona_stress 没训到的场景：多个事实先后告知 + 中间穿插 chat + 依次召回。
    用户实测失败：告知名字和年龄，问世界 QA，再分别召回，年龄丢了。
    """
    n_reveal = rng.randint(2, 3)
    persona = sample_persona(rng, num_reveal=n_reveal)
    turns = []

    # 1. 连续 reveal 2-3 个槽
    for slot in persona.reveal_order:
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(FACT_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl.format(v=value),
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })
        persona.revealed.add(slot)

    # 2. N 轮 chat/world_qa 穿插
    n_chat = rng.randint(min_chat, max_chat)
    for _ in range(n_chat):
        if rng.random() < 0.7:
            t = _general_chat(rng, cache)
        else:
            t = _world_qa(rng, cache)
        if t is None:
            t = _refusal(rng, persona)
        if t is not None:
            turns.append(t)

    # 3. 依次召回所有 revealed 槽（随机顺序，模拟真实查询模式）
    recall_order = list(persona.revealed)
    rng.shuffle(recall_order)
    for slot in recall_order:
        value = persona.slot_value(slot)
        user_tmpl, asst_tmpl = rng.choice(RECALL_TEMPLATES[slot])
        turns.append({
            "user": user_tmpl,
            "assistant": asst_tmpl.format(v=value),
            "train_loss": True,
            "value": value,
        })

    return turns


# ═══════════════════════════════════════════════════════════════════
# v6 双流 W_turn 专属课程 —— 4 个新 retention pattern 生成器
# ═══════════════════════════════════════════════════════════════════

from xinhe.data.turn_memory_templates import (
    REFERENCE_BACK_ALL, ENTITY_SETUP_TEMPLATES, PLACE_FILLERS,
    ENTITY_NAMES, RECALL_WITH_REF_TEMPLATES,
    SIMILAR_ENTITY_PAIRS, ANCHOR_REVEAL_TEMPLATES, ANCHOR_RECALL_TEMPLATES,
    DISTRACTOR_CHATTER_TEMPLATES,
    MINOR_DETAIL_TEMPLATES, MINOR_FILLERS, FORGET_ACKNOWLEDGMENT_TEMPLATES,
    VERBATIM_PHRASES, VERBATIM_SETUP_TEMPLATES, VERBATIM_SETUP_ACKS,
    VERBATIM_RECALL_USER_TEMPLATES,
    ADVERSARIAL_SETUP_TEMPLATES,
    ADVERSARIAL_ORDINAL_QUERY_EARLIEST,
    ADVERSARIAL_ORDINAL_QUERY_MIDDLE,
    ADVERSARIAL_ORDINAL_QUERY_LATEST,
    ADVERSARIAL_DISTANCE_QUERY_TEMPLATES,
)
from xinhe.data.generate_memory_data import FACT_TEMPLATES as _FACT_TPL
from xinhe.data.generate_memory_data import OVERWRITE_TEMPLATES as _OVERWRITE_TPL


def _distractor_chat_or_qa(rng: random.Random, cache: TurnCache, persona=None) -> Optional[dict]:
    """通用 distractor：70% chat，30% world_qa，兜底 refusal。强制 train_loss=False，
    让 distractor 只负责给 W_turn 写入"杂讯"让衰减运转，不给模型传递梯度。"""
    if rng.random() < 0.7:
        t = _general_chat(rng, cache)
    else:
        t = _world_qa(rng, cache)
    if t is None and persona is not None:
        t = _refusal(rng, persona)
    if t is not None:
        t["train_loss"] = False   # 统一关闭，不管底层生成器怎么设
    return t


def generate_variable_distance_pronoun_episode(
    rng: random.Random,
    cache: TurnCache,
    min_distance: int = 1,
    max_distance: int = 4,
) -> list[dict]:
    """变距代词消解：setup(entity) → Δτ 条 distractor → 指代召回。
    target_dtau 标注实际的"往回看几轮"距离，用于 eval_pronoun_resolution 分桶。

    结构：
      t=0 setup（train_loss=True, value=name）
        U: "昨天在三里屯吃了一家意大利菜，叫{name}"
        A: "{name}, {loc}那边..."
      t=1..Δτ distractor（train_loss=False）
      t=Δτ+1 recall（train_loss=True, value=name, target_dtau=Δτ+1）
        U: "那家店的招牌菜是什么？"
        A: "{name}的招牌菜..."
    """
    dtau = rng.randint(min_distance, max_distance)
    cat, setup_user, setup_asst = rng.choice(ENTITY_SETUP_TEMPLATES)
    name = rng.choice(ENTITY_NAMES[cat])
    loc = rng.choice(PLACE_FILLERS) if cat == "place" else ""

    # ── setup turn ──
    try:
        setup_u = setup_user.format(name=name, loc=loc)
        setup_a = setup_asst.format(name=name, loc=loc)
    except KeyError:
        setup_u = setup_user.format(name=name)
        setup_a = setup_asst.format(name=name)
    turns = [{
        "user": setup_u,
        "assistant": setup_a,
        "train_loss": True,
        "value": name,
    }]

    # ── distractor turns ──
    persona_for_fallback = sample_persona(rng, num_reveal=1)   # 兜底用
    for _ in range(dtau):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
        turns.append(t)

    # ── recall turn ──
    ref = rng.choice(REFERENCE_BACK_ALL[cat])
    recall_pairs = RECALL_WITH_REF_TEMPLATES.get(cat, RECALL_WITH_REF_TEMPLATES["place"])
    user_tmpl, asst_tmpl = rng.choice(recall_pairs)
    recall_u = user_tmpl.format(ref=ref)
    recall_a = asst_tmpl.format(name=name)
    turns.append({
        "user": recall_u,
        "assistant": recall_a,
        "train_loss": True,
        "value": name,
        # target_dtau = distractor 条数 = W_turn 里 setup-key 的 rotation 次数。
        # 写入时 W 每轮自转一次，recall 时 res_setup 已被旋转 `dtau` 次（read 发生在本轮 write 前）。
        "target_dtau": dtau,
    })

    return turns


_ALNUM_POOL = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _random_alphanumeric_phrase(rng: random.Random, min_len: int = 8, max_len: int = 12) -> str:
    """生成纯随机字母+数字字符串，LM 续句先验完全无效 —— 每 token 必须从 state 真实检索。
    使用字符间插入空格（"3 k 7 X ..."）避免 BPE 把随机串合并成少数长 token 造成监督信号集中。
    """
    length = rng.randint(min_len, max_len)
    chars = [rng.choice(_ALNUM_POOL) for _ in range(length)]
    return " ".join(chars)


def generate_meta_recall_episode(
    rng: random.Random,
    cache: TurnCache,
    min_distance: int = 0,
    max_distance: int = 2,
    total_turns: int = 4,
) -> list[dict]:
    """元认知自指召回 —— W_turn 基础能力训练（与 verbatim 互补）。

    与 verbatim 的关键差异：
      - setup 无"记住这句"指令 —— 纯自然陈述 → 纯程序性 W_turn 写入
      - recall "我刚才说了什么" —— 纯时间位置引用（非实体代词）
      - 这逼模型学会"每轮都默认入 W_turn，按 τ 相位回溯"的基本能力

    结构（固定 total_turns，用 pre-filler 补齐）：
      pre-filler × (total_turns - dtau - 2)   （train_loss=False）
      setup                                   U="<自然陈述 phrase>" A="<自然回应>"
      distractor × dtau                       （train_loss=False）
      recall                                  U="我刚才说了什么？" A="{phrase}" value=全段
    """
    from xinhe.data.turn_memory_templates import (
        VERBATIM_PHRASES, META_SETUP_TEMPLATES, META_SETUP_NATURAL_RESPONSES,
        META_RECALL_USER_TEMPLATES,
    )
    assert total_turns >= max_distance + 2, \
        f"total_turns({total_turns}) 必须 ≥ max_distance({max_distance}) + 2"
    dtau = rng.randint(min_distance, max_distance)
    n_pre = total_turns - dtau - 2
    phrase = rng.choice(VERBATIM_PHRASES)

    setup_u_tpl = rng.choice(META_SETUP_TEMPLATES)
    setup_a = rng.choice(META_SETUP_NATURAL_RESPONSES)
    recall_u = rng.choice(META_RECALL_USER_TEMPLATES)

    persona_for_fallback = sample_persona(rng, num_reveal=1)
    turns: list[dict] = []

    # Pre-filler
    for _ in range(n_pre):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
        turns.append(t)

    # Setup: 自然陈述，不训 loss（让模型不把 setup 当"显式命令"处理）
    turns.append({
        "user": setup_u_tpl.format(phrase=phrase),
        "assistant": setup_a,
        "train_loss": False,
    })

    # 后距 distractor
    for _ in range(dtau):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
        turns.append(t)

    # Recall：元指代 + 原话复述
    turns.append({
        "user": recall_u,
        "assistant": phrase,
        "train_loss": True,
        "value": phrase,
        "target_dtau": dtau,
    })

    assert len(turns) == total_turns, f"期望 {total_turns} turns，得到 {len(turns)}"
    return turns


def generate_verbatim_recall_episode(
    rng: random.Random,
    cache: TurnCache,
    min_distance: int = 0,
    max_distance: int = 2,
    total_turns: int = 4,
) -> list[dict]:
    """整段复述 —— W_turn 独门基础能力训练。

    phrase 是**运行时生成的纯随机字母+数字**序列，LM 续句先验失效 —— 每 token 都必须
    从 W_turn 真实检索。破除"per-token VALUE 高但实际没学会"的幻觉。

    W_turn 写侧 mean_t(v_t ⊗ k_t) 把整轮 token 分布压进 (d_v,d_k)。
    W_fact 的 β-selective 只挑槽位 token，装不下多 token 随机序列分布。

    结构（固定 total_turns，用 pre-filler 补齐让 batch 不损失数据）：
      pre-filler × (total_turns - dtau - 2)   （train_loss=False，在 setup 之前）
      setup                                    U="记住这句话: {rand}" A="好的记住了"
      distractor × dtau                        （train_loss=False，setup 与 recall 之间）
      recall                                   U="刚才那句话是？" A="{rand}" value=全段

    dtau ∈ [min_distance, max_distance]，默认 {0,1,2}。total_turns 必须 ≥ max_distance + 2。
    setup key 在 recall 时的旋转次数 = dtau（pre-filler 在 setup 之前发生，不改变 setup 年龄）。
    """
    assert total_turns >= max_distance + 2, \
        f"total_turns({total_turns}) 必须 ≥ max_distance({max_distance}) + 2"
    dtau = rng.randint(min_distance, max_distance)
    n_pre = total_turns - dtau - 2
    phrase = _random_alphanumeric_phrase(rng)

    setup_u_tpl = rng.choice(VERBATIM_SETUP_TEMPLATES)
    setup_a = rng.choice(VERBATIM_SETUP_ACKS)
    recall_u = rng.choice(VERBATIM_RECALL_USER_TEMPLATES)

    persona_for_fallback = sample_persona(rng, num_reveal=1)
    turns: list[dict] = []

    # ── Pre-filler: setup 之前的闲聊（train_loss=False）──
    for _ in range(n_pre):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
        turns.append(t)

    # ── Setup: 写 phrase 进 W_turn（ack 不训）──
    turns.append({
        "user": setup_u_tpl.format(phrase=phrase),
        "assistant": setup_a,
        "train_loss": False,
    })

    # ── 后距 distractor（让 setup key 旋转 dtau 次）──
    for _ in range(dtau):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "今天天气不错。", "assistant": "是啊，适合出门。", "train_loss": False}
        turns.append(t)

    # ── Recall：整段复述 ──
    turns.append({
        "user": recall_u,
        "assistant": phrase,
        "train_loss": True,
        "value": phrase,
        "target_dtau": dtau,
    })

    assert len(turns) == total_turns, f"期望 {total_turns} turns，得到 {len(turns)}"
    return turns


def generate_adversarial_temporal_episode(
    rng: random.Random,
    cache: TurnCache,
    n_entries: int = 3,
    phase_max: int = 5,
) -> list[dict]:
    """时序碰撞对抗集 —— 多条目 W_turn，按时序指定查询其中一条。

    结构（n_entries=3 固定）：
      setup_0, filler, setup_1, filler, setup_2, [trailing filler 50% 概率], recall

    dtau 分布：trailing=0 时 {4,2,0}，trailing=1 时 {5,3,1} —— 覆盖 phase_max=5 全区间。

    setup 模板**严格不含位置/序号信息**（避免 lexical 捷径），assistant ack 也中性；
    recall query 分两类：
      - ordinal (60%)：最早/中间/最后 —— 模型必须用 W_turn 相位选
      - distance (40%)：N 轮前 —— 直接指向 target τ

    选错相位 → 读出错 phrase → 每 token 惩罚（phrase 是 random alnum，LM 无法猜）。
    这是 phase_ent 真正收敛的**唯一**监督来源。
    """
    if n_entries != 3:
        raise ValueError(f"当前仅支持 n_entries=3（phase_max={phase_max} 约束）")

    phrases = [_random_alphanumeric_phrase(rng) for _ in range(n_entries)]
    setup_tpls = rng.sample(ADVERSARIAL_SETUP_TEMPLATES, n_entries)

    persona_for_fallback = sample_persona(rng, num_reveal=1)
    turns: list[dict] = []
    setup_turn_idx: list[int] = []

    for i in range(n_entries):
        setup_u_tpl, setup_a = setup_tpls[i]
        turns.append({
            "user": setup_u_tpl.format(phrase=phrases[i]),
            "assistant": setup_a,
            "train_loss": False,
        })
        setup_turn_idx.append(len(turns) - 1)

        # 相邻 setup 之间 1 条 filler（不在最后 setup 后硬加）
        if i < n_entries - 1:
            t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
            if t is None:
                t = {"user": "嗯。", "assistant": "好的。", "train_loss": False}
            turns.append(t)

    # trailing filler 让 latest 的 dtau 从 0 变成 1，增加 dtau 覆盖
    if rng.random() < 0.5:
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "嗯。", "assistant": "好的。", "train_loss": False}
        turns.append(t)

    recall_turn_idx = len(turns)
    target_i = rng.randint(0, n_entries - 1)
    target_phrase = phrases[target_i]
    # target_dtau = 旋转次数 = recall 和 setup 之间的 turn 数（与 verbatim 保持一致）
    target_dtau = recall_turn_idx - setup_turn_idx[target_i] - 1

    if target_dtau > phase_max:
        return []  # 越界 → 上层 fallback

    # 60% ordinal / 40% distance；distance 要求 dtau ≥ 1（"0 轮前"不自然）
    use_distance = (rng.random() < 0.4) and (target_dtau >= 1)
    if use_distance:
        recall_u = rng.choice(ADVERSARIAL_DISTANCE_QUERY_TEMPLATES).format(dtau=target_dtau)
    else:
        if target_i == 0:
            pool = ADVERSARIAL_ORDINAL_QUERY_EARLIEST
        elif target_i == n_entries - 1:
            pool = ADVERSARIAL_ORDINAL_QUERY_LATEST
        else:
            pool = ADVERSARIAL_ORDINAL_QUERY_MIDDLE
        recall_u = rng.choice(pool)

    turns.append({
        "user": recall_u,
        "assistant": target_phrase,
        "train_loss": True,
        "value": target_phrase,
        "target_dtau": target_dtau,
    })

    return turns


def generate_fact_vs_transient_episode(
    rng: random.Random,
    cache: TurnCache,
    num_distractors: int = 3,
) -> list[dict]:
    """Fact vs Transient：长期 anchor fact + 短期 distractor chatter + 最终召回 anchor。

    利用 VALUE_WEIGHT=5.0 的梯度非对称性：anchor 得高权，distractor 得常权。
    逼模型用 β/read 的差异把 anchor 放进 W_fact 语义锚点，distractor 留在 W_turn 时序。

    结构：
      t=0 anchor reveal（value=anchor, VALUE_WEIGHT=5x）
      t=1..N distractor（train_loss=True, value=None → 默认 1x 权重）
      t=N+1 recall anchor（value=anchor, target_dtau=N+1）
    """
    domain, anchor, distractor = rng.choice(SIMILAR_ENTITY_PAIRS)
    if domain not in ANCHOR_REVEAL_TEMPLATES or domain not in ANCHOR_RECALL_TEMPLATES:
        return []  # fallback，让上层重试
    if domain not in DISTRACTOR_CHATTER_TEMPLATES:
        return []

    reveal_u, reveal_a = rng.choice(ANCHOR_REVEAL_TEMPLATES[domain])
    turns = [{
        "user": reveal_u.format(anchor=anchor),
        "assistant": reveal_a.format(anchor=anchor),
        "train_loss": True,
        "value": anchor,  # VALUE 加权 5x
    }]

    # distractor 闲聊：陈述 distractor entity，assistant 也说 distractor 名，但 value=None
    # 使用不放回采样避免 3 条都是同一模板
    distractor_pairs = DISTRACTOR_CHATTER_TEMPLATES[domain]
    if len(distractor_pairs) >= num_distractors:
        chosen = rng.sample(distractor_pairs, num_distractors)
    else:
        chosen = [rng.choice(distractor_pairs) for _ in range(num_distractors)]
    for du, da in chosen:
        turns.append({
            "user": du.format(distractor=distractor),
            "assistant": da.format(distractor=distractor),
            "train_loss": True,    # 训练，但无 VALUE 加权
        })

    # recall anchor
    recall_u, recall_a = rng.choice(ANCHOR_RECALL_TEMPLATES[domain])
    turns.append({
        "user": recall_u,
        "assistant": recall_a.format(anchor=anchor),
        "train_loss": True,
        "value": anchor,
        # target_dtau = distractor 条数（rotation 次数），同 pronoun 逻辑
        "target_dtau": num_distractors,
    })

    return turns


def generate_rapid_overwrite_episode(
    rng: random.Random,
    cache: TurnCache,
    n_overwrites: int = 3,
    inter_chat_prob: float = 0.3,
) -> list[dict]:
    """快速覆写：同一 slot 连续改写 n 次，末轮 recall 应返回最后一个值。

    验证 Delta Rule 的 (v - W·k) 误差项能干净擦除旧值而非累积幻觉。
    强制 v0/v1/v2 前缀差异 ≥ 2 字符避免 lexical 混淆。
    """
    from xinhe.data.persona import _sample_slot, SLOT_NAMES
    # 只挑有 OVERWRITE_TEMPLATES 的槽
    candidate_slots = [s for s in SLOT_NAMES if s in _OVERWRITE_TPL and s in _FACT_TPL]
    if not candidate_slots:
        return []
    slot = rng.choice(candidate_slots)

    # 采 n+1 个显著不同的值
    values: list[str] = []
    for _ in range(n_overwrites + 1):
        for __ in range(10):
            v = _sample_slot(rng, slot)
            # 和已有值前缀差 ≥ 2 字符
            if all(v[:2] != e[:2] for e in values):
                values.append(v)
                break
        else:
            values.append(_sample_slot(rng, slot))  # 实在采不到就放一个

    persona_for_fallback = sample_persona(rng, num_reveal=1)
    turns = []

    # t=0 初始 reveal
    user_t, asst_t = rng.choice(_FACT_TPL[slot])
    turns.append({
        "user": user_t.format(v=values[0]),
        "assistant": asst_t.format(v=values[0]),
        "train_loss": True,
        "value": values[0],
    })

    # 连续 overwrite
    for i in range(1, len(values)):
        if rng.random() < inter_chat_prob:
            t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
            if t is not None:
                turns.append(t)
        ow_u, ow_a = rng.choice(_OVERWRITE_TPL[slot])
        turns.append({
            "user": ow_u.format(v=values[i]),
            "assistant": ow_a.format(v=values[i]),
            "train_loss": True,
            "value": values[i],
        })

    # recall：末轮问当前值
    final = values[-1]
    rec_u, rec_a = rng.choice(RECALL_TEMPLATES[slot])
    turns.append({
        "user": rec_u,
        "assistant": rec_a.format(v=final),
        "train_loss": True,
        "value": final,
        "target_dtau": 1,   # 最新 overwrite 是上一轮
    })

    return turns


def generate_decay_awareness_episode(
    rng: random.Random,
    cache: TurnCache,
    min_distance: int = 10,
    max_distance: int = 12,
) -> list[dict]:
    """遗忘感知：琐碎 detail 在 ~15 轮前一次性提及（train_loss=False 让其自然衰减），
    末轮问这个 detail，期望 FORGET 模板拒答。

    train 这个 pattern 能教会模型意识到"W_turn 记忆有物理极限"。
    关键：minor_detail 本身 train_loss=False，避免强记住；recall 的 value=None，避免死记拒答措辞。
    """
    n_fillers = rng.randint(min_distance, max_distance)
    tmpl_user, tmpl_asst, detail_hint = rng.choice(MINOR_DETAIL_TEMPLATES)

    # fill 模板
    fill = {}
    for key, pool in MINOR_FILLERS.items():
        fill[key] = rng.choice(pool)
    # 容错 format：有些模板只用到部分 key
    try:
        setup_u = tmpl_user.format(**fill)
        setup_a = tmpl_asst.format(**fill)
    except KeyError:
        return []

    turns = [{
        "user": setup_u,
        "assistant": setup_a,
        "train_loss": False,         # 不训 loss，让 detail 仅通过状态流动
    }]

    # 长 filler chain
    persona_for_fallback = sample_persona(rng, num_reveal=1)
    for _ in range(n_fillers):
        t = _distractor_chat_or_qa(rng, cache, persona_for_fallback)
        if t is None:
            t = {"user": "嗯嗯。", "assistant": "好的。", "train_loss": False}
        turns.append(t)

    # recall：期望 FORGET 模板拒答
    forget_reply = rng.choice(FORGET_ACKNOWLEDGMENT_TEMPLATES)
    turns.append({
        "user": f"对了，{detail_hint}？",
        "assistant": forget_reply,
        "train_loss": True,
        # value=None：不加 VALUE 权重，避免死记拒答措辞（和 _refusal 同一套逻辑）
    })

    return turns


def generate_persona_episode(
    rng: random.Random,
    cache: TurnCache,
    turn_mix: dict = None,
    min_turns: int = 12,
    max_turns: int = 20,
    max_attempts_per_turn: int = 4,
    stress_retention_ratio: float = 0.0,
    multi_slot_retention_ratio: float = 0.0,
    variable_distance_ratio: float = 0.0,
    fact_vs_transient_ratio: float = 0.0,
    rapid_overwrite_ratio: float = 0.0,
    decay_awareness_ratio: float = 0.0,
    verbatim_recall_ratio: float = 0.0,
    meta_recall_ratio: float = 0.0,
    adversarial_temporal_ratio: float = 0.0,
) -> list[dict]:
    """生成一个 persona episode。返回 turn 列表。

    结构化 retention pattern（按概率分支，命中即返回）：
      stress_retention_ratio      —— 单槽 stress retention（legacy）
      multi_slot_retention_ratio  —— 多槽 retention（legacy）
      variable_distance_ratio     —— v6 变距代词消解
      fact_vs_transient_ratio     —— v6 fact/transient 消歧
      rapid_overwrite_ratio       —— v6 快速覆写
      decay_awareness_ratio       —— v6 遗忘感知
      verbatim_recall_ratio       —— v6 整段复述（W_turn 专属）
      meta_recall_ratio           —— v6.1 元认知自指（W_turn 专属）
      adversarial_temporal_ratio  —— v6.3 时序碰撞对抗集（W_turn 相位选择专属）
    剩余概率走正常 turn mix。
    """
    if turn_mix is None:
        turn_mix = DEFAULT_TURN_MIX

    # 结构化 retention 分支（按序累加判定）
    r = rng.random()
    if multi_slot_retention_ratio > 0 and r < multi_slot_retention_ratio:
        return generate_multi_slot_retention_episode(rng, cache)
    r -= multi_slot_retention_ratio
    if stress_retention_ratio > 0 and r < stress_retention_ratio:
        return generate_stress_retention_episode(rng, cache)
    r -= stress_retention_ratio
    if variable_distance_ratio > 0 and r < variable_distance_ratio:
        ep = generate_variable_distance_pronoun_episode(rng, cache)
        if ep:
            return ep
    r -= variable_distance_ratio
    if fact_vs_transient_ratio > 0 and r < fact_vs_transient_ratio:
        ep = generate_fact_vs_transient_episode(rng, cache)
        if ep:
            return ep
    r -= fact_vs_transient_ratio
    if rapid_overwrite_ratio > 0 and r < rapid_overwrite_ratio:
        ep = generate_rapid_overwrite_episode(rng, cache)
        if ep:
            return ep
    r -= rapid_overwrite_ratio
    if decay_awareness_ratio > 0 and r < decay_awareness_ratio:
        ep = generate_decay_awareness_episode(rng, cache)
        if ep:
            return ep
    r -= decay_awareness_ratio
    if verbatim_recall_ratio > 0 and r < verbatim_recall_ratio:
        ep = generate_verbatim_recall_episode(rng, cache)
        if ep:
            return ep
    r -= verbatim_recall_ratio
    if meta_recall_ratio > 0 and r < meta_recall_ratio:
        ep = generate_meta_recall_episode(rng, cache)
        if ep:
            return ep
    r -= meta_recall_ratio
    if adversarial_temporal_ratio > 0 and r < adversarial_temporal_ratio:
        ep = generate_adversarial_temporal_episode(rng, cache)
        if ep:
            return ep

    persona = sample_persona(rng)
    n_turns = rng.randint(min_turns, max_turns)
    turns = []

    for _ in range(n_turns):
        turn = None
        for _ in range(max_attempts_per_turn):
            kind = _pick_kind(rng, turn_mix)
            if kind == "general_chat":
                turn = _general_chat(rng, cache)
            elif kind == "world_qa":
                turn = _world_qa(rng, cache)
            else:
                turn = _MAKE_FNS[kind](rng, persona)
            if turn is not None:
                break

        # Fallback chain
        if turn is None:
            turn = _general_chat(rng, cache)
        if turn is None:
            # cache 不可用，强制披露一个槽
            turn = _reveal_single(rng, persona)
        if turn is None:
            # 所有槽都披露了 → reveal_multi 或 recall
            turn = _recall(rng, persona)
        if turn is None:
            # 真没辙了（理论上不会发生，persona 总有 8 槽）
            continue

        turns.append(turn)

    return turns


# ─── 批量生成入口 ───

def generate_data(
    out_dir: str,
    num_train: int = 40000,
    num_val: int = 500,
    cache_dir: str = "data/cache",
    turn_mix: dict = None,
    min_turns: int = 12,
    max_turns: int = 20,
    seed: int = 42,
    stress_retention_ratio: float = 0.0,
    multi_slot_retention_ratio: float = 0.0,
    variable_distance_ratio: float = 0.0,
    fact_vs_transient_ratio: float = 0.0,
    rapid_overwrite_ratio: float = 0.0,
    decay_awareness_ratio: float = 0.0,
    verbatim_recall_ratio: float = 0.0,
    meta_recall_ratio: float = 0.0,
    adversarial_temporal_ratio: float = 0.0,
) -> tuple[str, str]:
    """生成训练/验证数据 JSONL。v6.3 支持 9 个 retention pattern ratio（含时序碰撞对抗集）。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_path = str(out_path / "train.jsonl")
    val_path = str(out_path / "val.jsonl")

    cache = load_cache(cache_dir)
    if not cache.chat_turns:
        print(f"  [警告] general_chat cache 为空 ({cache_dir}/general_chat.jsonl) —— 所有 chat turn 会 fallback 为 reveal/recall")
    if not cache.qa_turns:
        print(f"  [警告] world_qa cache 为空 ({cache_dir}/world_qa.jsonl)")

    rng = random.Random(seed)

    def _dump(path: str, n: int, label: str):
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(n):
                turns = generate_persona_episode(
                    rng, cache, turn_mix=turn_mix,
                    min_turns=min_turns, max_turns=max_turns,
                    stress_retention_ratio=stress_retention_ratio,
                    multi_slot_retention_ratio=multi_slot_retention_ratio,
                    variable_distance_ratio=variable_distance_ratio,
                    fact_vs_transient_ratio=fact_vs_transient_ratio,
                    rapid_overwrite_ratio=rapid_overwrite_ratio,
                    decay_awareness_ratio=decay_awareness_ratio,
                    verbatim_recall_ratio=verbatim_recall_ratio,
                    meta_recall_ratio=meta_recall_ratio,
                    adversarial_temporal_ratio=adversarial_temporal_ratio,
                )
                if turns:
                    f.write(episode_to_jsonl(turns) + "\n")
                    count += 1
        print(f"  [{label}] {count} episodes → {path}")

    _dump(train_path, num_train, "train")
    _dump(val_path, num_val, "val")

    return train_path, val_path


def generate_refusal_val_episode(rng: random.Random, cache: TurnCache,
                                  persona_turns: int = 8) -> list[dict]:
    """生成一个 val_refusal episode: 前 N-1 个 turn 披露部分槽 + 闲聊，最后一个 turn 强制为 refusal。"""
    persona = sample_persona(rng)
    turns = []

    # 披露 3-4 个槽（留至少 4 个未披露的用于最后拒答）
    n_reveal = min(rng.randint(3, 4), len(persona.reveal_order) - 2)
    for _ in range(n_reveal):
        t = _reveal_single(rng, persona)
        if t:
            turns.append(t)

    # 1-2 轮闲聊（如果 cache 有）
    for _ in range(rng.randint(1, 2)):
        t = _general_chat(rng, cache)
        if t:
            turns.append(t)

    # 最后一轮: refusal
    t = _refusal(rng, persona)
    if t:
        turns.append(t)

    return turns


def generate_compositional_val_episode(rng: random.Random, cache: TurnCache) -> list[dict]:
    """生成一个 val_compositional episode: 前几个 turn 闲聊，最后一个 turn 是多 fact 单 utterance。"""
    persona = sample_persona(rng)
    # 强制 persona 没 revealed 任何槽（multi_reveal 需要 unrevealed ≥ 2）
    turns = []

    # 1-3 轮闲聊开场
    for _ in range(rng.randint(1, 3)):
        t = _general_chat(rng, cache)
        if t:
            turns.append(t)

    # 最后一轮: multi_fact
    t = _reveal_multi(rng, persona)
    if t is None:
        # cache 也没有 chat → 插入 1 个 reveal_single 作为缓冲，然后 multi
        turns.append(_reveal_single(rng, persona) or {"user": "你好", "assistant": "你好！", "train_loss": False})
        t = _reveal_multi(rng, persona)
    if t:
        turns.append(t)

    return turns


def build_worldqa_val(cache: TurnCache, num: int, rng: random.Random) -> list[list[dict]]:
    """从 world_qa cache 采 num 条，包装成单轮 episode list。"""
    episodes = []
    available = list(cache.qa_turns)
    if not available:
        return episodes
    rng.shuffle(available)
    for t in available[:num]:
        episode = [{
            "user": t["user"],
            "assistant": t["assistant"],
            "train_loss": True,
            # 无 value: 整个 assistant 答案 1.0 权重
        }]
        episodes.append(episode)
    return episodes


def generate_pronoun_val_episode(rng, cache, distance: int = None) -> list[dict]:
    """val 专用：固定 Δτ 的变距代词消解 episode。"""
    if distance is None:
        distance = rng.randint(1, 4)
    return generate_variable_distance_pronoun_episode(
        rng, cache, min_distance=distance, max_distance=distance,
    )


def generate_disentangle_val_episode(rng, cache) -> list[dict]:
    """val 专用：fact vs transient，末轮召回 anchor。"""
    return generate_fact_vs_transient_episode(rng, cache, num_distractors=3)


def generate_rapid_overwrite_val_episode(rng, cache) -> list[dict]:
    """val 专用：3 次 overwrite 后召回最终值。"""
    return generate_rapid_overwrite_episode(rng, cache, n_overwrites=3, inter_chat_prob=0.3)


def generate_decay_val_episode(rng, cache) -> list[dict]:
    """val 专用：~15 轮前 minor detail，末轮问，期望 FORGET 拒答。"""
    return generate_decay_awareness_episode(rng, cache, min_distance=12, max_distance=14)


def generate_val_sets(
    out_dir: str = "data/val",
    cache_dir: str = "data/cache",
    n_value: int = 200,
    n_worldqa: int = 150,
    n_refusal: int = 200,
    n_compositional: int = 100,
    # v6 新增 4 个 val
    n_pronoun: int = 100,
    n_disentangle: int = 100,
    n_rapid_overwrite: int = 100,
    n_decay: int = 100,
    seed: int = 12345,
) -> dict:
    """生成 val jsonl 文件。返回 dict 映射: metric_name → val_path。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_dir)
    rng = random.Random(seed)

    paths = {}

    # val_value: 常规 persona episode，用作 VALUE/FRAME/TELL breakdown
    value_path = out_path / "val_value.jsonl"
    with open(value_path, "w", encoding="utf-8") as f:
        for _ in range(n_value):
            turns = generate_persona_episode(rng, cache)
            if turns:
                f.write(episode_to_jsonl(turns) + "\n")
    print(f"  [val] value → {value_path} ({n_value} ep)")
    paths["value"] = str(value_path)

    # val_worldqa: 单轮 QA 从 cache 抽
    wq_path = out_path / "val_worldqa.jsonl"
    wq_eps = build_worldqa_val(cache, n_worldqa, rng)
    with open(wq_path, "w", encoding="utf-8") as f:
        for ep in wq_eps:
            f.write(episode_to_jsonl(ep) + "\n")
    print(f"  [val] world_qa → {wq_path} ({len(wq_eps)} ep)")
    paths["worldqa"] = str(wq_path)

    # val_refusal: 最后一 turn 强制拒答
    rf_path = out_path / "val_refusal.jsonl"
    with open(rf_path, "w", encoding="utf-8") as f:
        count = 0
        for _ in range(n_refusal * 2):   # 允许 50% 生成失败
            if count >= n_refusal:
                break
            turns = generate_refusal_val_episode(rng, cache)
            if turns and not turns[-1].get("value"):  # 最后一轮确实是 refusal
                f.write(episode_to_jsonl(turns) + "\n")
                count += 1
    print(f"  [val] refusal → {rf_path} ({count} ep)")
    paths["refusal"] = str(rf_path)

    # val_compositional: 最后一 turn 是多 fact
    cp_path = out_path / "val_compositional.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        count = 0
        for _ in range(n_compositional * 2):
            if count >= n_compositional:
                break
            turns = generate_compositional_val_episode(rng, cache)
            if turns and isinstance(turns[-1].get("value"), list):
                f.write(episode_to_jsonl(turns) + "\n")
                count += 1
    print(f"  [val] compositional → {cp_path} ({count} ep)")
    paths["compositional"] = str(cp_path)

    # ── v6 新增 4 个 val ──────────────────────────────────────

    # val_pronoun: 变距代词消解，按 Δτ 分桶（每桶 n_pronoun/4 条）
    if n_pronoun > 0:
        pr_path = out_path / "val_pronoun.jsonl"
        with open(pr_path, "w", encoding="utf-8") as f:
            count = 0
            per_bucket = max(n_pronoun // 4, 1)
            for distance in [1, 2, 3, 4]:
                made = 0
                for _ in range(per_bucket * 3):   # 允许 2/3 失败
                    if made >= per_bucket:
                        break
                    turns = generate_pronoun_val_episode(rng, cache, distance=distance)
                    if turns and turns[-1].get("target_dtau") is not None:
                        f.write(episode_to_jsonl(turns) + "\n")
                        made += 1
                        count += 1
        print(f"  [val] pronoun → {pr_path} ({count} ep)")
        paths["pronoun"] = str(pr_path)

    # val_disentangle: fact vs transient
    if n_disentangle > 0:
        di_path = out_path / "val_disentangle.jsonl"
        with open(di_path, "w", encoding="utf-8") as f:
            count = 0
            for _ in range(n_disentangle * 2):
                if count >= n_disentangle:
                    break
                turns = generate_disentangle_val_episode(rng, cache)
                if turns and turns[-1].get("value"):
                    f.write(episode_to_jsonl(turns) + "\n")
                    count += 1
        print(f"  [val] disentangle → {di_path} ({count} ep)")
        paths["disentangle"] = str(di_path)

    # val_rapid_overwrite
    if n_rapid_overwrite > 0:
        ro_path = out_path / "val_rapid_overwrite.jsonl"
        with open(ro_path, "w", encoding="utf-8") as f:
            count = 0
            for _ in range(n_rapid_overwrite * 2):
                if count >= n_rapid_overwrite:
                    break
                turns = generate_rapid_overwrite_val_episode(rng, cache)
                if turns and turns[-1].get("value"):
                    f.write(episode_to_jsonl(turns) + "\n")
                    count += 1
        print(f"  [val] rapid_overwrite → {ro_path} ({count} ep)")
        paths["rapid_overwrite"] = str(ro_path)

    # val_decay
    if n_decay > 0:
        dc_path = out_path / "val_decay.jsonl"
        with open(dc_path, "w", encoding="utf-8") as f:
            count = 0
            for _ in range(n_decay * 2):
                if count >= n_decay:
                    break
                turns = generate_decay_val_episode(rng, cache)
                # 最后一轮 value=None（是拒答）
                if turns and turns[-1].get("train_loss") and not turns[-1].get("value"):
                    f.write(episode_to_jsonl(turns) + "\n")
                    count += 1
        print(f"  [val] decay → {dc_path} ({count} ep)")
        paths["decay"] = str(dc_path)

    return paths


def preview(n: int = 2, seed: int = 42, cache_dir: str = "data/cache"):
    """本地预览 n 个 episode（smoke test 用）。"""
    cache = load_cache(cache_dir)
    rng = random.Random(seed)
    for i in range(n):
        turns = generate_persona_episode(rng, cache)
        print(f"\n{'='*60}\n  Episode {i}  ({len(turns)} turns)\n{'='*60}")
        for j, t in enumerate(turns):
            loss = " [LOSS]" if t.get("train_loss") else ""
            val = f" val={t['value']}" if t.get("value") else ""
            print(f"  [{j+1}]{loss}{val}")
            print(f"    U: {t['user']}")
            print(f"    A: {t['assistant']}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--preview", type=int, default=2)
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    preview(args.preview, args.seed, args.cache_dir)
