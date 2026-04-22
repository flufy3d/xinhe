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


def generate_persona_episode(
    rng: random.Random,
    cache: TurnCache,
    turn_mix: dict = None,
    min_turns: int = 12,
    max_turns: int = 20,
    max_attempts_per_turn: int = 4,
    stress_retention_ratio: float = 0.0,
    multi_slot_retention_ratio: float = 0.0,
) -> list[dict]:
    """生成一个 persona episode。返回 turn 列表。

    stress_retention_ratio: 概率产出"单槽 stress retention"（单 reveal → N chat → recall）
    multi_slot_retention_ratio: 概率产出"多槽 retention"（reveal A/B/C → N chat → 依次召回）
    剩余概率走正常 turn mix。

    失败回退链: 某 kind 的 make 函数返回 None (比如没 revealed 就不能 recall)
    → 重新采 kind → 最多 max_attempts_per_turn 次 → 最后 fallback 到 general_chat
    → 如果 cache 也空 → fallback 到 reveal_single
    """
    if turn_mix is None:
        turn_mix = DEFAULT_TURN_MIX

    # 多槽 retention 结构化（用户实测失败的场景）
    r = rng.random()
    if multi_slot_retention_ratio > 0 and r < multi_slot_retention_ratio:
        return generate_multi_slot_retention_episode(rng, cache)
    r -= multi_slot_retention_ratio
    # 单槽 stress retention
    if stress_retention_ratio > 0 and r < stress_retention_ratio:
        return generate_stress_retention_episode(rng, cache)

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
) -> tuple[str, str]:
    """生成训练/验证数据 JSONL。

    stress_retention_ratio: 此比例的 episode 走"reveal→chat→recall"结构化
    pattern（强化 retention through chat 的信号）。默认 0（纯 turn mix）。

    返回: (train_path, val_path)
    """
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


def generate_val_sets(
    out_dir: str = "data/val",
    cache_dir: str = "data/cache",
    n_value: int = 200,
    n_worldqa: int = 150,
    n_refusal: int = 200,
    n_compositional: int = 100,
    seed: int = 12345,
) -> dict:
    """生成 4 个 val jsonl 文件。

    返回 dict 映射: metric_name → val_path
    """
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
