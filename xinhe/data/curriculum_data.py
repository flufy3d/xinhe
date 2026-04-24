"""
Curriculum Data (v7.1) —— 统一数据生成主入口

替代旧的 `type: memory` / `type: persona` 分发逻辑。所有 turn_kinds 和 patterns 通过
registry 注册后统一生成；stage config 用 declarative spec 描述要什么混合分布。

入口函数：
  generate_episode(rng, stage_cfg, cache, persona=None) -> list[turn dict]
  generate_data(stage_cfg, out_dir) -> (train_path, val_path)
  generate_val_sets(out_dir, cache_dir, **n_per_val) -> dict[name, path]
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from xinhe.data.persona import sample_persona, Persona
from xinhe.data.samplers import episode_to_jsonl
from xinhe.data.registry import TURN_KIND_FNS, PATTERN_FNS, VAL_FNS, ensure_patterns_loaded


# 触发 patterns 注册
ensure_patterns_loaded()


@dataclass
class TurnCache:
    """预加载的 DeepSeek cache。"""
    chat_turns: list = field(default_factory=list)
    qa_turns: list = field(default_factory=list)

    def pop_chat(self, rng: random.Random) -> Optional[dict]:
        if not self.chat_turns:
            return None
        return rng.choice(self.chat_turns)

    def pop_qa(self, rng: random.Random) -> Optional[dict]:
        if not self.qa_turns:
            return None
        return rng.choice(self.qa_turns)


def load_cache(cache_dir: str) -> TurnCache:
    """从 cache_dir 读 general_chat.jsonl + world_qa.jsonl。"""
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


# ── 生成主入口 ─────────────────────────────────────────────

def _pick_key(rng: random.Random, mix: dict) -> Optional[str]:
    """按概率分布从 mix 里挑一个 key。空 dict → None。"""
    if not mix:
        return None
    total = sum(mix.values())
    if total <= 0:
        return None
    r = rng.random() * total
    cum = 0.0
    for k, p in mix.items():
        cum += p
        if r < cum:
            return k
    return list(mix.keys())[-1]


def generate_episode(
    rng: random.Random,
    stage_cfg: dict,
    cache: TurnCache,
    persona: Optional[Persona] = None,
) -> list[dict]:
    """根据 stage_cfg 生成一个 episode（list of turn dicts）。

    stage_cfg 字段：
      turn_kinds: dict[str, float]     —— 单轮 kind 概率混合
      patterns: dict[str, float]       —— 结构化 pattern 概率混合（互斥：选中 pattern 后整个 episode 都是该 pattern）
      min_turns / max_turns: int       —— 总轮数范围（仅当使用 turn_kinds 时生效）
      num_reveal: int? = None          —— persona 的 reveal_order 长度（默认 4-6）
    """
    turn_kinds = stage_cfg.get("turn_kinds", {}) or {}
    patterns = stage_cfg.get("patterns", {}) or {}

    # 先决定本 episode 走 pattern 还是 turn_kinds 混合
    # 总和 < 1 则有概率走 turn_kinds（剩余）
    pattern_prob = sum(patterns.values())
    use_pattern = rng.random() < pattern_prob
    if use_pattern and patterns:
        pattern_name = _pick_key(rng, patterns)
        if pattern_name and pattern_name in PATTERN_FNS:
            ep = PATTERN_FNS[pattern_name](rng, persona, cache)
            if ep:
                return ep
        # pattern 生成失败则 fallback 到 turn_kinds

    # turn_kinds 混合
    if persona is None:
        num_reveal = stage_cfg.get("num_reveal")
        persona = sample_persona(rng, num_reveal=num_reveal)

    min_turns = stage_cfg.get("min_turns", 10)
    max_turns = stage_cfg.get("max_turns", 16)
    n_turns = rng.randint(min_turns, max_turns)

    turns = []
    for _ in range(n_turns):
        kind = _pick_key(rng, turn_kinds)
        if kind is None or kind not in TURN_KIND_FNS:
            continue
        t = TURN_KIND_FNS[kind](rng, persona, cache)
        if t is not None:
            turns.append(t)
    return turns


# ── 批量生成 train/val 文件 ───────────────────────────────

def generate_data(
    stage_cfg: dict,
    stage_name: str,
    out_dir: str = "data/curriculum",
    cache_dir: str = "data/cache",
    seed_offset: int = 0,
) -> tuple[str, str]:
    """为某个课程阶段生成 train.jsonl + val.jsonl。

    返回 (train_path, val_path)。"""
    data_cfg = stage_cfg  # stage_cfg 是 stage['data'] dict
    num_train = data_cfg.get("num_train") or data_cfg.get("num_episodes", 10000)
    num_val = data_cfg.get("num_val", 200)
    seed = data_cfg.get("seed", 42) + seed_offset

    cache = load_cache(data_cfg.get("cache_dir", cache_dir))
    out_stage = Path(out_dir) / stage_name
    out_stage.mkdir(parents=True, exist_ok=True)

    rng_train = random.Random(seed)
    rng_val = random.Random(seed + 1)

    train_path = out_stage / "train.jsonl"
    val_path = out_stage / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        count = 0
        attempts = 0
        while count < num_train and attempts < num_train * 4:
            attempts += 1
            ep = generate_episode(rng_train, data_cfg, cache)
            if ep:
                f.write(episode_to_jsonl(ep) + "\n")
                count += 1
        print(f"  [train] {count} episodes → {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        count = 0
        attempts = 0
        while count < num_val and attempts < num_val * 4:
            attempts += 1
            ep = generate_episode(rng_val, data_cfg, cache)
            if ep:
                f.write(episode_to_jsonl(ep) + "\n")
                count += 1
        print(f"  [val] {count} episodes → {val_path}")

    return str(train_path), str(val_path)


# ── Val 集生成（每类别独立）─────────────────────────────────

DEFAULT_VAL_SIZES = {
    "value": 200,
    "worldqa": 150,
    "refusal": 200,
    "compositional": 100,
    "rapid_overwrite": 100,
    "verbatim": 100,
    "reference_back": 100,
    "context_followup": 100,
    "topic_continuation": 80,
    "entity_tracking": 100,
    "irrelevant_forget": 100,
    "multi_slot_retention": 80,
}


def generate_val_sets(
    out_dir: str = "data/val",
    cache_dir: str = "data/cache",
    sizes: Optional[dict] = None,
    seed: int = 12345,
) -> dict[str, str]:
    """为每个注册的 val 集生成 jsonl。返回 name → path。"""
    sizes = sizes or DEFAULT_VAL_SIZES
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_dir)

    paths = {}
    rng = random.Random(seed)

    # 先生成 value / worldqa / refusal / compositional 这 4 个特殊集（沿用 persona dist）
    # value: 通用 persona episode
    from xinhe.data.patterns.continuity import generate_reference_back_episode  # noqa

    # value — 用默认 Stage 1 mix 生成
    n_value = sizes.get("value", 200)
    value_path = out_path / "val_value.jsonl"
    with open(value_path, "w", encoding="utf-8") as f:
        cfg = {
            "turn_kinds": {"reveal_single": 0.3, "recall": 0.3, "general_chat": 0.2,
                           "refusal": 0.1, "compositional": 0.1},
            "patterns": {},
            "min_turns": 8, "max_turns": 14,
        }
        count = 0
        attempts = 0
        while count < n_value and attempts < n_value * 4:
            attempts += 1
            ep = generate_episode(rng, cfg, cache)
            if ep:
                f.write(episode_to_jsonl(ep) + "\n")
                count += 1
    paths["value"] = str(value_path)
    print(f"  [val] value → {value_path} ({count} ep)")

    # worldqa: 纯 world_qa turn
    n_wq = sizes.get("worldqa", 150)
    wq_path = out_path / "val_worldqa.jsonl"
    with open(wq_path, "w", encoding="utf-8") as f:
        count = 0
        from xinhe.data.registry import get_turn_kind
        world_qa = get_turn_kind("world_qa")
        for _ in range(n_wq * 2):
            if count >= n_wq:
                break
            t = world_qa(rng, None, cache)
            if t is not None:
                ep = [t]
                f.write(episode_to_jsonl(ep) + "\n")
                count += 1
    paths["worldqa"] = str(wq_path)
    print(f"  [val] worldqa → {wq_path} ({count} ep)")

    # refusal: 末轮强制 refusal
    n_rf = sizes.get("refusal", 200)
    rf_path = out_path / "val_refusal.jsonl"
    with open(rf_path, "w", encoding="utf-8") as f:
        from xinhe.data.registry import get_turn_kind
        reveal_single = get_turn_kind("reveal_single")
        refusal_fn = get_turn_kind("refusal")
        count = 0
        for _ in range(n_rf * 2):
            if count >= n_rf:
                break
            persona = sample_persona(rng, num_reveal=rng.randint(2, 4))
            # 披露 1-2 个槽
            turns = []
            for _ in range(rng.randint(1, 2)):
                t = reveal_single(rng, persona, cache)
                if t:
                    turns.append(t)
            # 末轮 refusal
            ref = refusal_fn(rng, persona, cache)
            if ref:
                turns.append(ref)
                f.write(episode_to_jsonl(turns) + "\n")
                count += 1
    paths["refusal"] = str(rf_path)
    print(f"  [val] refusal → {rf_path} ({count} ep)")

    # compositional: 末轮强制 compositional
    n_cp = sizes.get("compositional", 100)
    cp_path = out_path / "val_compositional.jsonl"
    with open(cp_path, "w", encoding="utf-8") as f:
        from xinhe.data.registry import get_turn_kind
        reveal_single = get_turn_kind("reveal_single")
        reveal_multi = get_turn_kind("reveal_multi")
        compositional_fn = get_turn_kind("compositional")
        count = 0
        for _ in range(n_cp * 3):
            if count >= n_cp:
                break
            persona = sample_persona(rng, num_reveal=5)
            turns = []
            # 披露多个槽以满足 compositional 需求
            for _ in range(3):
                t = reveal_single(rng, persona, cache) or reveal_multi(rng, persona, cache)
                if t:
                    turns.append(t)
            cp = compositional_fn(rng, persona, cache)
            if cp and cp.get("value"):
                turns.append(cp)
                f.write(episode_to_jsonl(turns) + "\n")
                count += 1
    paths["compositional"] = str(cp_path)
    print(f"  [val] compositional → {cp_path} ({count} ep)")

    # 其他 val（通过 registry）
    for name, (gen_fn, _eval_fn) in VAL_FNS.items():
        if name in paths:
            continue   # 已手工处理
        n = sizes.get(name, 80)
        p = out_path / f"val_{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            episodes = gen_fn(rng, cache, n)
            count = 0
            for ep in episodes:
                f.write(episode_to_jsonl(ep) + "\n")
                count += 1
        paths[name] = str(p)
        print(f"  [val] {name} → {p} ({count} ep)")

    return paths
