"""
数据生成注册表 (v7.1)

三个注册表 + decorator 装饰器：
- TURN_KIND_FNS: 单轮 turn 生成器（签名：(rng, persona, cache?) -> turn dict | None）
- PATTERN_FNS: 完整 episode 生成器（签名：(rng, persona?, cache?, **kwargs) -> list[dict]）
- VAL_FNS: (val_generator, eval_fn) 对。val_generator 签名同 PATTERN_FNS，eval_fn 签名
  (model, tokenizer, val_path, device, seg_len, max_episodes) -> (score, episodes_done)

新增类别只需在对应模块里 @register_xxx，不用改任何 dispatch 代码。
"""
from __future__ import annotations
from typing import Callable, Optional

# ── 三个注册表 ────────────────────────────────────────────

TURN_KIND_FNS: dict[str, Callable] = {}
PATTERN_FNS: dict[str, Callable] = {}
VAL_FNS: dict[str, tuple[Callable, Optional[Callable]]] = {}


# ── Decorator 接口 ───────────────────────────────────────

def register_turn_kind(name: str):
    """注册单轮 turn 生成函数。
    签名：(rng, persona: Persona, cache: TurnCache | None = None) -> dict | None
    返回 None 表示"本次采样失败"（上层会重新选 kind）。
    """
    def _reg(fn: Callable) -> Callable:
        if name in TURN_KIND_FNS:
            raise ValueError(f"duplicate turn_kind: {name}")
        TURN_KIND_FNS[name] = fn
        return fn
    return _reg


def register_pattern(name: str):
    """注册完整 episode pattern 生成函数。
    签名：(rng, persona: Persona | None = None, cache: TurnCache | None = None, **kwargs) -> list[dict]
    返回 turn dict 的列表，对应一个 episode。
    """
    def _reg(fn: Callable) -> Callable:
        if name in PATTERN_FNS:
            raise ValueError(f"duplicate pattern: {name}")
        PATTERN_FNS[name] = fn
        return fn
    return _reg


def register_val(name: str, eval_fn: Optional[Callable] = None):
    """注册 val 集（生成器 + 可选 eval 函数）。
    生成器签名：(rng, cache, n_samples: int) -> list[list[dict]]
                （或单 episode 工厂：(rng, cache) -> list[dict]，会被 n_samples 次调用）
    eval 函数签名：(model, tokenizer, val_path: str, device, seg_len: int, max_episodes: int)
                    -> tuple[float, int]  # (score, episodes_done)
    eval_fn 为 None 时仅用于 val 生成，不纳入 joint early stop。
    """
    def _reg(gen_fn: Callable) -> Callable:
        if name in VAL_FNS:
            raise ValueError(f"duplicate val: {name}")
        VAL_FNS[name] = (gen_fn, eval_fn)
        return gen_fn
    return _reg


# ── 查询工具 ─────────────────────────────────────────────

def get_turn_kind(name: str) -> Callable:
    if name not in TURN_KIND_FNS:
        raise KeyError(f"unknown turn_kind: {name}. registered: {sorted(TURN_KIND_FNS.keys())}")
    return TURN_KIND_FNS[name]


def get_pattern(name: str) -> Callable:
    if name not in PATTERN_FNS:
        raise KeyError(f"unknown pattern: {name}. registered: {sorted(PATTERN_FNS.keys())}")
    return PATTERN_FNS[name]


def get_val(name: str) -> tuple[Callable, Optional[Callable]]:
    if name not in VAL_FNS:
        raise KeyError(f"unknown val: {name}. registered: {sorted(VAL_FNS.keys())}")
    return VAL_FNS[name]


def ensure_patterns_loaded() -> None:
    """确保所有 patterns/ 子模块被导入（触发 decorator 注册）。
    由 curriculum_data 主入口调用，用户代码不需要直接调。"""
    # 导入 patterns 子包的 __init__ 会按顺序 import 所有子模块
    from xinhe.data import patterns  # noqa: F401
