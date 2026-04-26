"""Stage 1 数据集生成主入口。

1A 流：DeepSeek 5-Beat 调用 + parser + validator
1B 流：world_qa 语料直接包装

流式写盘 + 断点续跑：
  - 每条 sample 完成立刻 append 写到 jsonl 并 flush，崩中断不丢已完成数据
  - 启动时检测已有文件行数，从断点续跑（仅补足差额）
  - 不再做最终 shuffle —— DataLoader 加载时 shuffle 即可
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from xinhe.data.schema import Sample, SchemaError, validate_sample


def _resolve_sampler(model: str):
    """按 model 名 dispatch：含 '/' 视为 OpenRouter（minimax/...、qwen/...、google/... 等），
    否则走 DeepSeek。返回 (call_with_retry, ApiError)，两个 sampler 接口已对齐。
    """
    if "/" in model:
        from xinhe.data.openrouter_sampler import call_with_retry as fn, OpenRouterError as Err
        return fn, Err
    from xinhe.data.deepseek_sampler import call_with_retry as fn, DeepSeekError as Err
    return fn, Err
from xinhe.data.stage1.beat_planner import BeatPlanner
from xinhe.data.stage1.mixer import sample_world_qa_pairs, wrap_world_qa_episodes
from xinhe.data.stage1.parser import parse_response, ParseError
from xinhe.data.stage1.prompts import build_messages
from xinhe.data.validator.api import validate


# v7+: 1A 在 Beat 1 之前概率插入 K 个 world_qa pair 当"开场闲聊",
# 同时解决 (1) fact 位置偏置 user[0] 93.8% (2) 1B 独立 episode padding 浪费。
# K 概率分布: 0=40% / 1=30% / 2=30% → fact 首次出现位置散到 user[0/1/2]。
_WARMUP_K_DIST = ([0, 1, 2], [0.4, 0.3, 0.3])


def _inject_warmup(sample, rng: random.Random, dict_split: str) -> None:
    """在 sample.conversations 前面插 K 对 world_qa 闲聊（标 lm_only）。
    原地修改 sample；同步更新 meta.n_turns / meta.warmup_pairs。"""
    k = rng.choices(_WARMUP_K_DIST[0], weights=_WARMUP_K_DIST[1], k=1)[0]
    if k <= 0:
        sample.meta["warmup_pairs"] = 0
        return
    pairs = sample_world_qa_pairs(k=k, rng=rng, dict_split=dict_split)
    if not pairs:
        sample.meta["warmup_pairs"] = 0
        return
    warmup = []
    for p in pairs:
        warmup.append({"role": "user", "content": p["user"]})
        warmup.append({
            "role": "assistant",
            "content": p["assistant"],
            "train_loss": "lm_only",
            "value": None,
            "value_span": [],
            "value_tier": None,
            "weight_per_span": 0.0,
        })
    sample.conversations = warmup + list(sample.conversations)
    sample.meta["n_turns"] = sample.meta.get("n_turns", 0) + len(pairs)
    sample.meta["warmup_pairs"] = len(pairs)


def _generate_one_5beat(
    rng: random.Random,
    planner: BeatPlanner,
    *,
    model: str,
    dict_split: str,
    max_retries: int = 3,
    weight_table: dict | None = None,
) -> Optional[Sample]:
    """生成一条 5-Beat 样本；失败返回 None（不抛异常）。
    validator 通过后概率插入 world_qa warmup（fact 位置打散 + episode 利用率）。"""
    call_with_retry, ApiError = _resolve_sampler(model)
    # OR 上的 reasoning model(gpt-oss / o1 / r1 等)默认全力 reason 会吃光 max_tokens
    # 导致 finish=length / content=null。显式降到 low,并加大 max_tokens 留余量。
    extra_kwargs = {}
    is_reasoning_model = any(tag in model.lower() for tag in ("oss", "o1-", "r1-", ":reasoning"))
    if "/" in model:  # OpenRouter 才支持 reasoning + freq/presence penalty
        if is_reasoning_model:
            extra_kwargs["reasoning_effort"] = "low"
        # 抑制 LLM degeneracy（同短句反复重复,实测 oss-20b 偶发）
        extra_kwargs["frequency_penalty"] = 0.5
        extra_kwargs["presence_penalty"] = 0.3
    max_tokens = 20000 if is_reasoning_model else 12000
    for attempt in range(max_retries):
        ts_attempt = time.time()
        try:
            plan = planner.plan(rng)
            sys_p, user_p = build_messages(plan)
            resp = call_with_retry(
                sys_p, user_p,
                model=model,
                temperature=0.85,
                top_p=0.92,
                max_tokens=max_tokens,
                json_mode=True,
                **extra_kwargs,
            )
            content = resp["choices"][0]["message"]["content"]
            sample = parse_response(content, plan, weight_table=weight_table, generator_model=model)
            result = validate(
                sample.to_dict(), stage="1", plan=plan.to_validator_plan()
            )
            if not result.ok:
                print(f"  [stage1] validator reject (attempt {attempt + 1}): {result.errors[:1]}", flush=True)
                continue
            # validator 通过后注入 warmup（不影响 5-Beat 主体）
            _inject_warmup(sample, rng, dict_split)
            # 单条 OK 日志 (codex 风格): n_turns/beat3_zh/recall/facts + 总耗时(含 retry)
            d = sample.to_dict()
            b3_zh = sum(
                len([c for c in t.get("content", "") if "一" <= c <= "鿿"])
                for t in d["conversations"]
                if t.get("role") == "assistant" and t.get("train_loss") == "lm_only"
            )
            facts = [(f["canonical_value"], f["scope"]) for f in d["meta"]["canonical_facts"]]
            dt = time.time() - ts_attempt
            print(
                f"  [stage1] OK ({dt:.1f}s a{attempt + 1}) n_turns={d['meta']['n_turns']} "
                f"beat3_zh={b3_zh} recall={d['meta']['recall_form']} facts={facts}",
                flush=True,
            )
            return sample
        except (ApiError, ParseError) as e:
            print(f"  [stage1] retryable fail (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        except Exception as e:
            # JSON decode / KeyError / 任意意外 → 当作 retryable
            print(f"  [stage1] unexpected fail (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
    return None


def _count_existing(path: Path) -> int:
    """数已存在 jsonl 文件的有效行数。"""
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _write_sample(fp_out, fp_rej_factory, sample: Sample) -> tuple[int, int]:
    """流式写一条 sample。返回 (ok_inc, rej_inc)。

    fp_rej_factory 是 lazy creator：只在真有 reject 时才打开 rejected 文件。
    """
    d = sample.to_dict()
    try:
        validate_sample(d)
    except SchemaError as e:
        fp_rej = fp_rej_factory()
        if fp_rej is not None:
            fp_rej.write(json.dumps({"reason": str(e), "sample": d}, ensure_ascii=False) + "\n")
            fp_rej.flush()
        return 0, 1
    fp_out.write(json.dumps(d, ensure_ascii=False) + "\n")
    fp_out.flush()
    return 1, 0


def generate_stage1_dataset(
    out_path: str | Path,
    *,
    n_samples: int,
    seed: int,
    mix: dict[str, float] | None = None,
    dict_split: str = "train",
    n_canonical_range: tuple[int, int] = (1, 3),
    n_turns_range: tuple[int, int] = (10, 14),
    beat3_min_turns: int = 1,
    beat3_min_chars: int = 500,
    beat3_chars_tolerance: float = 0.8,
    workers: int = 4,
    model: str = "deepseek-v4-flash",
    rejected_path: str | Path | None = None,
    weight_table: dict | None = None,
    resume: bool = True,
) -> tuple[int, int]:
    """生成 Stage 1 数据集（流式写 + 断点续跑）。

    mix: {"1A": 0.9, "1B": 0.1}（默认）
    1A 调 DeepSeek 5-Beat；1B 从 dicts/files/world_qa.jsonl 抽样。

    Args:
        resume: True 时若 out_path 已存在则只补足差额；False 时覆盖重写。
    """
    # v7+: 默认 100% 1A；1B 通过 1A 内嵌 warmup 实现，不再独立 episode。
    mix = mix or {"1A": 1.0}
    rng = random.Random(seed)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej = Path(rejected_path) if rejected_path else None
    if rej is not None:
        rej.parent.mkdir(parents=True, exist_ok=True)

    # ── resume 检测 ──
    if resume:
        existing = _count_existing(out)
    else:
        existing = 0
        if out.exists():
            out.unlink()
        if rej and rej.exists():
            rej.unlink()

    if existing >= n_samples:
        print(f"[stage1] 已满 ({existing} ≥ {n_samples})，跳过 {out}")
        return existing, 0

    n_remaining = n_samples - existing
    if existing > 0:
        print(f"[stage1] resume: 已有 {existing} 条，补 {n_remaining} 条到 {n_samples}")

    # 切分 1A / 1B（仅对待补足部分）；v7+ 默认 1A=1.0,1B=0
    n_1a = int(n_remaining * mix.get("1A", 1.0))
    n_1b = n_remaining - n_1a

    planner = BeatPlanner(
        dict_split=dict_split,
        n_canonical_range=n_canonical_range,
        n_turns_range=n_turns_range,
        beat3_min_turns=beat3_min_turns,
        beat3_min_chars=beat3_min_chars,
        beat3_chars_tolerance=beat3_chars_tolerance,
    )

    n_ok = existing
    n_rej = 0
    fp_out = open(out, "a", encoding="utf-8")
    # rejected 是兜底文件：parser 阶段已经把绝大多数 reject 处理了，
    # 到这层的 schema validate 失败属于 parser 漏检（应该 0 触发）。
    # 用 lazy open：只在真有 reject 时才创建文件，避免空文件污染目录。
    fp_rej = None
    rej_path = rej

    def _ensure_rej():
        nonlocal fp_rej
        if fp_rej is None and rej_path is not None:
            fp_rej = open(rej_path, "a", encoding="utf-8")
        return fp_rej

    try:
        # 1A: 多线程跑 DeepSeek，每条出炉立刻写盘
        if n_1a > 0:
            print(f"[stage1] 1A 5-Beat: {n_1a} 条 (workers={workers}, model={model})", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = []
                for _ in range(n_1a):
                    worker_seed = rng.randint(0, 2**31 - 1)
                    worker_rng = random.Random(worker_seed)
                    futures.append(ex.submit(
                        _generate_one_5beat, worker_rng, planner,
                        model=model, dict_split=dict_split,
                        weight_table=weight_table,
                    ))
                done_1a = 0
                for fut in as_completed(futures):
                    s = fut.result()
                    if s is None:
                        continue
                    ok_inc, rej_inc = _write_sample(fp_out, _ensure_rej, s)
                    n_ok += ok_inc
                    n_rej += rej_inc
                    done_1a += ok_inc
                    if done_1a > 0 and done_1a % 25 == 0:
                        print(f"  [stage1] 1A 完成 {done_1a}/{n_1a}（落盘 {n_ok}/{n_samples}）", flush=True)

        # 1B: world_qa 语料包装，流式写
        if n_1b > 0:
            print(f"[stage1] 1B world_qa: {n_1b} 条 (split={dict_split})", flush=True)
            done_1b = 0
            for s in wrap_world_qa_episodes(n_samples=n_1b, rng=rng, dict_split=dict_split):
                ok_inc, rej_inc = _write_sample(fp_out, fp_rej, s)
                n_ok += ok_inc
                n_rej += rej_inc
                done_1b += ok_inc
            if done_1b < n_1b:
                print(f"  [stage1] 1B 语料不足，仅生成 {done_1b}/{n_1b}", flush=True)
    finally:
        fp_out.close()
        if fp_rej is not None:
            fp_rej.close()

    return n_ok, n_rej
