"""Stage 1 数据集生成主入口。

1A 流：DeepSeek 5-Beat 调用 + parser + validator
1B 流：world_qa 语料直接包装

流式写盘 + 断点续跑：
  - 每条 sample 完成立刻 append 写到 jsonl 并 flush，崩中断不丢已完成数据
  - 启动时检测已有文件行数，从断点续跑（仅补足差额）
  - 不再做最终 shuffle —— DataLoader 加载时 shuffle 即可
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from xinhe.data.schema import Sample, SchemaError, validate_sample


def _lock_path_for(out_path: Path) -> Path:
    """按 out_path 绝对路径 hash 在系统 temp 目录派生锁文件,保持数据目录干净。
    不同 out_path 派生不同 hash → 不同 jsonl 不会互相干扰。"""
    key = hashlib.md5(str(out_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"xinhe_genlock_{key}.lock"


@contextlib.contextmanager
def _single_instance_lock(out_path: Path):
    """OS 级文件锁,同一 out_path 同时只允许一个 generator 进程。
    阻止"重启时漏杀 orphan python.exe → 双写同一 jsonl"的血案。

    锁文件放系统 temp 目录(数据目录不留垃圾),OS 自动管理:进程崩/被杀也会立刻释放。
    Windows: msvcrt.locking;Linux/macOS: fcntl.flock。两者都用 NB(非阻塞),
    抢不到立刻 raise(明确告诉用户已有进程在跑,不傻等)。
    """
    lock_path = _lock_path_for(out_path)
    fp = open(lock_path, "w", encoding="utf-8")
    try:
        msg = (
            f"!!! 已有 generator 进程在写 {out_path}\n"
            f"  锁文件: {lock_path}\n"
            f"  本进程立刻 abort,避免双写损失 token。\n"
            f"  确认对方真死了的话(Get-CimInstance Win32_Process | grep generate_data 确认无 python.exe 残留):\n"
            f"    Remove-Item '{lock_path}'  # 然后重试"
        )
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                fp.close()
                # from None 抑制 PermissionError 原始链,用户只看到友好提示
                raise RuntimeError(msg) from None
        else:
            import fcntl
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fp.close()
                raise RuntimeError(msg) from None
        fp.write(f"pid={os.getpid()}\nout={out_path}\n")
        fp.flush()
        yield
    finally:
        try:
            fp.close()
        except Exception:
            pass
        try:
            lock_path.unlink()
        except Exception:
            pass


def _is_quota_exhausted(e: BaseException) -> bool:
    """识别 sampler 抛出的 quota 耗尽错误。约定:类名以 QuotaExhaustedError 结尾的子类视为 quota 信号。
    用名字匹配避免在 driver 里跨家 import 各 sampler 的 Error 类。"""
    return type(e).__name__.endswith("QuotaExhaustedError")


def _supports_conditional_retry(model: str) -> bool:
    """走 conditional retry(多轮 messages + 主流 provider prefix cache 命中):
      - deepseek-* / 不带 / 的 OpenAI 兼容 model
      - openrouter (含 /,主流 provider 都支持 multi-turn,免费 provider 也能用,只是不享 cache)
    不支持:codex-cli / gemini-cli (subprocess 单次调用,不方便追加上下文)。"""
    head = model.split(":", 1)[0].lower()
    if head in ("codex-cli", "codex", "gemini-cli", "gemini"):
        return False
    return True


def _build_fix_message(reason: str, plan) -> str:
    """把 validator/parser reject reason 翻译成给 LLM 的修补 prompt。
    fix prompt 短促、具体、不发散——告诉模型哪里错了 + 怎么改 + 保留其他段。
    """
    if "beat3 struct" in reason and "zh_chars" in reason:
        m = re.search(r"zh_chars=(\d+)", reason)
        actual = m.group(1) if m else "?"
        target = plan.prompt_min_chars
        return (
            f"刚才输出的 Beat 3（干扰段）assistant 中文累计只写了 {actual} 字，目标 ≥{target} 字。\n"
            f"请重新输出**完整 JSON**，把 Beat 3 段在保持原话题种子和禁词约束的前提下扩写到 ≥{target} 字。\n"
            f"具体方法：增加 1-3 对 user/assistant，围绕原 Beat 3 主题种子继续展开吐槽/细节/举例/共鸣，"
            f"不要换主题、不要拉回 Beat 1 的 fact 话题。\n"
            f"其他段（植入/跟随/召回/收尾）尽量保持原内容，总轮数仍 = {plan.n_turns}。"
        )
    if "beat3 purity" in reason or "泄漏" in reason:
        m = re.search(r"泄漏=\[(.*?)\]", reason)
        leaked = m.group(1) if m else "(列表见上)"
        return (
            f"刚才输出的 Beat 3 段出现了禁词：{leaked}。\n"
            f"请重新输出**完整 JSON**，Beat 3 段绝对不要出现 canonical/alias 列表中的任意词及其变体、"
            f"同义改写、隐式延伸（哪怕是负面提及也不行）。其他段保持原样，总轮数仍 = {plan.n_turns}。"
        )
    if "beat3 repetition" in reason:
        return (
            f"刚才 Beat 3 段 assistant 出现了短句重复（degenerate）。\n"
            f"请重新输出**完整 JSON**，Beat 3 段每句话都要不同，自然展开话题，避免回环复读。"
            f"总轮数仍 = {plan.n_turns}。"
        )
    if "beat4 scope" in reason:
        return (
            f"刚才输出的 Beat 4（召回段）user 句不是真正的召回提问，或人称代词与 fact 归属错配。\n"
            f"硬约束（Beat 4 必须由 user 发起召回提问，asst 才召回）：\n"
            f"  - **必须是 user 提问 / 反问 / 求确认的句式**（不许 user 自己讲出答案让 asst echo）；\n"
            f"  - scope=self 的 fact，user 句必须用'我/我的'自指（如'我之前说过 X 是什么'/'我家是哪'/'我是不是说过 X'）；\n"
            f"  - scope=third_party 的 fact，user 句必须显式提该人物名字（或'朋友/同事/他/她'等第三方代词）。\n"
            f"反模式：user 在 Beat 4 直接说出 canonical（如'回去得去无极看看'）→ asst 召回成了纯 echo，无训练价值。\n"
            f"请重新输出**完整 JSON**，修正 Beat 4 user 提问的句式 + 人称指代，其他段保持原样。"
        )
    if "beat4 pronoun" in reason:
        return (
            "刚才输出的 Beat 4 中，user 句把 self-scope 的 fact 写成了'你的+fact'（指 assistant），"
            "这是把 user 的属性错配给了 assistant。\n"
            "请重新输出**完整 JSON**，user 句中 fact 的所有格必须用'我/我的'。其他段保持原样。"
        )
    if "fact drift" in reason:
        m = re.search(r"\[(.*?)\]", reason)
        drift = m.group(1) if m else ""
        return (
            f"刚才输出的 Beat 4 召回 value（{drift}）无法映射到 canonical/alias 表。\n"
            f"请重新输出**完整 JSON**，Beat 4 assistant 召回时 content 必须**逐字**包含 canonical 字面值或允许的别名，"
            f"value 字段填实际写进 content 的那个串。"
        )
    if "counter_anti_echo" in reason:
        m = re.search(r"\[(.*?)\]", reason)
        leaked = m.group(1) if m else "(列表见上)"
        return (
            f"刚才 Beat 4（反问纠错段）assistant 顺着 user 故意说错的实体回应了：{leaked}。\n"
            f"counter form 的语义是 user 故意把 fact 说错（如把颜色/食物/城市等张冠李戴），"
            f"assistant **必须明确否认**该错误（'我没记得你提过 X'/'你说的 X 我不记得'），"
            f"只确认 plan 内 canonical 的真实 fact，**绝不许 echo** {leaked} 这种 plan 外的实体（哪怕是修饰语/铺垫语）。\n"
            f"请重新输出**完整 JSON**，修正 Beat 4 assistant 内容，其他段保持原样。"
        )
    if "user_injection" in reason:
        return (
            "刚才输出的对话中，某些 canonical fact 没有被 user 自己说出来（assistant 自己'翻译'编造了）。\n"
            "请重新输出**完整 JSON**：Beat 1（植入段）里每个 canonical value 必须由 user content 自己**直接、显式**逐字说出，"
            "assistant 只能复述 user 已经说过的字面值，不许从暗示性描述里推断翻译。"
        )
    if "canonical_order" in reason:
        return (
            "刚才输出的对话中，assistant 在 user 还没说之前就先提到了某个 canonical value，顺序错了。\n"
            "请重新输出**完整 JSON**：每个 canonical value 必须由 user 先说，assistant 后复述/召回。"
        )
    if "echo" in reason:
        return (
            "刚才 assistant 整段复读了上一轮 user 的话（degenerate）。\n"
            "请重新输出**完整 JSON**，assistant 应自然回应 user，不许复读 user 原句。"
        )
    if "schema:" in reason or "ParseError" in reason:
        return (
            f"刚才输出的 JSON 不符合 schema 或字段缺失：{reason[:300]}\n"
            f"请重新输出**严格符合 schema** 的完整 JSON：\n"
            f"  - conversations: [{{role, content, ...}}]\n"
            f"  - assistant 必须有 train_loss/value/beat 字段\n"
            f"  - value（非 null 时）必须**逐字出现**在同 turn 的 content 中"
        )
    return (
        f"刚才输出的对话不符合校验：{reason[:300]}\n"
        f"请重新输出**完整 JSON** 修正该问题，其他符合的部分保持原样。"
    )


def _resolve_sampler(model: str):
    """按 model 名 dispatch,返回 (call_with_retry, ApiError),四家 sampler 接口已对齐。

      codex-cli[:backing]  → codex_cli_sampler   (subprocess, ChatGPT Plus 配额)
      gemini-cli[:backing] → gemini_cli_sampler  (subprocess, Google 账号配额)
      含 "/"               → openrouter_sampler  (HTTP, minimax/google/qwen 等)
      其他                  → deepseek_sampler    (HTTP, deepseek-v4-flash 等)
    """
    head = model.split(":", 1)[0].lower()
    if head in ("codex-cli", "codex"):
        from xinhe.data.codex_cli_sampler import call_with_retry as fn, CodexCliError as Err
        return fn, Err
    if head in ("gemini-cli", "gemini"):
        from xinhe.data.gemini_cli_sampler import call_with_retry as fn, GeminiCliError as Err
        return fn, Err
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

    cond_retry = _supports_conditional_retry(model)
    plan = planner.plan(rng)   # plan 在整个 retry 链路中复用,保持 prefix cache 命中
    sys_p, user_p = build_messages(plan)
    base_messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]
    last_response_str: Optional[str] = None
    last_reason: Optional[str] = None

    for attempt in range(max_retries):
        ts_attempt = time.time()
        # 构造本次调用的 messages / 参数
        if cond_retry and last_response_str is not None and last_reason is not None:
            # conditional retry: 拼上轮失败响应 + fix 指令,DeepSeek prefix cache 命中前 ~90% input
            fix_msg = _build_fix_message(last_reason, plan)
            messages = base_messages + [
                {"role": "assistant", "content": last_response_str},
                {"role": "user", "content": fix_msg},
            ]
            call_kwargs = dict(
                messages=messages,
                model=model,
                temperature=0.85,
                top_p=0.92,
                max_tokens=max_tokens,
                json_mode=True,
                **extra_kwargs,
            )
        else:
            # 首次或 transport 错误后 → blind 调用(同 plan,自动命中 cache)
            call_kwargs = dict(
                model=model,
                temperature=0.85,
                top_p=0.92,
                max_tokens=max_tokens,
                json_mode=True,
                **extra_kwargs,
            )
        try:
            if cond_retry and "messages" in call_kwargs:
                resp = call_with_retry(**call_kwargs)
            else:
                resp = call_with_retry(sys_p, user_p, **call_kwargs)
            content = resp["choices"][0]["message"]["content"]
            try:
                sample = parse_response(content, plan, weight_table=weight_table, generator_model=model)
            except ParseError as pe:
                last_response_str = content
                last_reason = f"ParseError: {str(pe)[:200]}"
                print(f"  [stage1] retryable fail (attempt {attempt + 1}): ParseError: {str(pe)[:120]}", flush=True)
                continue
            result = validate(
                sample.to_dict(), stage="1", plan=plan.to_validator_plan()
            )
            if not result.ok:
                last_response_str = content
                last_reason = result.errors[0] if result.errors else "unknown reject"
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
        except ApiError as e:
            # quota 耗尽 → 立刻 abort 整个进程,不再 retry(继续打会触发账号风控)
            if _is_quota_exhausted(e):
                print(
                    f"\n[stage1] !!! {type(e).__name__}: quota exhausted, aborting process to avoid ban\n"
                    f"  detail: {str(e)[:300]}\n"
                    f"  下次跑用 resume 续上即可,已落盘的样本不会丢。",
                    flush=True,
                )
                # os._exit 立刻终止所有 worker 线程,避免 ThreadPoolExecutor 排队中的任务继续触发 quota 调用
                import os
                os._exit(2)
            # 网络/认证/HTTP error → 跟模型输出无关,清掉 last_response 走 blind retry
            last_response_str = None
            last_reason = None
            print(f"  [stage1] api fail (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        except Exception as e:
            # JSON decode / KeyError / 任意意外 → 当作 retryable,清状态
            last_response_str = None
            last_reason = None
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

    # 单实例守护：阻止"重启时漏杀 orphan python.exe → 双写"。
    with _single_instance_lock(out):
        return _generate_locked(
            out=out, rej=rej, n_samples=n_samples, mix=mix, rng=rng,
            dict_split=dict_split,
            n_canonical_range=n_canonical_range, n_turns_range=n_turns_range,
            beat3_min_turns=beat3_min_turns, beat3_min_chars=beat3_min_chars,
            beat3_chars_tolerance=beat3_chars_tolerance,
            workers=workers, model=model, weight_table=weight_table, resume=resume,
        )


def _generate_locked(
    *,
    out: Path,
    rej: Optional[Path],
    n_samples: int,
    mix: dict[str, float],
    rng: random.Random,
    dict_split: str,
    n_canonical_range: tuple[int, int],
    n_turns_range: tuple[int, int],
    beat3_min_turns: int,
    beat3_min_chars: int,
    beat3_chars_tolerance: float,
    workers: int,
    model: str,
    weight_table: dict | None,
    resume: bool,
) -> tuple[int, int]:
    """已持锁后真正干活;拆出来纯粹是为了让 with 块代码读得顺。"""
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
