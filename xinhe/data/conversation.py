"""
多轮对话数据集。

schema:
  每个 assistant turn 自带:
    - train_loss: "true" / "lm_only" / "false"（也兼容历史 bool True/False）
    - value: list[str] | None
    - value_span: list[[start_char, end_char]]  (char 坐标系，相对 assistant content)
    - value_tier: "hard" / "soft" / null
    - weight_per_span: float

DataLoader 工作:
  - 把 char span 通过 tokenizer offset_mapping 映射为 token span
  - 按 weight_per_span 给 value token 加权
  - tri-state train_loss:
      "true"     → lm_weight=1.0, value tokens 用 weight_per_span
      "lm_only"  → lm_weight=0.3, value tokens 也用 0.3（无 value 加权）
      "false"    → labels 全 -100, 不算 loss

加速:
  - 多进程 tokenize:N workers 并行处理 conversation(用 ProcessPool initializer
    给每个 worker 加载一份 tokenizer,避免每次 task 都 cold start)。
  - episode 缓存:首次构建后 pickle 到 .cache/episodes/<hash>.pt;hash 入参
    覆盖 (tokenizer / turn 限制 / 数据源 + ratio + seed + n_samples / mtime),
    任一变就 miss。重启训练直接 torch.load 几秒过 tokenize 阶段。
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "episodes"


def _make_cache_key(payload: dict) -> str:
    """16 位 sha1 摘要,key 由调用方按 dataset 类别构造。"""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _slot_cache_path(slot: str, key: str) -> Path:
    """同一 slot 同时只保留 1 份 cache。改 config → 新 hash → 落新文件 → 旧同槽自动清。"""
    return CACHE_DIR / f"{slot}_{key}.pt"


def _purge_slot_orphans(slot: str, keep_path: Path) -> None:
    """删 CACHE_DIR 下所有 `{slot}_*.pt` 除 keep_path 外的文件。"""
    if not CACHE_DIR.exists():
        return
    keep = keep_path.resolve()
    deleted = []
    for f in CACHE_DIR.glob(f"{slot}_*.pt"):
        if f.resolve() == keep:
            continue
        try:
            sz = f.stat().st_size
            f.unlink()
            deleted.append((f.name, sz))
        except OSError:
            pass
    if deleted:
        mb = sum(sz for _, sz in deleted) / (1 << 20)
        names = ", ".join(n for n, _ in deleted)
        print(f"  [cache prune] slot={slot!r} 清 {len(deleted)} 个旧 cache "
              f"({mb:.0f} MB): {names}")


def _default_n_workers() -> int:
    """不超过 8;单 jsonl 太小时最终在 build_parallel 内还会再夹一次。"""
    cpu = os.cpu_count() or 1
    return min(8, max(1, cpu // 2))


# ── ChatML fallback template (用于没有 chat_template 的 tokenizer) ──

CHATML_FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


def ensure_chat_template(tokenizer):
    """强制使用干净的 ChatML 模板，避免 Qwen3 等模型的 <think> 干扰。"""
    tokenizer.chat_template = CHATML_FALLBACK_TEMPLATE


def _resolve_lm_weight(train_loss: Union[bool, str]) -> tuple[float, bool]:
    """tri-state → (lm_weight, value_active)。

    - "true" / True   → (1.0, True)
    - "lm_only"       → (0.1, False)  让 distract 仅微弱影响 backbone,
                                       LoRA/Hippocampus 容量留给 W 写读通路
    - "false" / False → (0.0, False)
    """
    if train_loss is True:
        return 1.0, True
    if train_loss is False:
        return 0.0, False
    if isinstance(train_loss, str):
        if train_loss == "true":
            return 1.0, True
        if train_loss == "lm_only":
            return 0.1, False
        if train_loss == "false":
            return 0.0, False
    raise ValueError(f"非法 train_loss: {train_loss!r}")


def tokenize_turn(
    tokenizer,
    user_content: str,
    assistant_content: str,
    turn_max_tokens: int,
    *,
    train_loss: Union[bool, str] = "true",
    value_spans: Optional[list[list[int]]] = None,
    weight_per_span: float = 0.0,
    stats: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将一轮 user+assistant 对话 tokenize 为 (input_ids, labels, weights)。

    value_spans: list of [start_char, end_char] in **assistant_content** 坐标系。
                 函数内部会偏移到 full_text 坐标，再映射 token。

    stats: 可选 mutable dict，用于累计截断计数（key: "turns_truncated"）。
           调用方传入后能在 dataset 加载完后报"turn 截断率"。
    """
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(prefix_ids)

    # tri-state lm_weight
    lm_weight, value_active = _resolve_lm_weight(train_loss)

    # labels: prefix → -100；assistant → full_ids 对应位置；lm_weight=0 → 全 -100
    if lm_weight == 0.0:
        labels = [-100] * len(full_ids)
    else:
        labels = [-100] * prefix_len + full_ids[prefix_len:]

    # weights: 默认 assistant token = lm_weight, -100 = 0
    weights = [lm_weight if lab != -100 else 0.0 for lab in labels]

    # value token 加权（仅 value_active=True 时）
    if value_active and value_spans and weight_per_span > 0 and assistant_content:
        # 计算 assistant_content 在 full_text 中的偏移
        # apply_chat_template 会在 prefix 后面添加 assistant content（含 ChatML 包裹）
        # 找 assistant_content 在 full_text 中的起始字符位置
        assistant_offset = full_text.find(assistant_content, len(prefix_text))
        if assistant_offset < 0:
            # fallback：全文搜
            assistant_offset = full_text.find(assistant_content)

        if assistant_offset >= 0:
            # 抗自身重复保护：assistant_content 必须在 full_text 中只出现一次（否则 offset 不可信）
            count = full_text.count(assistant_content)
            if count != 1:
                # 多次出现：取第一个 prefix 之后的位置（已用 search_from prefix_text）
                # 一般来说极罕见；不抛错以避免训练打断
                pass

            encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]

            for span in value_spans:
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                s_local, e_local = int(span[0]), int(span[1])
                if s_local < 0 or e_local <= s_local:
                    continue
                s_full = assistant_offset + s_local
                e_full = assistant_offset + e_local
                for i, (c_start, c_end) in enumerate(offsets):
                    if i >= len(weights):
                        break
                    if weights[i] > 0 and c_start < e_full and c_end > s_full:
                        weights[i] = weight_per_span

    # 截断
    if len(full_ids) > turn_max_tokens:
        if stats is not None:
            stats["turns_truncated"] = stats.get("turns_truncated", 0) + 1
            stats["max_turn_tokens_seen"] = max(stats.get("max_turn_tokens_seen", 0), len(full_ids))
        full_ids = full_ids[:turn_max_tokens]
        labels = labels[:turn_max_tokens]
        weights = weights[:turn_max_tokens]

    # Padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = turn_max_tokens - len(full_ids)
    if pad_len > 0:
        full_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        weights = weights + [0.0] * pad_len

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float),
    )


def _empty_stats() -> dict:
    return {
        "turns_total": 0, "turns_truncated": 0,
        "episodes_total": 0, "episodes_truncated": 0,
        "max_turn_tokens_seen": 0, "max_turns_in_episode_seen": 0,
    }


def _merge_stats(into: dict, src: dict) -> None:
    for k in ("turns_total", "turns_truncated", "episodes_total", "episodes_truncated"):
        into[k] += src.get(k, 0)
    for k in ("max_turn_tokens_seen", "max_turns_in_episode_seen"):
        into[k] = max(into[k], src.get(k, 0))


# ── 多进程 tokenize:每 worker 进程持久化一份 tokenizer + 公共参数 ──

_WORKER_TOKENIZER = None
_WORKER_PARAMS: dict = {}


def _worker_init(tokenizer_path: str, turn_max: int, max_turns: int,
                 value_weight_cap: Optional[float]) -> None:
    global _WORKER_TOKENIZER, _WORKER_PARAMS
    # 多进程 + Rust 内多线程双重并行会互相抢核。按 mp 一进程一核分。
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)
    _WORKER_TOKENIZER = tok
    _WORKER_PARAMS = {
        "turn_max_tokens": int(turn_max),
        "max_turns_per_episode": int(max_turns),
        "value_weight_cap": value_weight_cap,
    }


def _worker_process_chunk(items: list[dict]) -> tuple[bytes, dict]:
    """worker 入口:输入 conversation dict 列表,返回 (numpy episodes blob, stats)。

    跨进程不传 torch.Tensor —— 默认 sharing strategy 用 fd 级 shared memory,
    50k 样本 × 4 turn × 3 tensor 一次返回过来直接打爆 fd 上限(Too many open files)。
    转 numpy + 自定义打包后由主进程一次性 reconstruct。
    """
    import pickle
    stats = _empty_stats()
    out_episodes_np = []
    for item in items:
        ep = _process_conversation_pure(_WORKER_TOKENIZER, item, _WORKER_PARAMS, stats)
        if ep and len(ep) >= 2:
            ep_np = [(t0.numpy(), t1.numpy(), t2.numpy()) for (t0, t1, t2) in ep]
            out_episodes_np.append(ep_np)
    return pickle.dumps(out_episodes_np, protocol=pickle.HIGHEST_PROTOCOL), stats


def _process_conversation_pure(
    tokenizer,
    item: dict,
    params: dict,
    stats: dict,
) -> Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """无 self-state 的 _process_conversation 等价实现,用于 worker / serial 共用。"""
    conversations = item.get("conversations", [])
    if not conversations:
        return None
    turn_max = params["turn_max_tokens"]
    max_turns = params["max_turns_per_episode"]
    cap = params.get("value_weight_cap")

    turn_tensors = []
    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i].get("content", "")
        asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
        assistant_msg = asst_entry.get("content", "")
        train_loss = asst_entry.get("train_loss", "true")
        value_spans = asst_entry.get("value_span") or []
        weight_per_span = float(asst_entry.get("weight_per_span", 0.0) or 0.0)
        if cap is not None:
            weight_per_span = min(weight_per_span, cap)

        turn_tensor = tokenize_turn(
            tokenizer, user_msg, assistant_msg, turn_max,
            train_loss=train_loss,
            value_spans=value_spans,
            weight_per_span=weight_per_span,
            stats=stats,
        )
        turn_tensors.append(turn_tensor)
        stats["turns_total"] += 1

    stats["episodes_total"] += 1
    stats["max_turns_in_episode_seen"] = max(
        stats["max_turns_in_episode_seen"], len(turn_tensors)
    )

    if len(turn_tensors) > max_turns:
        stats["episodes_truncated"] += 1
        turn_tensors = turn_tensors[: max_turns - 1] + [turn_tensors[-1]]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while len(turn_tensors) < max_turns:
        dummy_ids = torch.full((turn_max,), pad_id, dtype=torch.long)
        dummy_labels = torch.full((turn_max,), -100, dtype=torch.long)
        dummy_weights = torch.zeros(turn_max, dtype=torch.float)
        turn_tensors.append((dummy_ids, dummy_labels, dummy_weights))
    return turn_tensors


def _build_episodes_parallel(
    items: list[dict],
    tokenizer,
    turn_max: int,
    max_turns: int,
    value_weight_cap: Optional[float],
    n_workers: Optional[int],
) -> tuple[list, dict]:
    """走 ProcessPool 并行 tokenize;n_workers<=1 或样本太少时退化为串行。"""
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    n_workers = min(n_workers, max(1, len(items) // 200))

    params = {
        "turn_max_tokens": int(turn_max),
        "max_turns_per_episode": int(max_turns),
        "value_weight_cap": value_weight_cap,
    }

    if n_workers <= 1:
        stats = _empty_stats()
        episodes = []
        for item in items:
            ep = _process_conversation_pure(tokenizer, item, params, stats)
            if ep and len(ep) >= 2:
                episodes.append(ep)
        return episodes, stats

    from concurrent.futures import ProcessPoolExecutor

    n_chunks = n_workers * 4   # 比 worker 多一些,负载更均匀
    chunks = [items[i::n_chunks] for i in range(n_chunks)]
    chunks = [c for c in chunks if c]

    print(f"  [tokenize] {len(items)} samples × {n_workers} workers ...")
    import pickle
    episodes: list = []
    total_stats = _empty_stats()
    tok_path = getattr(tokenizer, "name_or_path", None) or str(tokenizer)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(tok_path, turn_max, max_turns, value_weight_cap),
    ) as pool:
        for blob, st in pool.map(_worker_process_chunk, chunks):
            ep_list_np = pickle.loads(blob)
            for ep_np in ep_list_np:
                ep_t = [(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c))
                        for (a, b, c) in ep_np]
                episodes.append(ep_t)
            _merge_stats(total_stats, st)
    return episodes, total_stats


class ConversationDataset(Dataset):
    """
    多轮对话数据集。

    数据格式 (JSONL):
    {
      "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "train_loss": "true", "value": [...],
         "value_span": [[s,e], ...], "value_tier": "hard", "weight_per_span": 5.0}
      ]
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        turn_max_tokens: int = 256,
        max_turns_per_episode: int = 16,
        value_weight_cap: Optional[float] = None,
        n_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_slot: str = "single",
    ):
        """value_weight_cap: 把 weight_per_span 上限 cap 到这个值。
        v8 旧数据里 VALUE token weight 常是 5.0(强化记忆);v9 fast-weights 数学
        可学,不靠 weighted loss → 训练入口传 1.0 让所有 VALUE token 退化为普通 token。
        None = 不 cap(沿用 jsonl 原值)。

        n_workers:    ProcessPool 并行 tokenize 进程数;None 走默认(min(8, cpu//2))。
        use_cache:    True 时把构建好的 episodes pickle 到 .cache/episodes/<slot>_<hash>.pt。
        cache_slot:   同 slot 同时只保留 1 份 cache,落新文件时自动删旧同槽。
                      caller 用不同 slot(如 train.py 的 "train"/"val")避免互删。"""
        self.tokenizer = tokenizer
        self.turn_max_tokens = turn_max_tokens
        self.max_turns_per_episode = max_turns_per_episode
        self.value_weight_cap = value_weight_cap

        ensure_chat_template(tokenizer)

        self._stats = _empty_stats()
        self.episodes: list = []
        data_path_p = Path(data_path)
        data_path_str = str(data_path_p)

        cache_path: Optional[Path] = None
        if use_cache and data_path_p.exists():
            mtime = data_path_p.stat().st_mtime
            key = _make_cache_key({
                "kind": "single",
                "schema_version": "v9.5",  # paragraph distract / per-layer K/V 数据 schema 标识
                "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
                "turn_max": turn_max_tokens,
                "max_turns": max_turns_per_episode,
                "value_weight_cap": value_weight_cap,
                "data_path": str(data_path_p.resolve()),
                "mtime": mtime,
            })
            cache_path = _slot_cache_path(cache_slot, key)
            if cache_path.exists():
                blob = torch.load(cache_path, weights_only=False)
                self.episodes = blob["episodes"]
                self._stats = blob.get("stats", _empty_stats())
                self._report_truncation_stats(f"{cache_slot} [cache]")
                return

        if data_path_p.exists():
            with open(data_path_p, "r", encoding="utf-8") as f:
                items = [json.loads(ln) for ln in f if ln.strip()]
            self.episodes, self._stats = _build_episodes_parallel(
                items, tokenizer, turn_max_tokens, max_turns_per_episode,
                value_weight_cap, n_workers,
            )

        if cache_path is not None and self.episodes:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"episodes": self.episodes, "stats": self._stats}, cache_path)
            _purge_slot_orphans(cache_slot, keep_path=cache_path)

        self._report_truncation_stats(f"{cache_slot} [build]")

    def _report_truncation_stats(self, label: str) -> None:
        """两行格式:
           {label}: N ep × M turns
             截断: turn X.X%  ep X.X% ⚠ (实际 max 19 > 配置 4)   ← 仅有截断才出
        """
        s = self._stats
        if s["turns_total"] == 0:
            return

        print(f"  {label}: {s['episodes_total']} ep × {s['turns_total']} turns")

        turn_rate = s["turns_truncated"] / s["turns_total"]
        ep_rate = s["episodes_truncated"] / max(s["episodes_total"], 1)

        if turn_rate == 0.0 and ep_rate == 0.0:
            return

        notes: list[str] = []
        if turn_rate > 0.05:
            notes.append(
                f"turn {turn_rate:.1%} ⚠ (max {s['max_turn_tokens_seen']} > {self.turn_max_tokens})"
            )
        elif turn_rate > 0.0:
            notes.append(f"turn {turn_rate:.1%}")

        if ep_rate > 0.05:
            notes.append(
                f"ep {ep_rate:.1%} ⚠ (max {s['max_turns_in_episode_seen']} > {self.max_turns_per_episode})"
            )
        elif ep_rate > 0.0:
            notes.append(f"ep {ep_rate:.1%}")

        print("    截断: " + "  ".join(notes))

    def _process_conversation(
        self, item: dict
    ) -> Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """将一个对话转为 (input_ids, labels, weights) turn tensor 列表。
        每个 turn = 1 个 user-asst pair → 1 个 (B,T) 三元组。"""
        conversations = item.get("conversations", [])
        if not conversations:
            return None

        turn_tensors = []

        for i in range(0, len(conversations) - 1, 2):
            user_msg = conversations[i].get("content", "")
            asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
            assistant_msg = asst_entry.get("content", "")
            train_loss = asst_entry.get("train_loss", "true")
            value_spans = asst_entry.get("value_span") or []
            weight_per_span = float(asst_entry.get("weight_per_span", 0.0) or 0.0)
            if self.value_weight_cap is not None:
                weight_per_span = min(weight_per_span, self.value_weight_cap)

            turn_tensor = tokenize_turn(
                self.tokenizer, user_msg, assistant_msg, self.turn_max_tokens,
                train_loss=train_loss,
                value_spans=value_spans,
                weight_per_span=weight_per_span,
                stats=self._stats,
            )
            turn_tensors.append(turn_tensor)
            self._stats["turns_total"] += 1

        self._stats["episodes_total"] += 1
        self._stats["max_turns_in_episode_seen"] = max(
            self._stats["max_turns_in_episode_seen"], len(turn_tensors)
        )

        if len(turn_tensors) > self.max_turns_per_episode:
            # 截断：保留前 (max_turns_per_episode - 1) 个 + 末尾那个
            # （避免 episode 末轮 recall 被丢；中间被砍是因为 max_turns_per_episode 配小了）
            self._stats["episodes_truncated"] += 1
            turn_tensors = turn_tensors[: self.max_turns_per_episode - 1] + [turn_tensors[-1]]

        # Pad 到 max_turns_per_episode：避免 collate_episodes 用 min_len 截断整个 batch
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        while len(turn_tensors) < self.max_turns_per_episode:
            dummy_ids = torch.full((self.turn_max_tokens,), pad_id, dtype=torch.long)
            dummy_labels = torch.full((self.turn_max_tokens,), -100, dtype=torch.long)
            dummy_weights = torch.zeros(self.turn_max_tokens, dtype=torch.float)
            turn_tensors.append((dummy_ids, dummy_labels, dummy_weights))

        return turn_tensors

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.episodes[idx]


class MixedConversationDataset(ConversationDataset):
    """动态多源混合(不落 mix 文件,节约 data 目录大小)。

    sources: list of {path, ratio, tag}。按 ratio 抽样到内存 pool,然后整体 shuffle,
    最后调 _process_conversation 转 turn tensor。源不足时循环抽样保比例。
    """

    def __init__(
        self,
        sources: list[dict],
        n_samples: int,
        seed: int,
        tokenizer,
        turn_max_tokens: int = 256,
        max_turns_per_episode: int = 16,
        value_weight_cap: Optional[float] = None,
        n_workers: Optional[int] = None,
        use_cache: bool = True,
        cache_slot: str = "mix",
    ):
        self.tokenizer = tokenizer
        self.turn_max_tokens = turn_max_tokens
        self.max_turns_per_episode = max_turns_per_episode
        self.value_weight_cap = value_weight_cap
        ensure_chat_template(tokenizer)

        self._stats = _empty_stats()
        self.episodes: list = []

        # Cache key:tokenizer + 长度限制 + 各 source(path/ratio/mtime) + seed + n_samples
        # 任一变 → 重建。读 mtime 避免 jsonl 内容变了 cache stale。
        # cache_slot:同 slot 同时只保留 1 份(改任一字段就替换),caller 用不同 slot
        # (如 "mix_train" / "mix_val")避免互删。
        cache_path: Optional[Path] = None
        if use_cache:
            src_key = []
            for s in sources:
                p = Path(s["path"])
                src_key.append({
                    "path": str(p.resolve()),
                    "ratio": float(s.get("ratio", 0)),
                    "tag": s.get("tag", p.parent.name),
                    "mtime": p.stat().st_mtime if p.exists() else 0,
                })
            key = _make_cache_key({
                "kind": "mix_dynamic",
                "schema_version": "v9.5",  # paragraph distract / per-layer K/V 数据 schema 标识
                "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
                "turn_max": turn_max_tokens,
                "max_turns": max_turns_per_episode,
                "value_weight_cap": value_weight_cap,
                "sources": src_key,
                "seed": int(seed),
                "n_samples": int(n_samples),
            })
            cache_path = _slot_cache_path(cache_slot, key)
            if cache_path.exists():
                blob = torch.load(cache_path, weights_only=False)
                self.episodes = blob["episodes"]
                self._stats = blob.get("stats", _empty_stats())
                self._report_truncation_stats(f"{cache_slot} [cache]")
                return

        import random as _random
        rng = _random.Random(seed)

        total_ratio = sum(float(s.get("ratio", 0)) for s in sources)
        if total_ratio <= 0:
            raise ValueError("MixedConversationDataset: sources 比例总和必须 > 0")

        pool: list[dict] = []
        mix_breakdown: list[str] = []
        for src in sources:
            path = Path(src["path"])
            ratio = float(src.get("ratio", 0)) / total_ratio
            tag = src.get("tag", path.parent.name)
            n_take = int(round(n_samples * ratio))
            if n_take <= 0:
                continue
            if not path.exists():
                raise FileNotFoundError(f"MixedConversationDataset: source 不存在 {path}")
            with open(path, "r", encoding="utf-8") as f:
                items = [json.loads(ln) for ln in f if ln.strip()]
            if not items:
                raise ValueError(f"MixedConversationDataset: source 空 {path}")
            rng.shuffle(items)
            if len(items) >= n_take:
                taken = items[:n_take]
            else:
                taken = (items * (1 + n_take // len(items)))[:n_take]
            for s in taken:
                meta = dict(s.get("meta", {}))
                meta["mix_source"] = tag
                s["meta"] = meta
            pool.extend(taken)
            mix_breakdown.append(f"{tag}={len(taken)}")

        # mix 抽样组成单行打印
        print(f"  {cache_slot} [build mix]: " + " ".join(mix_breakdown))

        rng.shuffle(pool)
        pool = pool[:n_samples]

        self.episodes, self._stats = _build_episodes_parallel(
            pool, tokenizer, turn_max_tokens, max_turns_per_episode,
            value_weight_cap, n_workers,
        )

        if cache_path is not None and self.episodes:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"episodes": self.episodes, "stats": self._stats}, cache_path)
            _purge_slot_orphans(cache_slot, keep_path=cache_path)

        self._report_truncation_stats(f"{cache_slot} [build]")


def collate_episodes(
    batch: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """将多个 episode 整理为 batch（取最短的 episode turn 数）。"""
    min_len = min(len(episode) for episode in batch)
    turn_batches = []
    for turn_idx in range(min_len):
        ids_batch = torch.stack([episode[turn_idx][0] for episode in batch])
        labels_batch = torch.stack([episode[turn_idx][1] for episode in batch])
        weights_batch = torch.stack([episode[turn_idx][2] for episode in batch])
        turn_batches.append((ids_batch, labels_batch, weights_batch))
    return turn_batches
