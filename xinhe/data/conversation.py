"""
多轮对话数据集 (v8)。

v8 schema:
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
"""
import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset


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
    - "lm_only"       → (0.3, False)
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
            return 0.3, False
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


class ConversationDataset(Dataset):
    """
    多轮对话数据集。

    数据格式 (v8 JSONL):
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
    ):
        self.tokenizer = tokenizer
        self.turn_max_tokens = turn_max_tokens
        self.max_turns_per_episode = max_turns_per_episode

        ensure_chat_template(tokenizer)

        # 截断统计：dataloader 加载完后输出，用户能看到配置/数据是否对齐
        # turn 层面：tokenize_turn 截断（turn_max_tokens 偏小）
        # episode 层面：_process_conversation 截断（max_turns_per_episode 偏小）
        self._stats = {
            "turns_total": 0,
            "turns_truncated": 0,
            "episodes_total": 0,
            "episodes_truncated": 0,
            "max_turn_tokens_seen": 0,
            "max_turns_in_episode_seen": 0,
        }

        self.episodes = []
        data_path = Path(data_path)

        if data_path.exists():
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    episode = self._process_conversation(item)
                    if episode and len(episode) >= 2:
                        self.episodes.append(episode)

        self._report_truncation_stats(str(data_path))

    def _report_truncation_stats(self, data_path: str) -> None:
        """加载完后输出截断率；任一 > 5% 触发 warning（含可操作 Hint）。"""
        import logging
        logger = logging.getLogger(__name__)

        s = self._stats
        if s["turns_total"] == 0:
            return

        turn_rate = s["turns_truncated"] / s["turns_total"]
        ep_rate = s["episodes_truncated"] / max(s["episodes_total"], 1)

        msg = (
            f"[ConversationDataset] {data_path}: "
            f"episodes={s['episodes_total']} turns={s['turns_total']} | "
            f"turn_truncation_rate={turn_rate:.2%} (max_turn_tokens_seen={s['max_turn_tokens_seen']}, turn_max_tokens={self.turn_max_tokens}) | "
            f"episode_truncation_rate={ep_rate:.2%} (max_turns_in_episode_seen={s['max_turns_in_episode_seen']}, max_turns_per_episode={self.max_turns_per_episode})"
        )
        print(msg)

        if turn_rate > 0.05:
            logger.warning(
                f"turn_truncation_rate={turn_rate:.2%} > 5% in {data_path}. "
                f"max_turn_tokens_seen={s['max_turn_tokens_seen']} > turn_max_tokens={self.turn_max_tokens}. "
                f"Hint: raise turn_max_tokens to ≥{((s['max_turn_tokens_seen'] + 127) // 128) * 128} "
                f"or shorten turn content (e.g. lower beat3_min_chars in stage1)."
            )
        if ep_rate > 0.05:
            logger.warning(
                f"episode_truncation_rate={ep_rate:.2%} > 5% in {data_path}. "
                f"max_turns_in_episode_seen={s['max_turns_in_episode_seen']} > max_turns_per_episode={self.max_turns_per_episode}. "
                f"Hint: raise max_turns_per_episode (yaml) to ≥{s['max_turns_in_episode_seen']} "
                f"so dataloader covers all turns without truncation."
            )

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
