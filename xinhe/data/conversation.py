"""
多轮对话数据集

将对话按轮次切分为 segment，同一对话的多轮组成 episode。
state 跨轮次传递，训练模型利用持久状态记忆对话上下文。

每个 segment 返回 (input_ids, labels, weights) 三元组:
- input_ids: 完整 token 序列
- labels: user/template token 位置为 -100，只在 assistant token 上计算 loss
- weights: 每 token 的 loss 权重 (VALUE token=VALUE_WEIGHT，其他 assistant=1.0，-100=0.0)
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# 加权 loss: VALUE token 相对其他 assistant token 的梯度权重
VALUE_WEIGHT = 5.0


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


def tokenize_turn(
    tokenizer,
    user_content: str,
    assistant_content: str,
    segment_length: int,
    compute_loss: bool = True,
    value_str: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将一轮 user+assistant 对话 tokenize 为 (input_ids, labels, weights)。

    - labels: user/template 部分 = -100, assistant 部分 = 实际 token id, padding = -100
    - compute_loss=False 时，整个 segment 的 labels 全为 -100
    - weights: VALUE token=VALUE_WEIGHT, 其他 assistant token=1.0, -100 位置=0.0
    - value_str: recall 轮的目标字符串 (如 "他是高级木匠"); 为 None 时所有 assistant token 权重=1.0
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

    # 构建 labels
    if compute_loss:
        labels = [-100] * prefix_len + full_ids[prefix_len:]
    else:
        labels = [-100] * len(full_ids)

    # 构建 weights: 默认 assistant token 权重 1.0, -100 位置 0.0
    weights = [1.0 if lab != -100 else 0.0 for lab in labels]

    # VALUE token 加权: 用 offset_mapping 定位 value 子串
    if compute_loss and value_str:
        value_start = full_text.find(value_str, len(prefix_text))
        if value_start < 0:
            value_start = full_text.find(value_str)  # fallback: 全文搜索
        if value_start >= 0:
            value_end = value_start + len(value_str)
            encoded = tokenizer(full_text, add_special_tokens=False,
                                return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            for i, (c_start, c_end) in enumerate(offsets):
                if i >= len(weights):
                    break
                # token 与 value 区间有重叠 → VALUE token
                if weights[i] > 0 and c_start < value_end and c_end > value_start:
                    weights[i] = VALUE_WEIGHT

    # 截断
    if len(full_ids) > segment_length:
        full_ids = full_ids[:segment_length]
        labels = labels[:segment_length]
        weights = weights[:segment_length]

    # Padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = segment_length - len(full_ids)
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

    数据格式 (JSONL):
    {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    每个对话被切分为多个 segment (轮次)，组成一个 episode。
    每个 segment 是 (input_ids, labels) tuple。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        segment_length: int = 256,
        episode_length: int = 16,
    ):
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.episode_length = episode_length

        ensure_chat_template(tokenizer)

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

    def _process_conversation(
        self, item: dict
    ) -> Optional[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """将一个对话转为 (input_ids, labels, weights) segment 列表。"""
        conversations = item.get("conversations", [])
        if not conversations:
            return None

        segments = []

        for i in range(0, len(conversations) - 1, 2):
            user_msg = conversations[i].get("content", "")
            asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
            assistant_msg = asst_entry.get("content", "")
            compute_loss = asst_entry.get("train_loss", True)
            value_str = asst_entry.get("value")

            segment = tokenize_turn(
                self.tokenizer, user_msg, assistant_msg, self.segment_length,
                compute_loss=compute_loss,
                value_str=value_str,
            )
            segments.append(segment)

        if len(segments) > self.episode_length:
            segments = segments[: self.episode_length]

        return segments

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """返回一个 episode (segment 列表)，每个 segment 是 (input_ids, labels, weights)"""
        return self.episodes[idx]


def collate_episodes(
    batch: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    将多个 episode 整理为 batch。

    取最短的 episode 长度，截断较长的。

    返回: segment 列表，每个 segment 是 (input_ids_batch, labels_batch, weights_batch)
    """
    min_len = min(len(episode) for episode in batch)
    segments = []

    for seg_idx in range(min_len):
        ids_batch = torch.stack([episode[seg_idx][0] for episode in batch])
        labels_batch = torch.stack([episode[seg_idx][1] for episode in batch])
        weights_batch = torch.stack([episode[seg_idx][2] for episode in batch])
        segments.append((ids_batch, labels_batch, weights_batch))

    return segments
