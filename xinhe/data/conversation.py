"""
多轮对话数据集

将对话按轮次切分为 segment，同一对话的多轮组成 episode。
state 跨轮次传递，训练模型利用持久状态记忆对话上下文。

每个 segment 返回 (input_ids, labels) tuple:
- input_ids: 完整 token 序列
- labels: user/template token 位置为 -100，只在 assistant token 上计算 loss
"""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# ── ChatML fallback template (用于没有 chat_template 的 tokenizer，如 MiniMind) ──

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将一轮 user+assistant 对话 tokenize 为 (input_ids, labels)。

    - 用 apply_chat_template 两步确定 assistant 起始位置
    - labels: user/template 部分 = -100, assistant 部分 = 实际 token id, padding = -100
    - compute_loss=False 时，整个 segment 的 labels 全为 -100（不参与 loss 计算）
    """
    # Step 1: tokenize user 部分 + generation prompt → 得到 prefix 长度
    # 使用强制设置的 ChatML 模板，无 <think> 干扰
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    # Step 2: tokenize 完整 turn (user + assistant)
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
        # 正常: prefix 部分 -100, assistant 部分保留
        labels = [-100] * prefix_len + full_ids[prefix_len:]
    else:
        # 整个 segment 不参与 loss
        labels = [-100] * len(full_ids)

    # 截断
    if len(full_ids) > segment_length:
        full_ids = full_ids[:segment_length]
        labels = labels[:segment_length]

    # Padding
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = segment_length - len(full_ids)
    if pad_len > 0:
        full_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
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

        # 加载对话数据
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
    ) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        """将一个对话转为 (input_ids, labels) segment 列表。"""
        conversations = item.get("conversations", [])
        if not conversations:
            return None

        segments = []

        for i in range(0, len(conversations) - 1, 2):
            user_msg = conversations[i].get("content", "")
            asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
            assistant_msg = asst_entry.get("content", "")
            # train_loss 字段控制该 segment 是否参与 loss 计算（默认 True）
            compute_loss = asst_entry.get("train_loss", True)

            segment = tokenize_turn(
                self.tokenizer, user_msg, assistant_msg, self.segment_length,
                compute_loss=compute_loss,
            )
            segments.append(segment)

        # 限制 episode 长度
        if len(segments) > self.episode_length:
            segments = segments[: self.episode_length]

        return segments

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """返回一个 episode (segment 列表)，每个 segment 是 (input_ids, labels)"""
        return self.episodes[idx]


def collate_episodes(
    batch: list[list[tuple[torch.Tensor, torch.Tensor]]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    将多个 episode 整理为 batch。

    由于不同 episode 可能有不同数量的 segment，
    取最短的 episode 长度，截断较长的。

    返回: segment 列表，每个 segment 是 (input_ids_batch, labels_batch)
    """
    min_len = min(len(episode) for episode in batch)
    segments = []

    for seg_idx in range(min_len):
        ids_batch = torch.stack([episode[seg_idx][0] for episode in batch])
        labels_batch = torch.stack([episode[seg_idx][1] for episode in batch])
        segments.append((ids_batch, labels_batch))

    return segments


