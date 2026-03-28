"""
多轮对话数据集

将对话按轮次切分为 segment，同一对话的多轮组成 episode。
state 跨轮次传递，训练模型利用持久状态记忆对话上下文。
"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    """
    多轮对话数据集。

    数据格式 (JSONL):
    {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    每个对话被切分为多个 segment (轮次)，组成一个 episode。
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

    def _process_conversation(self, item: dict) -> Optional[list[torch.Tensor]]:
        """
        将一个对话转为 segment 列表。

        每个轮次 (user + assistant) 编码为一个 segment。
        """
        conversations = item.get("conversations", [])
        if not conversations:
            return None

        segments = []

        # 将对话轮次编码为 token segment
        for i in range(0, len(conversations) - 1, 2):
            user_msg = conversations[i].get("content", "")
            assistant_msg = conversations[i + 1].get("content", "") if i + 1 < len(conversations) else ""

            # 格式化为对话模板
            text = f"<s>用户：{user_msg}\n助手：{assistant_msg}</s>"
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # 截断或填充到 segment_length
            if len(token_ids) > self.segment_length:
                token_ids = token_ids[:self.segment_length]
            elif len(token_ids) < self.segment_length:
                # 用 pad token 填充
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                token_ids = token_ids + [pad_id] * (self.segment_length - len(token_ids))

            segments.append(torch.tensor(token_ids, dtype=torch.long))

        # 限制 episode 长度
        if len(segments) > self.episode_length:
            segments = segments[:self.episode_length]

        return segments

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        """返回一个 episode (segment 列表)"""
        return self.episodes[idx]


def collate_episodes(batch: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """
    将多个 episode 整理为 batch。

    由于不同 episode 可能有不同数量的 segment，
    取最短的 episode 长度，截断较长的。

    返回: segment 列表，每个 segment 是 (B, T) tensor
    """
    min_len = min(len(episode) for episode in batch)
    segments = []

    for seg_idx in range(min_len):
        seg_batch = torch.stack([episode[seg_idx] for episode in batch])  # (B, T)
        segments.append(seg_batch)

    return segments


class SyntheticMemoryDataset(Dataset):
    """
    合成记忆测试数据集。

    生成需要跨 segment 记忆的对话 episode:
    - Segment k: "请记住，我的名字是 {name}"
    - Segment k+d: "我叫什么名字？" → 期望模型输出 {name}

    用于可控地测试记忆保留能力。
    """

    def __init__(
        self,
        tokenizer,
        num_episodes: int = 1000,
        segment_length: int = 256,
        episode_length: int = 8,
        max_distance: int = 4,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.episode_length = episode_length

        rng = random.Random(seed)

        # 预定义的记忆内容
        names = ["小明", "小红", "张三", "李四", "王五", "小刚", "小芳", "小华"]
        cities = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安"]
        foods = ["火锅", "烤鸭", "拉面", "寿司", "披萨", "饺子", "米粉", "煎饼"]

        self.episodes = []

        for _ in range(num_episodes):
            name = rng.choice(names)
            city = rng.choice(cities)
            food = rng.choice(foods)

            segments = []
            # 创建一个 episode 的对话
            queries = [
                (f"请记住，我叫{name}，我住在{city}，我最喜欢吃{food}。", f"好的，我记住了！你叫{name}，住在{city}，最喜欢吃{food}。"),
            ]

            # 中间填充一些闲聊
            fillers = [
                ("今天天气怎么样？", "今天天气不错，阳光明媚。"),
                ("给我讲个笑话吧。", "好的，为什么程序员喜欢黑色？因为黑色是所有颜色的集合！"),
                ("你觉得AI的未来会怎样？", "AI技术在不断进步，未来会有更多有趣的应用。"),
                ("推荐一本书吧。", "我推荐《三体》，这是一部非常精彩的科幻小说。"),
            ]

            # 记忆段
            queries_encoded = []
            for user_text, assistant_text in queries:
                text = f"<s>用户：{user_text}\n助手：{assistant_text}</s>"
                token_ids = self._encode_and_pad(text)
                queries_encoded.append(token_ids)

            segments.extend(queries_encoded)

            # 填充段
            distance = rng.randint(1, min(max_distance, self.episode_length - 2))
            for i in range(distance):
                filler = rng.choice(fillers)
                text = f"<s>用户：{filler[0]}\n助手：{filler[1]}</s>"
                segments.append(self._encode_and_pad(text))

            # 回忆测试段
            recall_questions = [
                (f"我叫什么名字？", f"你叫{name}。"),
                (f"我住在哪里？", f"你住在{city}。"),
                (f"我喜欢吃什么？", f"你最喜欢吃{food}。"),
            ]
            recall = rng.choice(recall_questions)
            text = f"<s>用户：{recall[0]}\n助手：{recall[1]}</s>"
            segments.append(self._encode_and_pad(text))

            # 补充到 episode_length
            while len(segments) < self.episode_length:
                filler = rng.choice(fillers)
                text = f"<s>用户：{filler[0]}\n助手：{filler[1]}</s>"
                segments.append(self._encode_and_pad(text))

            self.episodes.append(segments[:self.episode_length])

    def _encode_and_pad(self, text: str) -> torch.Tensor:
        """编码文本并填充/截断到 segment_length"""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > self.segment_length:
            token_ids = token_ids[:self.segment_length]
        elif len(token_ids) < self.segment_length:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            token_ids = token_ids + [pad_id] * (self.segment_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        return self.episodes[idx]
