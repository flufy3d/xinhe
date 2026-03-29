"""
测试训练循环在小数据上能跑通
"""
import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import collate_episodes


class TinyBackbone(nn.Module, BackboneBase):
    """极小 backbone 用于训练测试"""

    def __init__(self):
        nn.Module.__init__(self)
        self._embed = nn.Embedding(50, 32)
        self._block = nn.Linear(32, 32)
        self._head = nn.Linear(32, 50, bias=False)

    def embed(self, input_ids):
        return self._embed(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None):
        return self._block(hidden_states)

    def get_lm_head(self):
        return self._head

    def get_hidden_size(self):
        return 32


class DummyDataset:
    """极小数据集: 每个 episode 有 4 个 segment"""

    def __init__(self, num_episodes=10, seg_len=16, num_segments=4):
        self.episodes = []
        for _ in range(num_episodes):
            episode = [torch.randint(0, 50, (seg_len,)) for _ in range(num_segments)]
            self.episodes.append(episode)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def test_training_loop_runs():
    """训练循环能跑通几步"""
    config = XinheConfig(
        hidden_size=32,
        n_state=4,
        state_dim=32,
        state_scale_init=-5.0,
        lora_rank=0,
        freeze_backbone=False,
        tbptt_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        grad_clip=1.0,
        warmup_steps=2,
        max_steps=4,
        device="cpu",
        dtype="float32",
    )

    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=8, seg_len=16, num_segments=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    # 不应报错
    trainer.train()
    assert trainer.global_step > 0


def test_state_detach_in_tbptt():
    """验证截断 BPTT 正确 detach 状态"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=32,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    B, T = 2, 8
    state = model.init_state(B)

    # 模拟 4 个 segment，tbptt=2
    for seg_idx in range(4):
        if seg_idx > 0 and seg_idx % config.tbptt_steps == 0:
            state = state.detach()
            assert not state.requires_grad

        input_ids = torch.randint(0, 50, (B, T))
        result = model(input_ids, state, labels=input_ids)
        state = result["state_next"]

        # 在 BPTT 窗口内，state 应该可以反向传播
        if seg_idx % config.tbptt_steps > 0:
            assert state.requires_grad or result["loss"].requires_grad
