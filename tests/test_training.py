"""
测试训练循环在小数据上能跑通 (v5c)
"""
import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import collate_episodes


N_LAYERS = 4


class TinyBackbone(nn.Module, BackboneBase):
    """极小 backbone 用于训练测试"""

    def __init__(self):
        nn.Module.__init__(self)
        self._embed = nn.Embedding(50, 32)
        self._block = nn.Linear(32, 32)
        self._head = nn.Linear(32, 50, bias=False)

    def embed(self, input_ids):
        return self._embed(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None, position_ids=None, layer_hook=None):
        if layer_hook is not None:
            for i in range(N_LAYERS):
                hidden_states = layer_hook(hidden_states, i)
        return self._block(hidden_states)

    def get_lm_head(self):
        return self._head

    def get_hidden_size(self):
        return 32

    def get_num_layers(self):
        return N_LAYERS


class DummyDataset:
    """极小数据集: 每个 episode 有 num_segments 个 segment, 每个 (ids, labels, weights) 三元组"""

    def __init__(self, num_episodes=10, seg_len=16, num_segments=4):
        self.episodes = []
        for _ in range(num_episodes):
            episode = []
            for _ in range(num_segments):
                ids = torch.randint(0, 50, (seg_len,))
                labels = ids.clone()
                weights = torch.ones(seg_len, dtype=torch.float)
                episode.append((ids, labels, weights))
            self.episodes.append(episode)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def test_training_loop_runs():
    """训练循环能跑通几步"""
    config = XinheConfig(
        hidden_size=32,
        n_heads=4,
        head_dim=8,
        read_scale_init=-5.0,
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

    trainer.train()
    assert trainer.global_step > 0


def test_state_detach_in_tbptt():
    """验证截断 BPTT 正确 detach 状态"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    B, T = 2, 8
    state = model.init_state(B)

    for seg_idx in range(4):
        if seg_idx > 0 and seg_idx % config.tbptt_steps == 0:
            state = state.detach()
            assert not state.requires_grad

        input_ids = torch.randint(0, 50, (B, T))
        labels = input_ids.clone()
        result = model(input_ids, state, labels=labels)
        state = result["state_next"]

        if seg_idx % config.tbptt_steps > 0:
            assert state.requires_grad or result["loss"].requires_grad


def test_v5a_two_param_groups():
    """v6: optimizer 分 fact / turn / lora 三组 (lora_rank=0 时仅 fact+turn 两组，turn off 时仅 fact 一组)"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
        enable_turn_memory=False,   # 显式关闭，回到单 fact 路径
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=4, seg_len=8, num_segments=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    # turn_off + lora_rank=0 → 只剩 fact 一组
    assert len(trainer.optimizer.param_groups) == 1

    # 开启 turn → fact + turn 两组
    config2 = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
        enable_turn_memory=True,
    )
    backbone2 = TinyBackbone()
    model2 = XinheModel(config2, backbone=backbone2)
    trainer2 = Trainer(model2, config2, loader)
    assert len(trainer2.optimizer.param_groups) == 2


def test_v5a_plugin_lr_multiplier():
    """plugin_lr_multiplier 控制 plugin 组的 LR"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        learning_rate=1e-3, plugin_lr_multiplier=2.0,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=4, seg_len=8, num_segments=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    initial_lrs = [g["initial_lr"] for g in trainer.optimizer.param_groups]
    assert pytest.approx(1e-3 * 2.0, rel=1e-5) in initial_lrs


def test_weighted_loss_equals_unweighted_when_uniform():
    """uniform weights=1 的加权 loss 等于不加权 loss"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)
    model.eval()

    B, T = 2, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (B, T))
    labels = input_ids.clone()
    state = model.init_state(B)

    res_a = model(input_ids, state, labels=labels)

    weights = torch.ones(B, T)
    state2 = model.init_state(B)
    res_b = model(input_ids, state2, labels=labels, weights=weights)

    assert torch.allclose(res_a["loss"], res_b["loss"], atol=1e-5)


def test_weighted_loss_value_token_5x():
    """value token 权重 5x 时, loss 应向 value 位置倾斜"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)
    model.eval()

    B, T = 1, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (B, T))
    labels = input_ids.clone()

    weights = torch.ones(B, T)
    weights[:, T // 2:] = 5.0

    state1 = model.init_state(B)
    res_uniform = model(input_ids, state1, labels=labels, weights=torch.ones(B, T))

    state2 = model.init_state(B)
    res_weighted = model(input_ids, state2, labels=labels, weights=weights)

    assert not torch.allclose(res_uniform["loss"], res_weighted["loss"], atol=1e-6)


def test_weighted_loss_ignores_minus_100():
    """weights=0 的位置 (对应 -100 label) 不贡献 loss"""
    config = XinheConfig(
        hidden_size=32, n_heads=4, head_dim=8,
        read_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)
    model.eval()

    B, T = 1, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (B, T))
    labels = input_ids.clone()
    labels[:, :4] = -100

    weights = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    state = model.init_state(B)
    res = model(input_ids, state, labels=labels, weights=weights)

    assert torch.isfinite(res["loss"])
    state2 = model.init_state(B)
    res_ref = model(input_ids, state2, labels=labels)
    assert torch.allclose(res["loss"], res_ref["loss"], atol=1e-5)
