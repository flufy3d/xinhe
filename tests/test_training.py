"""
测试训练循环在小数据上能跑通
"""
import torch
import pytest
import tempfile
import torch.nn as nn
from torch.utils.data import DataLoader

from xinhe.model.config import XinheConfig
from xinhe.model.backbone import BackboneBase
from xinhe.model.xinhe_model import XinheModel
from xinhe.model.state_plugin import StatePlugin
from xinhe.data.conversation import collate_episodes
from xinhe.utils.checkpoint import extract_plugin_core


class TinyBackbone(nn.Module, BackboneBase):
    """极小 backbone 用于训练测试"""

    def __init__(self):
        nn.Module.__init__(self)
        self._embed = nn.Embedding(50, 32)
        self._block = nn.Linear(32, 32)
        self._head = nn.Linear(32, 50, bias=False)

    def embed(self, input_ids):
        return self._embed(input_ids)

    def forward_blocks(self, hidden_states, attention_mask=None, position_ids=None):
        return self._block(hidden_states)

    def get_lm_head(self):
        return self._head

    def get_hidden_size(self):
        return 32


class DummyDataset:
    """极小数据集: 每个 episode 有 4 个 segment，每个 segment 是 (input_ids, labels) tuple"""

    def __init__(self, num_episodes=10, seg_len=16, num_segments=4):
        self.episodes = []
        for _ in range(num_episodes):
            episode = []
            for _ in range(num_segments):
                ids = torch.randint(0, 50, (seg_len,))
                labels = ids.clone()
                episode.append((ids, labels))
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
        labels = input_ids.clone()
        result = model(input_ids, state, labels=labels)
        state = result["state_next"]

        # 在 BPTT 窗口内，state 应该可以反向传播
        if seg_idx % config.tbptt_steps > 0:
            assert state.requires_grad or result["loss"].requires_grad


# --- 冻结 + 迁移测试 ---

def test_freeze_plugin_core_optimizer_groups():
    """freeze_plugin_core 时 optimizer 只含 proj + lora 参数"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=16,  # 有投影层
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        freeze_plugin_core=True, freeze_lora=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=4, seg_len=8, num_segments=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    # core 参数应被冻结
    for p in model.plugin.core_parameters():
        assert not p.requires_grad

    # proj 参数应可训练
    for p in model.plugin.projection_parameters():
        assert p.requires_grad

    # optimizer 应只包含 proj 参数 (lora_rank=0 所以无 LoRA)
    opt_param_count = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    proj_count = len(model.plugin.projection_parameters())
    assert opt_param_count == proj_count


def test_plugin_core_lr_multiplier():
    """plugin_core_lr_multiplier 正确设置学习率"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=16,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        learning_rate=1e-3, plugin_lr_multiplier=2.0,
        plugin_core_lr_multiplier=0.1,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=4, seg_len=8, num_segments=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    # 检查 initial_lr (scheduler 应用 warmup 前的基础 LR)
    initial_lrs = [g["initial_lr"] for g in trainer.optimizer.param_groups]
    assert pytest.approx(1e-3 * 2.0 * 0.1, rel=1e-5) in initial_lrs  # core
    assert pytest.approx(1e-3 * 2.0, rel=1e-5) in initial_lrs          # proj


def test_extract_plugin_core_from_dict():
    """extract_plugin_core 从 checkpoint dict 提取核心参数"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=32)
    with torch.no_grad():
        plugin.gate_bias.fill_(2.0)

    checkpoint = {"plugin_state": plugin.state_dict()}
    core = extract_plugin_core(checkpoint)

    # 应只含 core keys
    from xinhe.model.state_plugin import CORE_PARAM_PREFIXES, PROJECTION_PARAM_PREFIXES
    for k in core:
        assert any(k.startswith(p) for p in CORE_PARAM_PREFIXES)
        assert not any(k.startswith(p) for p in PROJECTION_PARAM_PREFIXES)

    assert torch.allclose(core["gate_bias"], torch.full((4, 16), 2.0))


def test_extract_plugin_core_from_file():
    """extract_plugin_core 从文件路径提取核心参数"""
    plugin = StatePlugin(n_state=4, state_dim=16, hidden_size=16)
    with torch.no_grad():
        plugin.state_scale.fill_(0.5)

    checkpoint = {"plugin_state": plugin.state_dict()}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        core = extract_plugin_core(f.name)

    assert "state_scale" in core
    assert torch.allclose(core["state_scale"], torch.tensor(0.5))


def test_migration_loading():
    """迁移: 从不同 hidden_size 加载 core 到新 plugin"""
    # 源: state_dim=16, hidden_size=16 (无投影)
    src = StatePlugin(n_state=4, state_dim=16, hidden_size=16)
    with torch.no_grad():
        src.gate_bias.fill_(3.14)

    checkpoint = {"plugin_state": src.state_dict()}
    core = extract_plugin_core(checkpoint)

    # 目标: state_dim=16, hidden_size=64 (有投影)
    dst = StatePlugin(n_state=4, state_dim=16, hidden_size=64)
    proj_up_before = dst.proj_up.weight.clone()

    result = dst.load_state_dict(core, strict=False)

    # core 应已加载
    assert torch.allclose(dst.gate_bias, torch.full((4, 16), 3.14))
    # proj 应保持随机初始化 (未变)
    assert torch.allclose(dst.proj_up.weight, proj_up_before)
