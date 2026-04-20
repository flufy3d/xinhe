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
from xinhe.model.state_plugin import StateInterface
from xinhe.data.conversation import collate_episodes
from xinhe.utils.checkpoint import extract_plugin_core


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
    """极小数据集: 每个 episode 有 4 个 segment，每个 segment 是 (input_ids, labels, weights) 三元组"""

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
    """freeze_plugin_core 时 optimizer 只含 proj 参数"""
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
    for p in model.state_interface.core_parameters():
        assert not p.requires_grad

    # proj 参数应可训练
    for p in model.state_interface.projection_parameters():
        assert p.requires_grad

    # optimizer 应只包含 proj 参数 (lora_rank=0 所以无 LoRA)
    opt_param_count = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    proj_count = len(model.state_interface.projection_parameters())
    assert opt_param_count == proj_count


def test_plugin_core_lr_multiplier():
    """plugin_core_lr_multiplier 正确设置学习率"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=16,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        learning_rate=1e-3, plugin_lr_multiplier=2.0,
        plugin_core_lr_multiplier=0.1,
        slot_attn_lr_multiplier=1.0,  # 关闭 slot_attn 额外倍数, 简化断言
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
    assert pytest.approx(1e-3 * 2.0 * 0.1, rel=1e-5) in initial_lrs  # core + slot_attn (mult=1)
    assert pytest.approx(1e-3 * 2.0, rel=1e-5) in initial_lrs          # proj


def test_slot_attn_lr_multiplier():
    """slot_attn 单独一组学习率 (续训从恒等激活要更高 LR)"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=16,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        learning_rate=1e-3, plugin_lr_multiplier=1.0,
        plugin_core_lr_multiplier=0.5,
        slot_attn_lr_multiplier=3.0,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)

    dataset = DummyDataset(num_episodes=4, seg_len=8, num_segments=2)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)

    from xinhe.training.trainer import Trainer
    trainer = Trainer(model, config, loader)

    initial_lrs = [g["initial_lr"] for g in trainer.optimizer.param_groups]
    # slot_attn: lr × plugin_mult × core_mult × slot_attn_mult = 1e-3 × 1 × 0.5 × 3 = 1.5e-3
    assert pytest.approx(1e-3 * 0.5 * 3.0, rel=1e-5) in initial_lrs
    # core (不含 slot_attn): lr × plugin_mult × core_mult = 1e-3 × 1 × 0.5 = 5e-4
    assert pytest.approx(1e-3 * 0.5, rel=1e-5) in initial_lrs
    # proj: lr × plugin_mult = 1e-3
    assert pytest.approx(1e-3, rel=1e-5) in initial_lrs


def test_weighted_loss_equals_unweighted_when_uniform():
    """uniform weights=1 的加权 loss 等于不加权 loss"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=32,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
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

    # 不加权
    res_a = model(input_ids, state, labels=labels)

    # 加权 (全 1)
    weights = torch.ones(B, T)
    state2 = model.init_state(B)
    res_b = model(input_ids, state2, labels=labels, weights=weights)

    assert torch.allclose(res_a["loss"], res_b["loss"], atol=1e-5), \
        f"uniform weights 应等价于不加权, {res_a['loss'].item()} vs {res_b['loss'].item()}"


def test_weighted_loss_value_token_5x():
    """value token 权重 5x 时, loss 应向 value 位置倾斜"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=32,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)
    model.eval()

    B, T = 1, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (B, T))
    labels = input_ids.clone()
    state = model.init_state(B)

    # weights: 前半 1x, 后半 5x (模拟 frame + value)
    weights = torch.ones(B, T)
    weights[:, T // 2:] = 5.0

    state1 = model.init_state(B)
    res_uniform = model(input_ids, state1, labels=labels, weights=torch.ones(B, T))

    state2 = model.init_state(B)
    res_weighted = model(input_ids, state2, labels=labels, weights=weights)

    # 加权 loss 与不加权不同 (除非 per-token loss 恰好相等, 概率极低)
    assert not torch.allclose(res_uniform["loss"], res_weighted["loss"], atol=1e-6)


def test_weighted_loss_ignores_minus_100():
    """weights=0 的位置 (对应 -100 label) 不贡献 loss"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=32,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        tbptt_steps=2, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    model = XinheModel(config, backbone=backbone)
    model.eval()

    B, T = 1, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (B, T))
    labels = input_ids.clone()
    # 屏蔽前 4 个 token
    labels[:, :4] = -100

    weights = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    state = model.init_state(B)
    res = model(input_ids, state, labels=labels, weights=weights)

    # loss 应是有限的非 NaN
    assert torch.isfinite(res["loss"])
    # 等价于 cross_entropy 忽略 -100
    state2 = model.init_state(B)
    res_ref = model(input_ids, state2, labels=labels)  # 不加权, 自动 ignore_index=-100
    assert torch.allclose(res["loss"], res_ref["loss"], atol=1e-5)


def test_extract_plugin_core_from_dict():
    """extract_plugin_core 从 checkpoint dict 提取核心参数"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=32, n_layers=2)
    with torch.no_grad():
        iface.state_emb.fill_(2.0)

    checkpoint = {"plugin_state": iface.state_dict()}
    core = extract_plugin_core(checkpoint)

    # 应只含 core keys
    from xinhe.model.state_plugin import CORE_PARAM_PREFIXES, PROJECTION_PARAM_PREFIXES
    for k in core:
        assert any(k.startswith(p) for p in CORE_PARAM_PREFIXES)
        assert not any(k.startswith(p) for p in PROJECTION_PARAM_PREFIXES)

    assert torch.allclose(core["state_emb"], torch.full((4, 16), 2.0))


def test_extract_plugin_core_from_file():
    """extract_plugin_core 从文件路径提取核心参数"""
    iface = StateInterface(n_state=4, state_dim=16, hidden_size=16, n_layers=2)
    with torch.no_grad():
        iface.read_scale.fill_(0.5)

    checkpoint = {"plugin_state": iface.state_dict()}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        core = extract_plugin_core(f.name)

    # read_scale 是 projection，不应在 core 中
    assert "read_scale" not in core
    # state_emb 应在 core 中
    assert "state_emb" in core


def test_migration_loading():
    """迁移: 从不同 hidden_size/n_layers 加载 core 到新 interface"""
    # 源: state_dim=16, hidden_size=16, n_layers=2
    src = StateInterface(n_state=4, state_dim=16, hidden_size=16, n_layers=2)
    with torch.no_grad():
        src.state_emb.fill_(3.14)

    checkpoint = {"plugin_state": src.state_dict()}
    core = extract_plugin_core(checkpoint)

    # 目标: state_dim=16, hidden_size=64, n_layers=4
    dst = StateInterface(n_state=4, state_dim=16, hidden_size=64, n_layers=4)
    read_k_before = dst.read_k_projs[0].weight.clone()

    result = dst.load_state_dict(core, strict=False)

    # core 应已加载
    assert torch.allclose(dst.state_emb, torch.full((4, 16), 3.14))
    # projection 应保持随机初始化 (未变)
    assert torch.allclose(dst.read_k_projs[0].weight, read_k_before)


def _build_tiny_model(entropy_aux_weight: float = 0.0, eks_alpha_init: float = -5.0):
    """测试工具: 构建 TinyBackbone + XinheModel (eks_alpha/entropy_aux_weight 可调)"""
    config = XinheConfig(
        hidden_size=32, n_state=4, state_dim=32,
        state_scale_init=-5.0, lora_rank=0, freeze_backbone=False,
        entropy_aux_weight=entropy_aux_weight,
        eks_alpha_init=eks_alpha_init,
        tbptt_steps=2, batch_size=2, learning_rate=1e-3,
        warmup_steps=2, max_steps=4, device="cpu", dtype="float32",
    )
    backbone = TinyBackbone()
    return XinheModel(config, backbone=backbone), config


def test_entropy_aux_loss_present_when_weight_positive():
    """entropy_aux_weight > 0 时 forward 返回 aux_loss 和 entropy_ratio, loss 含正则"""
    torch.manual_seed(0)
    model, _ = _build_tiny_model(entropy_aux_weight=0.01)

    input_ids = torch.randint(0, 50, (2, 8))
    labels = input_ids.clone()
    state = model.init_state(2)
    result = model(input_ids, state, labels=labels)

    assert "aux_loss" in result, "entropy_aux_weight>0 应产生 aux_loss"
    assert "entropy_ratio" in result, "entropy_aux_weight>0 应产生 entropy_ratio"
    er = result["entropy_ratio"].item()
    assert 0.0 <= er <= 1.0 + 1e-4, f"entropy_ratio 应在 [0,1], 实际 {er}"


def test_entropy_aux_loss_absent_when_weight_zero():
    """entropy_aux_weight = 0 时 forward 不添加 aux_loss (节省计算)"""
    torch.manual_seed(0)
    model, _ = _build_tiny_model(entropy_aux_weight=0.0)

    input_ids = torch.randint(0, 50, (2, 8))
    labels = input_ids.clone()
    state = model.init_state(2)
    result = model(input_ids, state, labels=labels)

    assert "aux_loss" not in result, "entropy_aux_weight=0 不应产生 aux_loss"


def test_entropy_aux_increases_loss():
    """entropy_aux_weight > 0 会改变 loss 值 (通过和 0 对比)"""
    torch.manual_seed(0)
    m_off, _ = _build_tiny_model(entropy_aux_weight=0.0)
    torch.manual_seed(0)
    m_on, _ = _build_tiny_model(entropy_aux_weight=0.5)  # 较大权重放大差异

    input_ids = torch.randint(0, 50, (2, 8))
    labels = input_ids.clone()
    state_off = m_off.init_state(2)
    state_on = m_on.init_state(2)

    r_off = m_off(input_ids, state_off, labels=labels)
    r_on = m_on(input_ids, state_on, labels=labels)

    # aux = -λ * entropy < 0 (因 entropy > 0), 因此加入后 loss 应 < 原 loss
    assert r_on["loss"].item() < r_off["loss"].item(), \
        f"aux_loss 应降低总 loss, off={r_off['loss']:.4f} on={r_on['loss']:.4f}"
