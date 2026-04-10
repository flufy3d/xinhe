"""
测试 XinheConfig 加载与 yaml 继承
"""
import pytest
import tempfile
from pathlib import Path

from xinhe.model.config import XinheConfig


def test_from_yaml_qwen():
    """qwen3.5-0.8b.yaml 能正确加载并继承 base"""
    config, _ = XinheConfig.from_yaml("configs/qwen3.5-0.8b.yaml")
    assert config.backbone_type == "qwen"
    assert config.hidden_size == 1024
    assert config.state_dim == 1024
    # 继承自 base.yaml
    assert config.tbptt_steps == 8
    assert config.learning_rate == 3e-4


def test_from_yaml_4b():
    """qwen3.5-4b.yaml state_dim 已解耦"""
    config, _ = XinheConfig.from_yaml("configs/qwen3.5-4b.yaml")
    assert config.hidden_size == 2560
    assert config.state_dim == 1024  # 解耦: state_dim < hidden_size


def test_yaml_override():
    """子配置能覆盖 base 的值"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "base.yaml"
        base.write_text(
            "training:\n  batch_size: 4\n  learning_rate: 3.0e-4\n"
            "state:\n  n_state: 32\n  state_dim: 1024\n"
        )
        child = Path(tmpdir) / "child.yaml"
        child.write_text(
            "base: base.yaml\ntraining:\n  batch_size: 8\n"
        )
        config, _ = XinheConfig.from_yaml(str(child))
        assert config.batch_size == 8         # 被覆盖
        assert config.learning_rate == 3e-4   # 继承自 base


def test_backbone_type_default():
    """默认 backbone_type 是 qwen"""
    config = XinheConfig()
    assert config.backbone_type == "qwen"
