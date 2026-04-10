"""
Checkpoint 工具 — 保存/加载模型 + 状态
"""
from pathlib import Path

import torch

from ..model.state_plugin import CORE_PARAM_PREFIXES


def extract_plugin_core(source, device="cpu") -> dict:
    """
    从 checkpoint 提取 backbone 无关的 plugin 核心参数 (用于跨 backbone 迁移)。

    参数:
        source: str/Path (checkpoint 路径) 或 dict (已加载的 checkpoint)
        device: 目标设备

    返回:
        dict: 只含核心参数的 state_dict, 可用于 plugin.load_state_dict(core, strict=False)
    """
    if isinstance(source, (str, Path)):
        source = torch.load(source, map_location=device, weights_only=False)
    plugin_state = source.get("plugin_state", {})
    return {k: v.to(device) if hasattr(v, 'to') else v
            for k, v in plugin_state.items()
            if any(k.startswith(prefix) for prefix in CORE_PARAM_PREFIXES)}


def save_state(state: torch.Tensor, path: str):
    """保存持久状态到文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.cpu(), path)


def load_state(path: str, device: torch.device = None) -> torch.Tensor:
    """从文件加载持久状态"""
    state = torch.load(path, map_location=device or "cpu", weights_only=True)
    return state


def save_checkpoint(model, optimizer, scheduler, global_step: int, path: str,
                    curriculum_stage: str = None):
    """保存完整 checkpoint"""
    from ..model.lora import LoRALinear

    plugin_state = model.plugin.state_dict()

    lora_state = {}
    for name, module in model.backbone.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()

    checkpoint = {
        "global_step": global_step,
        "plugin_state": plugin_state,
        "lora_state": lora_state,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "curriculum_stage": curriculum_stage,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device=None):
    """加载完整 checkpoint"""
    from ..model.lora import LoRALinear

    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)

    result = model.plugin.load_state_dict(checkpoint["plugin_state"], strict=False)
    if result.missing_keys:
        print(f"  注意: checkpoint 缺少 {result.missing_keys}，使用默认初始化")

    lora_state = checkpoint.get("lora_state", {})
    for name, module in model.backbone.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in lora_state:
                module.lora_A.data = lora_state[f"{name}.lora_A"].to(device or "cpu")
            if f"{name}.lora_B" in lora_state:
                module.lora_B.data = lora_state[f"{name}.lora_B"].to(device or "cpu")

    if optimizer and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint.get("global_step", 0)
