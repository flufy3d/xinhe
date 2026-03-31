"""
训练入口脚本

用法:
    # 单阶段训练
    python scripts/train.py --config configs/qwen3-0.6b.yaml
    python scripts/train.py --config configs/qwen3-0.6b.yaml --resume checkpoints/xinhe_step_1000.pt

    # 课程学习 (config 中包含 curriculum 段)
    python scripts/train.py --config configs/curriculum_qwen.yaml
    python scripts/train.py --config configs/curriculum_qwen.yaml --from-stage 3_distance
"""
import argparse
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到 path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import (
    ConversationDataset, collate_episodes, ensure_chat_template,
)
from xinhe.training.trainer import Trainer


def load_tokenizer(config: XinheConfig):
    """加载 tokenizer（backbone 通用）"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(config.backbone_model_path).resolve()),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)
    return tokenizer


def make_dataloaders(config, tokenizer):
    """根据 config 创建 train/val DataLoader"""
    train_dataset = ConversationDataset(
        data_path=config.train_path,
        tokenizer=tokenizer,
        segment_length=config.segment_length,
        episode_length=config.episode_length,
    )
    val_dataset = ConversationDataset(
        data_path=config.val_path,
        tokenizer=tokenizer,
        segment_length=config.segment_length,
        episode_length=config.episode_length,
    ) if Path(config.val_path).exists() else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_episodes, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_episodes, num_workers=0,
    ) if val_dataset else None

    return train_loader, val_loader, len(train_dataset), len(val_dataset) if val_dataset else 0


def apply_stage_overrides(base_config: XinheConfig, stage: dict) -> XinheConfig:
    """将课程阶段的 training 参数覆盖到 base config 上"""
    overrides = {}
    training = stage.get("training", {})

    # training 段的字段直接映射到 config
    field_map = {
        "episode_length": "episode_length",
        "tbptt_steps": "tbptt_steps",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "warmup_steps": "warmup_steps",
        "max_steps": "max_steps",
        "early_stop_loss": "early_stop_loss",
        "early_stop_patience": "early_stop_patience",
        "log_every": "log_every",
        "save_every": "save_every",
        "eval_every": "eval_every",
    }
    for yaml_key, field_name in field_map.items():
        if yaml_key in training:
            overrides[field_name] = training[yaml_key]

    return replace(base_config, **overrides)


def generate_stage_data(stage: dict, stage_name: str) -> tuple[str, str]:
    """为课程阶段生成合成数据，返回 (train_path, val_path)"""
    from generate_memory_data import generate_data

    data_cfg = stage.get("data", {})
    out_dir = f"data/curriculum/{stage_name}"

    return generate_data(
        out_dir=out_dir,
        num_train=data_cfg.get("num_train", 5000),
        num_val=data_cfg.get("num_val", 200),
        min_distance=data_cfg.get("min_distance", 1),
        max_distance=data_cfg.get("max_distance", 4),
        max_turns=data_cfg.get("max_turns", 16),
        num_facts=data_cfg.get("num_facts", 1),
        num_fillers=data_cfg.get("num_fillers", 0),
        no_pre_filler=data_cfg.get("no_pre_filler", False),
        max_pre_filler=data_cfg.get("max_pre_filler", 3),
        no_overwrite=data_cfg.get("no_overwrite", False),
        overwrite_ratio=data_cfg.get("overwrite_ratio", 0.4),
        seed=data_cfg.get("seed", 42),
    )


def train_single(config, args):
    """单阶段训练（向后兼容）"""
    tokenizer = load_tokenizer(config)
    train_loader, val_loader, n_train, n_val = make_dataloaders(config, tokenizer)

    if n_train == 0:
        print(f"错误: 训练数据为空，请先生成数据")
        sys.exit(1)

    print(f"训练集: {n_train} episodes | 验证集: {n_val} episodes")

    model = XinheModel(config)
    trainer = Trainer(model, config, train_loader, val_loader)

    resume_path = args.resume or (config.resume_from if config.resume_from else None)
    if resume_path:
        trainer.load_checkpoint(resume_path)
        if args.reset_step:
            trainer.reset_for_new_stage(config, train_loader, val_loader)
            print(f"[课程学习] 已重置 step=0, 优化器和调度器已重建")

    trainer.train()


def train_curriculum(base_config, stages, args):
    """课程学习：按阶段依次训练"""
    tokenizer = load_tokenizer(base_config)

    # 确定从哪个阶段开始
    stage_names = [s["name"] for s in stages]
    start_idx = 0

    if args.from_stage:
        if args.from_stage not in stage_names:
            print(f"错误: 阶段 '{args.from_stage}' 不存在。可用阶段: {stage_names}")
            sys.exit(1)
        start_idx = stage_names.index(args.from_stage)
    else:
        # 自动跳过已完成的阶段
        for i, name in enumerate(stage_names):
            ckpt_path = Path(f"checkpoints/curriculum/{name}.pt")
            if ckpt_path.exists():
                start_idx = i + 1
            else:
                break

    if start_idx >= len(stages):
        print(f"所有 {len(stages)} 个阶段已完成。使用 --from-stage 重跑指定阶段。")
        return

    if start_idx > 0:
        print(f"跳过已完成的阶段: {stage_names[:start_idx]}")

    # 创建模型（只创建一次）
    model = XinheModel(base_config)
    model.setup_device(torch.device(base_config.device))

    # 加载初始权重: --resume 优先，否则加载前一阶段的 checkpoint
    init_ckpt = None
    if args.resume:
        init_ckpt = args.resume
    elif start_idx > 0:
        prev_ckpt = Path(f"checkpoints/curriculum/{stage_names[start_idx - 1]}.pt")
        if prev_ckpt.exists():
            init_ckpt = str(prev_ckpt)

    if init_ckpt:
        ckpt = torch.load(init_ckpt, map_location=base_config.device, weights_only=False)
        model.plugin.load_state_dict(ckpt["plugin_state"])
        from xinhe.model.lora import LoRALinear
        lora_state = ckpt.get("lora_state", {})
        for name, module in model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state:
                    module.lora_B.data = lora_state[f"{name}.lora_B"]
        print(f"[课程学习] 从 {init_ckpt} 加载权重")

    trainer = None

    for i in range(start_idx, len(stages)):
        stage = stages[i]
        stage_name = stage["name"]
        data_cfg = stage.get("data", {})

        print(f"\n{'='*60}")
        print(f"  课程阶段 [{i+1}/{len(stages)}]: {stage_name}")
        print(f"{'='*60}")

        # 准备数据
        generate = data_cfg.get("generate", True)
        if generate:
            print(f"[数据生成]")
            train_path, val_path = generate_stage_data(stage, stage_name)
        else:
            train_path = data_cfg["train_path"]
            val_path = data_cfg.get("val_path", "")

        # 构建本阶段 config
        stage_config = apply_stage_overrides(base_config, stage)
        stage_config = replace(stage_config, train_path=train_path, val_path=val_path)

        # 创建 DataLoader
        train_loader, val_loader, n_train, n_val = make_dataloaders(stage_config, tokenizer)
        print(f"[数据] 训练集: {n_train} | 验证集: {n_val}")
        print(f"[参数] lr={stage_config.learning_rate} batch={stage_config.batch_size} "
              f"ep_len={stage_config.episode_length} max_steps={stage_config.max_steps}")

        # 训练
        if trainer is None:
            trainer = Trainer(model, stage_config, train_loader, val_loader)
        else:
            trainer.reset_for_new_stage(stage_config, train_loader, val_loader)

        trainer.train()

        # 保存阶段 checkpoint
        ckpt_path = f"checkpoints/curriculum/{stage_name}.pt"
        trainer._save_checkpoint(ckpt_path)
        print(f"[阶段完成] {stage_name} → {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  课程学习全部完成! 共 {len(stages)} 个阶段")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="心核 (Xinhe) 训练")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--reset-step", action="store_true", help="恢复权重但重置 step 和优化器")
    parser.add_argument("--from-stage", type=str, default=None, help="课程学习: 从指定阶段开始")
    args = parser.parse_args()

    # 加载配置
    config, curriculum = XinheConfig.from_yaml(args.config)
    print(f"=== 心核 (Xinhe) 训练 ===")
    print(f"Backbone: {config.backbone_type} ({config.backbone_model_path}) | 设备: {config.device} | 精度: {config.dtype}")
    print(f"状态 token: {config.n_state} | 维度: {config.state_dim}")
    print(f"LoRA rank: {config.lora_rank} | 目标模块: {config.lora_target_modules}")

    if curriculum:
        print(f"课程学习: {len(curriculum)} 个阶段")
        train_curriculum(config, curriculum, args)
    else:
        train_single(config, args)


if __name__ == "__main__":
    main()
