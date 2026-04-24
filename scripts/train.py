"""
训练入口脚本 (v5a: 移除 4B 迁移支持，专注 0.8B)

用法:
    # 单阶段训练
    python scripts/train.py --config configs/qwen3.5-0.8b.yaml
    python scripts/train.py --config configs/qwen3.5-0.8b.yaml --resume checkpoints/xinhe_step_1000.pt

    # 课程学习 (config 中包含 curriculum 段)
    python scripts/train.py --config configs/curriculum_qwen3.5-0.8b.yaml
    python scripts/train.py --config configs/curriculum_qwen3.5-0.8b.yaml --from-stage 3_entity_SAME
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
        "segment_length": "segment_length",
        "episode_length": "episode_length",
        "tbptt_steps": "tbptt_steps",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "gradient_checkpointing": "gradient_checkpointing",
        "learning_rate": "learning_rate",
        "plugin_lr_multiplier": "plugin_lr_multiplier",
        "turn_lr_multiplier": "turn_lr_multiplier",
        "turn_phase_max": "turn_phase_max",
        "turn_phase_temperature": "turn_phase_temperature",
        "freeze_lora": "freeze_lora",
        "freeze_fact": "freeze_fact",
        "freeze_turn": "freeze_turn",
        "suppress_turn_read": "suppress_turn_read",
        "suppress_fact_read": "suppress_fact_read",
        "lora_reset": "lora_reset",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "warmup_steps": "warmup_steps",
        "max_steps": "max_steps",
        "early_stop_loss": "early_stop_loss",
        "early_stop_patience": "early_stop_patience",
        "early_stop_value": "early_stop_value",
        "early_stop_tell": "early_stop_tell",
        "use_joint_early_stop": "use_joint_early_stop",
        "early_stop_world_qa": "early_stop_world_qa",
        "early_stop_refusal": "early_stop_refusal",
        "early_stop_compositional": "early_stop_compositional",
        "early_stop_pronoun": "early_stop_pronoun",
        "early_stop_disentangle": "early_stop_disentangle",
        "early_stop_rapid_overwrite": "early_stop_rapid_overwrite",
        "early_stop_decay": "early_stop_decay",
        "log_every": "log_every",
        "save_every": "save_every",
        "eval_every": "eval_every",
        # v5c: 每阶段可调 Delta Rule 配置
        "n_heads": "n_heads",
        "head_dim": "head_dim",
        "read_scale_init": "read_scale_init",
        "beta_bias_init": "beta_bias_init",
    }
    for yaml_key, field_name in field_map.items():
        if yaml_key in training:
            overrides[field_name] = training[yaml_key]

    return replace(base_config, **overrides)


def generate_stage_data(stage: dict, stage_name: str) -> tuple[str, str]:
    """为课程阶段生成数据，自动分发到 memory/persona 生成器"""
    from generate_data import generate_stage_data as _gen
    return _gen(stage, stage_name)


def train_single(config, args):
    """单阶段训练（向后兼容）"""
    tokenizer = load_tokenizer(config)
    train_loader, val_loader, n_train, n_val = make_dataloaders(config, tokenizer)

    if n_train == 0:
        print(f"错误: 训练数据为空，请先生成数据")
        sys.exit(1)

    print(f"训练集: {n_train} episodes | 验证集: {n_val} episodes")

    model = XinheModel(config)
    trainer = Trainer(model, config, train_loader, val_loader, pad_token_id=tokenizer.pad_token_id)

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
        if "fact_plugin_state" not in ckpt:
            raise RuntimeError(
                f"checkpoint {init_ckpt} 缺少 'fact_plugin_state' 键。v6 不再兼容旧 'plugin_state' 格式，请从零重训。"
            )
        result = model.fact_interface.load_state_dict(ckpt["fact_plugin_state"], strict=False)
        if result.missing_keys:
            print(f"  注意: fact checkpoint 缺少 {result.missing_keys}，使用默认初始化")

        # TurnInterface 加载（0a → 0b 过渡时 turn_plugin_state 可能为 None，保持随机初始化静默启动）
        turn_iface = getattr(model, "turn_interface", None)
        if turn_iface is not None:
            turn_state = ckpt.get("turn_plugin_state")
            if turn_state is not None:
                tres = turn_iface.load_state_dict(turn_state, strict=False)
                if tres.missing_keys:
                    print(f"  注意: turn checkpoint 缺少 {tres.missing_keys}，使用默认初始化")
            else:
                print(f"  [info] ckpt {init_ckpt} 无 turn_plugin_state，turn 保持随机初始化 + 静默 read_scale")

        # persona 统一训练: 可以加载 plugin 但 reset LoRA（新 LoRA 从随机 kaiming_A + zero_B 开始）
        first_stage = stages[start_idx]
        first_stage_training = first_stage.get("training", {})
        lora_reset = first_stage_training.get("lora_reset", False)

        if lora_reset:
            print(f"[课程学习] lora_reset=True，跳过 LoRA 加载（保持新随机初始化）")
        else:
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

        # 准备数据 (统一分发: memory/persona 自动识别)
        print(f"[数据生成] type={data_cfg.get('type', 'memory')}")
        train_path, val_path = generate_stage_data(stage, stage_name)

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
            trainer = Trainer(model, stage_config, train_loader, val_loader, pad_token_id=tokenizer.pad_token_id)
            # 首阶段: 如果 --resume 指向的是 mid-stage xinhe_step_* checkpoint,
            # 完整恢复 optimizer + scheduler + global_step, 避免 LR 重新 warmup 震崩。
            # 通过 checkpoint 是否有 optimizer_state 判断。
            if args.resume and "xinhe_step_" in args.resume:
                ckpt_has_opt = torch.load(
                    args.resume, map_location="cpu", weights_only=False,
                ).get("optimizer_state") is not None
                if ckpt_has_opt:
                    trainer.load_checkpoint(args.resume)
                    print(f"[resume] 已恢复 optimizer+scheduler 状态, global_step={trainer.global_step}")
        else:
            trainer.reset_for_new_stage(stage_config, train_loader, val_loader)
        trainer.current_stage_name = stage_name

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
    mem_size = config.n_heads * config.head_dim * config.head_dim
    print(f"状态 W: H={config.n_heads} d_k=d_v={config.head_dim} "
          f"(每样本 {mem_size} floats) | hidden↔proj: {config.hidden_size}")
    print(f"LoRA rank: {config.lora_rank} | 目标模块: {config.lora_target_modules}")

    if curriculum:
        print(f"课程学习: {len(curriculum)} 个阶段")
        train_curriculum(config, curriculum, args)
    else:
        train_single(config, args)


if __name__ == "__main__":
    main()
