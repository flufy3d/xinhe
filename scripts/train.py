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
# import xinhe(包 init 里关掉 JIT fuser,见 xinhe/__init__.py)

from xinhe.config import validate_stage_config
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import (
    ConversationDataset, MixedConversationDataset,
    collate_episodes, ensure_chat_template,
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


def make_dataloaders(config, tokenizer, stage_data: dict | None = None):
    """根据 config 创建 train/val DataLoader。

    stage_data: stage["data"] 段。kind="mix_dynamic" 时走 MixedConversationDataset
                动态从多源抽样,不落 mix 物理文件;否则走 ConversationDataset 单文件。
    v9 mode 默认 value_weight_cap=1.0(NeuralMemory fast-weights 不靠 weight reinforcement)。
    """
    cap = getattr(config, "value_weight_cap", 1.0)
    is_mix_dynamic = bool(stage_data) and stage_data.get("kind") == "mix_dynamic"

    if is_mix_dynamic:
        sources = stage_data["sources"]
        val_sources = stage_data.get("val_sources", sources)
        seed = int(stage_data.get("seed", 42))
        n_train = int(stage_data.get("num_train", 5000))
        n_val = int(stage_data.get("num_val", 50))

        train_dataset = MixedConversationDataset(
            sources=sources,
            n_samples=n_train,
            seed=seed,
            tokenizer=tokenizer,
            turn_max_tokens=config.turn_max_tokens,
            max_turns_per_episode=config.max_turns_per_episode,
            value_weight_cap=cap,
            cache_slot="mix_train",
        )
        val_dataset = (
            MixedConversationDataset(
                sources=val_sources,
                n_samples=n_val,
                seed=seed + 1,
                tokenizer=tokenizer,
                turn_max_tokens=config.turn_max_tokens,
                max_turns_per_episode=config.max_turns_per_episode,
                value_weight_cap=cap,
                cache_slot="mix_val",
            )
            if n_val > 0 else None
        )
    else:
        train_dataset = ConversationDataset(
            data_path=config.train_path,
            tokenizer=tokenizer,
            turn_max_tokens=config.turn_max_tokens,
            max_turns_per_episode=config.max_turns_per_episode,
            value_weight_cap=cap,
            cache_slot="single_train",
        )
        val_dataset = ConversationDataset(
            data_path=config.val_path,
            tokenizer=tokenizer,
            turn_max_tokens=config.turn_max_tokens,
            max_turns_per_episode=config.max_turns_per_episode,
            value_weight_cap=cap,
            cache_slot="single_val",
        ) if Path(config.val_path).exists() else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_episodes,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_episodes,
        num_workers=2, pin_memory=True, persistent_workers=True,
    ) if val_dataset else None

    return train_loader, val_loader, len(train_dataset), len(val_dataset) if val_dataset else 0


def apply_stage_overrides(base_config: XinheConfig, stage: dict) -> XinheConfig:
    """将课程阶段的 training 参数覆盖到 base config 上。

    依赖前置：caller 已对 stage 调用 validate_stage_config，保证 training 含
    turn_max_tokens / max_turns_per_episode / tbptt_turns（tbptt_turns 可能派生）。"""
    overrides = {}
    training = stage.get("training", {})

    # training 段的字段直接映射到 config
    field_map = {
        "turn_max_tokens": "turn_max_tokens",
        "max_turns_per_episode": "max_turns_per_episode",
        "tbptt_turns": "tbptt_turns",
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "gradient_checkpointing": "gradient_checkpointing",
        "per_segment_checkpoint": "per_segment_checkpoint",
        "learning_rate": "learning_rate",
        "plugin_lr_multiplier": "plugin_lr_multiplier",
        "freeze_alpha": "freeze_alpha",
        "freeze_gate_q": "freeze_gate_q",
        "value_weight_cap": "value_weight_cap",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "warmup_steps": "warmup_steps",
        "max_steps": "max_steps",
        "early_stop_loss": "early_stop_loss",
        "early_stop_patience": "early_stop_patience",
        "early_stop_value": "early_stop_value",
        "early_stop_tell": "early_stop_tell",
        "use_joint_early_stop": "use_joint_early_stop",
        "early_stop": "early_stop",     # dict-form 通用早停
        "log_every": "log_every",
        "save_every": "save_every",
        "eval_every": "eval_every",
        # v9: 每阶段可调 NeuralMemoryPair 配置
        "n_heads": "n_heads",
        "head_dim": "head_dim",
        "hippo_retention": "hippo_retention",
        "hippo_base_lr": "hippo_base_lr",
        "phase": "phase",
    }
    for yaml_key, field_name in field_map.items():
        if yaml_key in training:
            overrides[field_name] = training[yaml_key]

    # data.val_sets 也要一起传到 config(event_eval 消费)
    data_block = stage.get("data", {})
    if "val_sets" in data_block or "val_sets" in stage:
        overrides["val_sets"] = stage.get("val_sets") or data_block.get("val_sets") or []
    elif "val_sets" in stage:
        overrides["val_sets"] = stage["val_sets"]

    return replace(base_config, **overrides)


def generate_stage_data(stage: dict, stage_name: str) -> tuple[str, str]:
    """通过 generate_data.run_stage 调用对应 kind 的 generator/mix。

    Returns: (train_path, val_path) — 由 stage.data.out_dir 推导;
             kind=mix_dynamic 时返回 sentinel,实际抽样在 make_dataloaders 内做。

    特例:kind=novel 不在训练时触发生成(novel_path 必须从 CLI 传给
    generate_data.py)。仅检查文件存在,缺失打 hint 后 sys.exit。
    """
    kind = stage["data"].get("kind", "skeleton")

    if kind == "mix_dynamic":
        # 动态混合:不预生成文件,sources 在 MixedConversationDataset 内读
        # 仅校验所有 source 文件存在
        for src in stage["data"].get("sources", []):
            if not Path(src["path"]).exists():
                print(
                    f"\n[mix_dynamic] source 文件不存在: {src['path']}\n"
                    f"  Hint:先生成对应 source(eg. novel/dialog/skeleton)。\n"
                )
                sys.exit(1)
        return "<MIX_DYNAMIC>", "<MIX_DYNAMIC>"

    out_dir = Path(stage["data"].get("out_dir", f"data/{kind}"))
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    if kind == "novel":
        missing = [p for p in (train_path, val_path) if not p.exists()]
        if missing:
            print(
                f"\n[novel] 数据未生成: {[str(p) for p in missing]}\n"
                f"  训练入口不会自动生成 novel 数据 (path 只能从 CLI 传)。请先运行:\n"
                f"    python scripts/generate_data.py --config <novel-config.yaml> "
                f"--stage {stage_name} --novel-path /path/to/novel.txt\n"
            )
            sys.exit(1)
        return str(train_path), str(val_path)

    from scripts.generate_data import run_stage as _run_stage
    _run_stage(stage, force=False)
    return str(train_path), str(val_path)


def train_single(config, args):
    """单阶段训练（向后兼容）"""
    tokenizer = load_tokenizer(config)
    train_loader, val_loader, n_train, n_val = make_dataloaders(config, tokenizer)

    if n_train == 0:
        print(f"错误: 训练数据为空,请先生成数据")
        sys.exit(1)

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
        if "memory_pair_state" not in ckpt:
            raise RuntimeError(
                f"checkpoint {init_ckpt} 缺少 'memory_pair_state' 键。v9 不兼容 v8 hippocampus_state,请从零重训。"
            )
        model.memory.load_state_dict(ckpt["memory_pair_state"], strict=True)
        print(f"[课程学习] 从 {init_ckpt} 加载权重")

    trainer = None

    for i in range(start_idx, len(stages)):
        stage = stages[i]
        stage_name = stage["name"]
        data_cfg = stage.get("data", {})

        print(f"\n{'='*60}")
        print(f"  课程阶段 [{i+1}/{len(stages)}]: {stage_name}")
        print(f"{'='*60}")

        # 启动期校验 + 派生 tbptt_turns（写回 stage["training"]）
        # 配错在这里立刻抛 ConfigError 带 Hint，不让训练静默跑被截断的数据
        validate_stage_config(stage_name, stage)

        train_path, val_path = generate_stage_data(stage, stage_name)

        stage_config = apply_stage_overrides(base_config, stage)
        stage_config = replace(stage_config, train_path=train_path, val_path=val_path)

        train_loader, val_loader, n_train, n_val = make_dataloaders(
            stage_config, tokenizer, stage_data=stage.get("data"),
        )
        print(f"  lr={stage_config.learning_rate} batch={stage_config.batch_size} "
              f"max_turns={stage_config.max_turns_per_episode} max_steps={stage_config.max_steps}")

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
    print(f"NeuralMemoryPair: H={config.n_heads} d_head={config.head_dim} "
          f"chunk={config.mem_chunk_size} phase={config.phase}")
    print(f"  Hippo: depth={config.hippo_mlp_depth} retention={config.hippo_retention} lr={config.hippo_base_lr} (TTT inner SGD)")
    print(f"  Neo:   depth={config.neo_mlp_depth} expansion={config.neo_mlp_expansion} (普通 MLP + 标准 backprop)")
    n_per_layer = getattr(config, "n_persistent_per_layer", 0)
    n_mem = getattr(config, "n_mem_tokens", 0)
    lora_rank = getattr(config, "lora_rank", 0)
    bits = []
    if n_per_layer > 0:
        bits.append(f"per-layer K/V={n_per_layer}")
    if n_mem > 0:
        bits.append(f"fresh_mem/turn={n_mem}")
    if lora_rank > 0:
        bits.append(f"LoRA(rank={lora_rank})")
    if bits:
        print(f"  MAC v9.5: " + " | ".join(bits))

    if curriculum:
        print(f"课程学习: {len(curriculum)} 个阶段")
        train_curriculum(config, curriculum, args)
    else:
        train_single(config, args)


if __name__ == "__main__":
    main()
