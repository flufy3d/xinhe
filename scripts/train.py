"""
训练入口脚本

用法:
    python scripts/train.py --config configs/qwen3-0.6b.yaml
    python scripts/train.py --config configs/qwen3-0.6b.yaml --resume checkpoints/xinhe_step_1000.pt
"""
import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(description="心核 (Xinhe) 训练")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    args = parser.parse_args()

    # 加载配置
    config = XinheConfig.from_yaml(args.config)
    print(f"=== 心核 (Xinhe) 训练 ===")
    print(f"Backbone: {config.backbone_type} ({config.backbone_model_path}) | 设备: {config.device} | 精度: {config.dtype}")
    print(f"状态 token: {config.n_state} | 维度: {config.state_dim}")
    print(f"LoRA rank: {config.lora_rank} | 目标模块: {config.lora_target_modules}")

    # 加载 tokenizer
    tokenizer = load_tokenizer(config)
    print(f"Tokenizer 已加载, vocab_size={getattr(tokenizer, 'vocab_size', '?')}")

    # 创建数据集
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

    if len(train_dataset) == 0:
        print(f"错误: 训练数据为空，请先生成数据:")
        print(f"  python scripts/generate_memory_data.py")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
        num_workers=0,
    ) if val_dataset else None

    print(f"训练集: {len(train_dataset)} episodes | 验证集: {len(val_dataset) if val_dataset else 0} episodes")

    # 创建模型
    model = XinheModel(config)
    print(f"模型已创建")

    # 创建 trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # 从 checkpoint 恢复
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
