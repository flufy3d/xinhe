"""
训练入口脚本

用法:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --resume checkpoints/xinhe_step_1000.pt
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
from xinhe.data.conversation import ConversationDataset, SyntheticMemoryDataset, collate_episodes
from xinhe.training.trainer import Trainer


def load_tokenizer(config: XinheConfig):
    """加载 MiniMind 的 tokenizer"""
    minimind_path = Path(config.backbone_model_path).resolve()
    tokenizer_path = minimind_path / "model" / "tokenizer.json"

    # MiniMind 使用自定义 tokenizer，尝试加载
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(minimind_path / "model"))
        return tokenizer
    except Exception:
        pass

    # 备选: 用 MiniMind 自带的 tokenizer
    try:
        sys.path.insert(0, str(minimind_path))
        from model.tokenizer import Tokenizer
        tokenizer = Tokenizer(str(tokenizer_path))
        return tokenizer
    except Exception:
        pass

    raise RuntimeError(f"无法加载 tokenizer，请检查路径: {minimind_path}")


def main():
    parser = argparse.ArgumentParser(description="心核 (Xinhe) 训练")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--synthetic", action="store_true", help="使用合成记忆数据集 (用于初始验证)")
    args = parser.parse_args()

    # 加载配置
    config = XinheConfig.from_yaml(args.config)
    print(f"=== 心核 (Xinhe) 训练 ===")
    print(f"设备: {config.device} | 精度: {config.dtype}")
    print(f"状态 token: {config.n_state} | 维度: {config.state_dim}")
    print(f"LoRA rank: {config.lora_rank} | 目标模块: {config.lora_target_modules}")

    # 加载 tokenizer
    tokenizer = load_tokenizer(config)
    print(f"Tokenizer 已加载, vocab_size={getattr(tokenizer, 'vocab_size', '?')}")

    # 创建数据集
    if args.synthetic:
        print("使用合成记忆数据集")
        train_dataset = SyntheticMemoryDataset(
            tokenizer=tokenizer,
            num_episodes=2000,
            segment_length=config.segment_length,
            episode_length=config.episode_length,
            seed=42,
        )
        val_dataset = SyntheticMemoryDataset(
            tokenizer=tokenizer,
            num_episodes=200,
            segment_length=config.segment_length,
            episode_length=config.episode_length,
            seed=123,
        )
    else:
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
