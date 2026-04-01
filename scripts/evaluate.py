"""
自动化评估入口

用法:
    python scripts/evaluate.py --checkpoint checkpoints/xinhe_step_5000.pt
    python scripts/evaluate.py --checkpoint checkpoints/xinhe_step_5000.pt --test retention
"""
import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.evaluation.metrics import retention_test, wipe_degradation


def load_model_and_tokenizer(config, checkpoint_path, device):
    """加载模型和 tokenizer"""
    model = XinheModel(config)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model.plugin.load_state_dict(checkpoint["plugin_state"])
        except RuntimeError as e:
            raise RuntimeError(
                "checkpoint 与 --config 不匹配。"
            ) from e
        from xinhe.model.lora import LoRALinear
        lora_state = checkpoint.get("lora_state", {})
        for name, module in model.backbone.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state:
                    module.lora_B.data = lora_state[f"{name}.lora_B"]

    model.to(device)
    model.eval()

    # 加载 tokenizer
    from transformers import AutoTokenizer
    from xinhe.data.conversation import ensure_chat_template
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(config.backbone_model_path).resolve()),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="心核 自动化评估")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "retention", "wipe"])
    parser.add_argument("--num-trials", type=int, default=10, help="每项测试的试验次数")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")
    elif not config_explicit and "config" not in checkpoint:
        print("  提示: 请使用与 checkpoint 匹配的 --config。")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核 评估 ===")
    if config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        ckpt_cfg = checkpoint["config"]
        if (ckpt_cfg.backbone_type != config.backbone_type) or (ckpt_cfg.hidden_size != config.hidden_size):
            print("警告: --config 与 checkpoint 不匹配。")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    results = {}

    # 记忆保留测试
    if args.test in ("all", "retention"):
        print("\n[1/2] 记忆保留测试...")
        ret = retention_test(model, tokenizer, distances=[1, 2, 4, 6, 8, 10], num_trials=args.num_trials, device=device)
        print("  距离 → 准确率:")
        for d, acc in ret.items():
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"    d={d:2d}: {bar} {acc:.1%}")
        results["retention"] = ret

    # 状态清除对比
    if args.test in ("all", "wipe"):
        print("\n[2/2] 状态清除对比...")
        wipe = wipe_degradation(model, tokenizer, num_trials=args.num_trials, device=device)
        print(f"  有状态准确率:   {wipe['with_state']:.1%}")
        print(f"  无状态准确率:   {wipe['without_state']:.1%}")
        print(f"  性能下降:       {wipe['degradation']:.1%}")
        results["wipe"] = wipe

    # 保存结果
    if args.output:
        # 转换为可序列化格式
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable[k] = v
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n结果已保存到 {args.output}")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
