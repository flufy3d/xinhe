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
from xinhe.evaluation.metrics import retention_test, wipe_degradation, sleep_effect


def load_model_and_tokenizer(config, checkpoint_path, device):
    """加载模型和 tokenizer"""
    model = XinheModel(config)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.plugin.load_state_dict(checkpoint["plugin_state"])
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
    minimind_path = Path(config.backbone_model_path).resolve()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(minimind_path / "model"))
    except Exception:
        sys.path.insert(0, str(minimind_path))
        from model.tokenizer import Tokenizer
        tokenizer = Tokenizer(str(minimind_path / "model" / "tokenizer.json"))

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="心核 自动化评估")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "retention", "wipe", "sleep"])
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    config = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核 评估 ===")
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    results = {}

    # 记忆保留测试
    if args.test in ("all", "retention"):
        print("\n[1/3] 记忆保留测试...")
        ret = retention_test(model, tokenizer, distances=[1, 2, 4, 8], device=device)
        print("  距离 → 准确率:")
        for d, acc in ret.items():
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            print(f"    d={d:2d}: {bar} {acc:.1%}")
        results["retention"] = ret

    # 状态清除对比
    if args.test in ("all", "wipe"):
        print("\n[2/3] 状态清除对比...")
        wipe = wipe_degradation(model, tokenizer, device=device)
        print(f"  有状态准确率:   {wipe['with_state']:.1%}")
        print(f"  无状态准确率:   {wipe['without_state']:.1%}")
        print(f"  性能下降:       {wipe['degradation']:.1%}")
        results["wipe"] = wipe

    # Sleep 效果
    if args.test in ("all", "sleep"):
        print("\n[3/3] Sleep 效果...")
        # 创建一个有内容的状态
        state = model.init_state(1).to(device)
        text = "<s>用户：我叫张三，我住在北京。\n助手：好的。</s>"
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model(input_tensor, state)
            state = result["state_next"]

        sl = sleep_effect(model, state)
        print(f"  有效秩: {sl['rank_before']:.1f} → {sl['rank_after']:.1f} ({sl['rank_change']:+.1f})")
        print(f"  状态范数: {sl['norm_before']:.1f} → {sl['norm_after']:.1f} ({sl['norm_change']:+.1f})")
        results["sleep"] = sl

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
