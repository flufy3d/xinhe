"""
v8 评估入口（替换 v7 retention/wipe）。

用法:
    python scripts/evaluate.py --checkpoint checkpoints/curriculum/0_atomic_skeletons.pt
    python scripts/evaluate.py --checkpoint ... --val data/val/stage0/seen_entity.jsonl
    python scripts/evaluate.py --checkpoint ... --val-dir data/val
"""
import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.evaluation.event_eval import eval_event_jsonl
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


def load_model_and_tokenizer(config, checkpoint_path, device):
    model = XinheModel(config)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "hippocampus_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'hippocampus_state' 键。v8 仅兼容 v7+ 格式。"
            )
        model.hippocampus.load_state_dict(checkpoint["hippocampus_state"], strict=True)
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


def _discover_val_jsonl(root: Path) -> list[Path]:
    """递归找 jsonl 文件。"""
    if root.is_file():
        return [root] if root.suffix == ".jsonl" else []
    return sorted(root.rglob("*.jsonl"))


def main():
    parser = argparse.ArgumentParser(description="心核 v8 事件级评估")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val", type=str, default=None,
                        help="单个 jsonl 评估文件")
    parser.add_argument("--val-dir", type=str, default="data/val",
                        help="批量评估目录（递归找 jsonl）")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--seg-len", type=int, default=256)
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 路径")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核 v8 评估 ===")
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    # 收集要评估的 val 文件
    val_files: list[Path] = []
    if args.val:
        val_files.append(Path(args.val))
    if args.val_dir:
        root = Path(args.val_dir)
        if root.exists():
            val_files.extend(_discover_val_jsonl(root))
    val_files = sorted(set(val_files))
    if not val_files:
        print("没有找到 val 文件。请通过 --val 或 --val-dir 指定。")
        return

    all_results = {}
    for vf in val_files:
        name = vf.relative_to(project_root) if vf.is_absolute() else vf
        print(f"\n[{name}]")
        res = eval_event_jsonl(
            model, tokenizer, vf,
            device=device,
            seg_len=args.seg_len,
            max_episodes=args.max_episodes,
        )
        print(f"  episodes={res['n_episodes']}, value_turns={res['n_value_turns']}, "
              f"correct={res['n_correct']}, overall_acc={res['overall_acc']:.2%}")
        for label, dim in (("skeleton", res.get("by_skeleton")),
                            ("distance", res.get("by_distance")),
                            ("tier", res.get("by_tier")),
                            ("substream", res.get("by_substream"))):
            if dim:
                line = " ".join(f"{k}={v['acc']:.2%}({v['n']})" for k, v in dim.items())
                print(f"    {label}: {line}")
        all_results[str(name)] = res

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {args.output}")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
