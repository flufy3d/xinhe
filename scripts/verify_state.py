"""
State 记忆验证脚本 — 用训练方式处理 segment，检查 state 是否能记住信息

不依赖生成质量，直接检查模型对 recall segment 中名字 token 的预测概率。
"""
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import ensure_chat_template, tokenize_turn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/xinhe_step_2000.pt")
    args = parser.parse_args()

    config = XinheConfig.from_yaml("configs/base.yaml")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = XinheModel(config)
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).exists():
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
        print(f"Checkpoint 已加载: {checkpoint_path}")

    model.to(device)
    model.eval()

    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(config.backbone_model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)

    scale = torch.sigmoid(model.plugin.state_scale).item()
    print(f"Scale: {scale:.4f}")
    print()

    # 测试用例
    test_cases = [
        ("name", "陈杰", "我叫陈杰。", "好的，陈杰，很高兴认识你！", "我叫什么名字？", "你叫陈杰。"),
        ("name", "张三", "我的名字是张三。", "你好张三！我记住你的名字了。", "你还记得我的名字吗？", "当然记得，你叫张三。"),
        ("number", "8472", "我的编号是8472。", "好的，我记住了，你的编号是8472。", "我的编号是什么？", "你的编号是8472。"),
        ("city", "杭州", "我住在杭州。", "好的，杭州是个好地方！", "我住在哪里？", "你住在杭州。"),
    ]

    seg_len = config.segment_length
    total_correct = 0
    total_tests = 0

    for cat, value, tell_user, tell_asst, recall_user, recall_asst in test_cases:
        print(f"{'='*60}")
        print(f"  测试: {cat} = {value}")
        print(f"{'='*60}")

        # === 有 state 的情况 ===
        state = model.init_state(1).to(device)

        # 处理 tell segment (完整 turn，和训练一样)
        tell_ids, tell_labels = tokenize_turn(tokenizer, tell_user, tell_asst, seg_len)
        tell_ids = tell_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(tell_ids, state)
            state_after_tell = result["state_next"]

        # 处理 filler segment
        filler_ids, _ = tokenize_turn(tokenizer, "今天天气怎么样？", "今天天气不错。", seg_len)
        filler_ids = filler_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            result = model(filler_ids, state_after_tell)
            state_after_filler = result["state_next"]

        # 处理 recall segment — 只送 user prompt，检查模型预测
        recall_ids, recall_labels = tokenize_turn(tokenizer, recall_user, recall_asst, seg_len)
        recall_ids = recall_ids.unsqueeze(0).to(device)
        recall_labels = recall_labels.unsqueeze(0).to(device)

        with torch.no_grad():
            result = model(recall_ids, state_after_filler, labels=recall_labels)
            logits = result["logits"]  # (1, T, V)
            loss_with_state = result["loss"].item()

        # 找到有效 label 位置，检查 top-1 预测
        labels_flat = recall_labels[0]
        valid_mask = labels_flat != -100
        valid_positions = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_positions) > 0:
            correct = 0
            total = 0
            print(f"  [有 State] loss={loss_with_state:.6f}")
            for pos in valid_positions:
                # next-token prediction: logits[pos-1] 预测 labels[pos]
                pred_pos = pos - 1
                if pred_pos < 0:
                    continue
                pred_logits = logits[0, pred_pos]
                pred_token = pred_logits.argmax().item()
                true_token = labels_flat[pos].item()
                is_correct = pred_token == true_token
                correct += int(is_correct)
                total += 1
                pred_str = tokenizer.decode([pred_token])
                true_str = tokenizer.decode([true_token])
                marker = "✓" if is_correct else "✗"
                print(f"    {marker} 预测='{pred_str}' 真实='{true_str}'")

            acc = correct / total if total > 0 else 0
            print(f"  准确率: {correct}/{total} = {acc:.1%}")
            total_correct += correct
            total_tests += total

        # === 无 state (wipe) 的情况 ===
        blank_state = model.init_state(1).to(device)
        with torch.no_grad():
            result_blank = model(recall_ids, blank_state, labels=recall_labels)
            loss_no_state = result_blank["loss"].item()

        if len(valid_positions) > 0:
            logits_blank = result_blank["logits"]
            correct_blank = 0
            print(f"\n  [无 State (wipe)] loss={loss_no_state:.6f}")
            for pos in valid_positions:
                pred_pos = pos - 1
                if pred_pos < 0:
                    continue
                pred_logits = logits_blank[0, pred_pos]
                pred_token = pred_logits.argmax().item()
                true_token = labels_flat[pos].item()
                is_correct = pred_token == true_token
                correct_blank += int(is_correct)
                pred_str = tokenizer.decode([pred_token])
                true_str = tokenizer.decode([true_token])
                marker = "✓" if is_correct else "✗"
                print(f"    {marker} 预测='{pred_str}' 真实='{true_str}'")

            acc_blank = correct_blank / total if total > 0 else 0
            print(f"  准确率: {correct_blank}/{total} = {acc_blank:.1%}")

        print(f"\n  Loss 对比: 有state={loss_with_state:.6f} vs 无state={loss_no_state:.6f}")
        print()

    print(f"{'='*60}")
    print(f"  总计: {total_correct}/{total_tests} = {total_correct/max(total_tests,1):.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
