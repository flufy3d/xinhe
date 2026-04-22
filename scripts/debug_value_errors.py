"""
Dump stage 5 val 里 VALUE token 预测错误的具体 case，用于诊断 96% 瓶颈根因。

用法:
    python scripts/debug_value_errors.py \
        --config configs/curriculum_qwen3.5-0.8b.yaml \
        --checkpoint checkpoints/curriculum/5_3entity_hard.pt \
        --val-data data/curriculum/5_3entity_hard/val.jsonl \
        --max-cases 20

输出按错误类型分类:
  CROSSED:  预测的 token 来自本 episode 其他 entity 的 value  → key 碰撞 / 路由错
  OFFCAT:   预测的 token 不在本 episode 任何 value 里         → 没记住 / 容量不足
  PARTIAL:  预测介于两者之间                                   → 局部正确
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xinhe.model.config import XinheConfig
from scripts.evaluate import load_model_and_tokenizer
from scripts.eval_value_breakdown import (
    tokenize_turn_with_class, CLS_VALUE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--max-episodes", type=int, default=100)
    args = ap.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)
    model.eval()

    episodes = []
    with open(args.val_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes = episodes[: args.max_episodes]
    print(f"[debug] 载入 {len(episodes)} 条 val episode")

    crossed_cases = []      # 预测来自其他 value
    offcat_cases = []       # 预测不在任何 value
    withincurr_cases = []   # 预测在当前 value 的 token 集内，但位置错了
    correct_count = 0
    total_value_tokens = 0

    for ep_idx, ep in enumerate(episodes):
        conversations = ep.get("conversations", [])
        if not conversations:
            continue

        # 本 episode 所有 value 的 token 集
        all_values_text = []
        all_value_tokens = set()
        for msg in conversations:
            v = msg.get("value")
            if v:
                all_values_text.append(v)
                all_value_tokens.update(tokenizer.encode(v, add_special_tokens=False))

        state = model.init_state(1).to(device)
        for i in range(0, len(conversations) - 1, 2):
            user_msg = conversations[i].get("content", "")
            asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
            assistant_msg = asst_entry.get("content", "")
            compute_loss = asst_entry.get("train_loss", True)
            value_str = asst_entry.get("value")

            ids, labels, token_class = tokenize_turn_with_class(
                tokenizer, user_msg, assistant_msg, config.segment_length,
                compute_loss=compute_loss, value_str=value_str,
            )
            ids = ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            with torch.no_grad():
                result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
            state = result["state_next"]

            if not compute_loss or value_str is None:
                continue

            logits = result["logits"][0, :-1]
            shift_labels = labels[0, 1:]
            shift_class = token_class[1:].to(device)
            preds = logits.argmax(dim=-1)

            value_mask = (shift_class == CLS_VALUE) & (shift_labels != -100)
            if not value_mask.any():
                continue
            total_value_tokens += value_mask.sum().item()
            correct_count += ((preds == shift_labels) & value_mask).sum().item()

            curr_value_tokens = set(tokenizer.encode(value_str, add_special_tokens=False))
            other_value_tokens = all_value_tokens - curr_value_tokens

            wrong_mask = value_mask & (preds != shift_labels)
            wrong_positions = wrong_mask.nonzero(as_tuple=True)[0].tolist()

            for pos in wrong_positions:
                gold_tok = shift_labels[pos].item()
                pred_tok = preds[pos].item()
                gold_str = tokenizer.decode([gold_tok])
                pred_str = tokenizer.decode([pred_tok])

                # 上下文：本 turn 的 user/assistant + 本 episode 所有 values
                case = {
                    "ep_idx": ep_idx,
                    "turn_idx": i // 2,
                    "user_msg": user_msg[:200],
                    "assistant_msg": assistant_msg[:200],
                    "all_values_this_episode": all_values_text,
                    "curr_value": value_str,
                    "gold_token": f"{gold_str!r} (id={gold_tok})",
                    "pred_token": f"{pred_str!r} (id={pred_tok})",
                }

                # 打印时还包含 gold token 左右几个 context token 以看清位置
                left_ctx = shift_labels[max(0, pos - 3):pos].tolist()
                right_ctx = shift_labels[pos + 1:pos + 4].tolist()
                case["context_decode"] = (
                    tokenizer.decode([t for t in left_ctx if t != -100])
                    + f" 【{gold_str} → {pred_str}】 "
                    + tokenizer.decode([t for t in right_ctx if t != -100])
                )

                if pred_tok in other_value_tokens:
                    case["type"] = "CROSSED"
                    crossed_cases.append(case)
                elif pred_tok not in all_value_tokens:
                    case["type"] = "OFFCAT"
                    offcat_cases.append(case)
                else:
                    case["type"] = "WITHINCURR"
                    withincurr_cases.append(case)

    acc = correct_count / max(total_value_tokens, 1)
    print(f"\n[总览] VALUE 准确率: {acc:.2%} ({correct_count}/{total_value_tokens})")
    print(f"[错误分布]")
    print(f"  CROSSED    (跨 entity 污染):         {len(crossed_cases)}")
    print(f"  OFFCAT     (完全不相关):              {len(offcat_cases)}")
    print(f"  WITHINCURR (当前 value 内 token 错位): {len(withincurr_cases)}")

    for label, bucket in [("CROSSED", crossed_cases),
                          ("OFFCAT", offcat_cases),
                          ("WITHINCURR", withincurr_cases)]:
        print(f"\n=== {label} 错误样例 (前 {args.max_cases}) ===")
        for case in bucket[: args.max_cases]:
            print_case(case)


def print_case(case):
    print(f"\n  [ep{case['ep_idx']} turn{case['turn_idx']} {case['type']}]")
    print(f"  user:      {case['user_msg']}")
    print(f"  assistant: {case['assistant_msg']}")
    print(f"  正确 value: {case['curr_value']!r}")
    print(f"  本 ep 所有 value: {case['all_values_this_episode']}")
    print(f"  gold token: {case['gold_token']}")
    print(f"  pred token: {case['pred_token']}")
    print(f"  context:   {case['context_decode']}")


if __name__ == "__main__":
    main()
