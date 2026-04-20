"""
VALUE / FRAME / TELL 分类准确率评估 + Blank-state 消融

按 token 作用分三类:
- VALUE: recall 轮 assistant 里 `value` 字段对齐的 token (真正的 state 考试)
- FRAME: recall 轮里 value 以外的 assistant token (标点等, 从模板可预测)
- TELL:  tell 轮的所有 assistant token (重复用户输入即可)

Blank-state 消融: 每轮开始前重置 state → backbone + 模板的裸能力基线
With-state: 正常演化 → (With - Blank) 就是 state 的真实贡献

用法:
    python scripts/eval_value_breakdown.py --checkpoint checkpoints/xxx.pt
    python scripts/eval_value_breakdown.py --checkpoint xxx.pt --config configs/qwen3.5-4b.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from scripts.evaluate import load_model_and_tokenizer


# token class 编码
CLS_IGNORE = 0
CLS_VALUE = 1
CLS_FRAME = 2
CLS_TELL = 3
CLASS_NAMES = {CLS_VALUE: "VALUE", CLS_FRAME: "FRAME", CLS_TELL: "TELL"}


def tokenize_turn_with_class(tokenizer, user_content, assistant_content, segment_length,
                             compute_loss=True, value_str=None):
    """tokenize 一轮并返回 (input_ids, labels, token_class)。

    token_class:
        CLS_IGNORE (0): user / template / padding (label=-100)
        CLS_VALUE  (1): recall 轮 value 对齐 token
        CLS_FRAME  (2): recall 轮 value 以外 assistant token
        CLS_TELL   (3): tell 轮 assistant token
    """
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(prefix_ids)

    if compute_loss:
        labels = [-100] * prefix_len + full_ids[prefix_len:]
    else:
        labels = [-100] * len(full_ids)

    # 初始化 class: 所有 assistant token 默认为 TELL (非 recall 轮) 或 FRAME (recall 轮)
    is_recall = (value_str is not None)
    default_asst_class = CLS_FRAME if is_recall else CLS_TELL
    token_class = [CLS_IGNORE if lab == -100 else default_asst_class for lab in labels]

    # 如果是 recall 轮, 用 offset_mapping 把 value 区间内的 token 标为 VALUE
    if is_recall:
        v_start = full_text.find(value_str, len(prefix_text))
        if v_start < 0:
            v_start = full_text.find(value_str)
        if v_start >= 0:
            v_end = v_start + len(value_str)
            encoded = tokenizer(full_text, add_special_tokens=False,
                                return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            for i, (cs, ce) in enumerate(offsets):
                if i >= len(token_class):
                    break
                if token_class[i] != CLS_IGNORE and cs < v_end and ce > v_start:
                    token_class[i] = CLS_VALUE

    # 截断 / padding
    if len(full_ids) > segment_length:
        full_ids = full_ids[:segment_length]
        labels = labels[:segment_length]
        token_class = token_class[:segment_length]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = segment_length - len(full_ids)
    if pad_len > 0:
        full_ids += [pad_id] * pad_len
        labels += [-100] * pad_len
        token_class += [CLS_IGNORE] * pad_len

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(token_class, dtype=torch.long),
    )


@torch.no_grad()
def eval_episode(model, tokenizer, episode, device, segment_length=256, blank_state=False):
    """评估一个 episode, 返回每类的 (correct, total) 字典"""
    conversations = episode.get("conversations", [])
    if not conversations:
        return {c: (0, 0) for c in CLASS_NAMES}

    # 初始化 state
    state = model.init_state(1).to(device)

    counts = {c: [0, 0] for c in CLASS_NAMES}

    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i].get("content", "")
        asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
        assistant_msg = asst_entry.get("content", "")
        compute_loss = asst_entry.get("train_loss", True)
        value_str = asst_entry.get("value")

        ids, labels, token_class = tokenize_turn_with_class(
            tokenizer, user_msg, assistant_msg, segment_length,
            compute_loss=compute_loss, value_str=value_str,
        )

        if not compute_loss:
            # non-loss 轮仍要过 forward 更新 state, 但不统计精度
            ids = ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            if blank_state:
                state = model.init_state(1).to(device)
            result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
            state = result["state_next"]
            continue

        ids = ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        token_class = token_class.to(device)

        # Blank-state 消融: 每轮开始前重置
        if blank_state:
            state = model.init_state(1).to(device)

        result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
        state = result["state_next"]
        logits = result["logits"]  # (1, T, V)

        # Shift: logits[:, :-1] 预测 labels[:, 1:]
        shift_logits = logits[0, :-1]  # (T-1, V)
        shift_labels = labels[0, 1:]   # (T-1,)
        # token_class 也要 shift (对齐 label)
        shift_class = token_class[1:]  # (T-1,)

        preds = shift_logits.argmax(dim=-1)
        correct_mask = (preds == shift_labels)

        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
            cls_mask = (shift_class == cls) & (shift_labels != -100)
            counts[cls][1] += cls_mask.sum().item()
            counts[cls][0] += (correct_mask & cls_mask).sum().item()

    return {c: tuple(v) for c, v in counts.items()}


def aggregate(all_counts):
    """合并多个 episode 的 (correct, total) 字典"""
    agg = {c: [0, 0] for c in CLASS_NAMES}
    for ep in all_counts:
        for c, (correct, total) in ep.items():
            agg[c][0] += correct
            agg[c][1] += total
    return {c: tuple(v) for c, v in agg.items()}


def format_table(title, with_state, blank_state):
    print(f"\n=== {title} ===")
    print(f"{'类型':<8} {'With-state':<20} {'Blank-state':<20} {'State 贡献':<12}")
    print("-" * 60)
    total_correct_ws = total_count_ws = 0
    for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
        name = CLASS_NAMES[cls]
        ws_c, ws_t = with_state[cls]
        bs_c, bs_t = blank_state[cls]
        ws_acc = ws_c / max(ws_t, 1)
        bs_acc = bs_c / max(bs_t, 1)
        contrib = ws_acc - bs_acc
        print(f"{name:<8} {ws_acc:>6.2%} ({ws_c}/{ws_t})     "
              f"{bs_acc:>6.2%} ({bs_c}/{bs_t})     {contrib:+.2%}")
        total_correct_ws += ws_c
        total_count_ws += ws_t
    print("-" * 60)
    if total_count_ws > 0:
        print(f"总计    {total_correct_ws / total_count_ws:>6.2%} "
              f"({total_correct_ws}/{total_count_ws})")


def main():
    parser = argparse.ArgumentParser(description="VALUE/FRAME/TELL 分类准确率评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, default=None,
                        help="val jsonl 路径, 默认用 config 里的 val_path")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=None,
                        help="不填则用 config 的 segment_length")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # 加载 config, 允许 checkpoint 内置 config 覆盖
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")
    del checkpoint  # 释放内存

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seg_len = args.segment_length or config.segment_length
    data_path = args.data or config.val_path

    print(f"=== VALUE/FRAME/TELL 分类评估 ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  data: {data_path}")
    print(f"  max_episodes: {args.max_episodes}")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    # 加载 val 数据
    episodes = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
            if len(episodes) >= args.max_episodes:
                break
    print(f"  loaded {len(episodes)} episodes")

    # With-state
    print("\n[1/2] With-state 评估...")
    ws_results = []
    for ep in episodes:
        ws_results.append(eval_episode(model, tokenizer, ep, device, seg_len, blank_state=False))
    ws_agg = aggregate(ws_results)

    # Blank-state 消融
    print("[2/2] Blank-state 消融评估...")
    bs_results = []
    for ep in episodes:
        bs_results.append(eval_episode(model, tokenizer, ep, device, seg_len, blank_state=True))
    bs_agg = aggregate(bs_results)

    format_table(Path(args.checkpoint).name, ws_agg, bs_agg)

    if args.output:
        out = {
            "checkpoint": args.checkpoint,
            "with_state": {CLASS_NAMES[c]: list(v) for c, v in ws_agg.items()},
            "blank_state": {CLASS_NAMES[c]: list(v) for c, v in bs_agg.items()},
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
