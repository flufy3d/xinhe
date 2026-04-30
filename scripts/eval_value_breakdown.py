"""
VALUE / FRAME / TELL 分类准确率评估 + Blank-state 消融

按 token 作用分三类:
- VALUE: recall 轮 assistant 里 `value` 字段对齐的 token (真正的 state 考试)
- FRAME: recall 轮里 value 以外的 assistant token (标点等, 从模板可预测)
- TELL:  tell 轮的所有 assistant token (重复用户输入即可)

trainer 在每个 eval_every 通过 eval_value_breakdown_fast 调用,产 [val breakdown] 行;
独立 CLI 还能跑 Blank-state 消融(每轮重置 state) → 算 "state 贡献" 差值,
以及 VALUE 错误的同类混淆诊断 (routing-错-到-本-episode-另一-value vs off-cat)。

用法:
    python scripts/eval_value_breakdown.py --checkpoint xxx.pt
    python scripts/eval_value_breakdown.py --checkpoint xxx.pt --no-blank
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


def tokenize_turn_with_class(tokenizer, user_content, assistant_content, turn_max_tokens,
                             *, train_loss="true", value_spans=None):
    """tokenize 一轮并返回 (input_ids, labels, token_class)。

    接口:value_spans 是 list[[start_char, end_char]](assistant_content 坐标系),
    train_loss 三态: "true" / "lm_only" / "false" (兼容 bool)。

    token_class:
        CLS_IGNORE (0): user / template / padding / lm_only / false
        CLS_VALUE  (1): recall 轮 value span 内 token
        CLS_FRAME  (2): recall 轮 value span 外 assistant token
        CLS_TELL   (3): tell 轮 assistant token(无 value,但 train_loss=true)
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

    if train_loss is True or train_loss == "true":
        compute_loss = True
        is_lm_only = False
    elif train_loss == "lm_only":
        compute_loss = True
        is_lm_only = True
    else:
        compute_loss = False
        is_lm_only = False

    if compute_loss:
        labels = [-100] * prefix_len + full_ids[prefix_len:]
    else:
        labels = [-100] * len(full_ids)

    spans = list(value_spans or [])
    is_recall = bool(spans) and not is_lm_only
    if is_lm_only:
        token_class = [CLS_IGNORE] * len(labels)
    else:
        default_asst_class = CLS_FRAME if is_recall else CLS_TELL
        token_class = [CLS_IGNORE if lab == -100 else default_asst_class for lab in labels]

    if is_recall:
        asst_offset = full_text.find(assistant_content, len(prefix_text))
        if asst_offset < 0:
            asst_offset = full_text.find(assistant_content)

        if asst_offset >= 0:
            encoded = tokenizer(full_text, add_special_tokens=False,
                                return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            for s, e in spans:
                s_full = asst_offset + int(s)
                e_full = asst_offset + int(e)
                for i, (cs, ce) in enumerate(offsets):
                    if i >= len(token_class):
                        break
                    if token_class[i] != CLS_IGNORE and cs < e_full and ce > s_full:
                        token_class[i] = CLS_VALUE

    if len(full_ids) > turn_max_tokens:
        full_ids = full_ids[:turn_max_tokens]
        labels = labels[:turn_max_tokens]
        token_class = token_class[:turn_max_tokens]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_len = turn_max_tokens - len(full_ids)
    if pad_len > 0:
        full_ids += [pad_id] * pad_len
        labels += [-100] * pad_len
        token_class += [CLS_IGNORE] * pad_len

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(token_class, dtype=torch.long),
    )


def _collect_episode_value_tokens(tokenizer, episode) -> set:
    """采集 episode 里所有 recall 轮的 value token ids(去重,用于同类混淆检测)。"""
    all_value_tokens = set()
    for msg in episode.get("conversations", []):
        v = msg.get("value")
        if not v:
            continue
        items = [v] if isinstance(v, str) else list(v)
        for it in items:
            if it:
                all_value_tokens.update(tokenizer.encode(it, add_special_tokens=False))
    return all_value_tokens


@torch.no_grad()
def eval_episode(model, tokenizer, episode, device, turn_max_tokens=256,
                 *, blank_state=False, track_confusion=False):
    """评估一个 episode。

    Returns:
        counts: {cls: (correct, total)}
        extras: {value_err_crossed, value_err_offcat, value_err_total}
                只在 track_confusion=True 时填充,否则全 0
    """
    conversations = episode.get("conversations", [])
    counts = {c: [0, 0] for c in CLASS_NAMES}
    extras = {"value_err_crossed": 0, "value_err_offcat": 0, "value_err_total": 0}
    if not conversations:
        return {c: tuple(v) for c, v in counts.items()}, extras

    state = model.init_state(1).to(device)
    episode_value_tokens = (
        _collect_episode_value_tokens(tokenizer, episode) if track_confusion else None
    )

    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i].get("content", "")
        asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
        assistant_msg = asst_entry.get("content", "")
        train_loss = asst_entry.get("train_loss", "true")
        value_str = asst_entry.get("value")
        value_spans = asst_entry.get("value_span") or []

        ids, labels, token_class = tokenize_turn_with_class(
            tokenizer, user_msg, assistant_msg, turn_max_tokens,
            train_loss=train_loss, value_spans=value_spans,
        )

        no_loss = (train_loss is False or train_loss == "false")
        no_breakdown = no_loss or train_loss == "lm_only"

        ids = ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        if blank_state:
            state = model.init_state(1).to(device)

        if no_breakdown:
            result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
            state = result["state_next"]
            continue

        token_class = token_class.to(device)
        result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
        state = result["state_next"]
        logits = result["logits"]  # (1, T, V)

        shift_logits = logits[0, :-1]
        shift_labels = labels[0, 1:]
        shift_class = token_class[1:]

        preds = shift_logits.argmax(dim=-1)
        correct_mask = (preds == shift_labels)

        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
            cls_mask = (shift_class == cls) & (shift_labels != -100)
            counts[cls][1] += cls_mask.sum().item()
            counts[cls][0] += (correct_mask & cls_mask).sum().item()

        if track_confusion and value_str is not None:
            _items = [value_str] if isinstance(value_str, str) else [v for v in value_str if v]
            curr_value_tokens = set()
            for _it in _items:
                curr_value_tokens.update(tokenizer.encode(_it, add_special_tokens=False))
            other_value_tokens = episode_value_tokens - curr_value_tokens
            value_mask = (shift_class == CLS_VALUE) & (shift_labels != -100)
            wrong_mask = value_mask & ~correct_mask
            for tok in preds[wrong_mask].tolist():
                extras["value_err_total"] += 1
                if tok in other_value_tokens:
                    extras["value_err_crossed"] += 1
                elif tok not in episode_value_tokens:
                    extras["value_err_offcat"] += 1

    return {c: tuple(v) for c, v in counts.items()}, extras


def aggregate(all_results):
    agg_counts = {c: [0, 0] for c in CLASS_NAMES}
    agg_extras = {"value_err_crossed": 0, "value_err_offcat": 0, "value_err_total": 0}
    for counts, extras in all_results:
        for c, (correct, total) in counts.items():
            agg_counts[c][0] += correct
            agg_counts[c][1] += total
        for k in agg_extras:
            agg_extras[k] += extras[k]
    return {c: tuple(v) for c, v in agg_counts.items()}, agg_extras


def format_table(title, with_state_agg, blank_state_agg=None):
    ws_counts, ws_extras = with_state_agg
    print(f"\n=== {title} ===")
    if blank_state_agg is not None:
        bs_counts, _ = blank_state_agg
        print(f"{'类型':<8} {'With-state':<22} {'Blank-state':<22} {'State 贡献':<12}")
        print("-" * 66)
        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
            name = CLASS_NAMES[cls]
            ws_c, ws_t = ws_counts[cls]
            bs_c, bs_t = bs_counts[cls]
            ws_acc = ws_c / max(ws_t, 1)
            bs_acc = bs_c / max(bs_t, 1)
            print(f"{name:<8} {ws_acc:>7.2%} ({ws_c}/{ws_t})     "
                  f"{bs_acc:>7.2%} ({bs_c}/{bs_t})     {ws_acc - bs_acc:+.2%}")
        print("-" * 66)
    else:
        print(f"{'类型':<8} {'Accuracy':<22}")
        print("-" * 32)
        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
            name = CLASS_NAMES[cls]
            c, t = ws_counts[cls]
            print(f"{name:<8} {c / max(t, 1):>7.2%} ({c}/{t})")
        print("-" * 32)

    if ws_extras["value_err_total"] > 0:
        crossed = ws_extras["value_err_crossed"]
        offcat = ws_extras["value_err_offcat"]
        tot = ws_extras["value_err_total"]
        other = tot - crossed - offcat
        print(f"VALUE 错误分解: routing(同类预测另一 value)={crossed}/{tot} ({crossed/tot:.1%}) | "
              f"off-cat={offcat}/{tot} ({offcat/tot:.1%}) | 其他={other}/{tot} ({other/tot:.1%})")


def run_eval_on_path(model, tokenizer, data_path, device, seg_len, max_episodes,
                     *, also_blank=False, track_confusion=True):
    """在指定 val jsonl 上跑完整评估,返回 (with_state_agg, blank_state_agg_or_None, n_episodes)"""
    episodes = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
            if len(episodes) >= max_episodes:
                break

    ws_results = [
        eval_episode(model, tokenizer, ep, device, seg_len,
                     blank_state=False, track_confusion=track_confusion)
        for ep in episodes
    ]
    ws_agg = aggregate(ws_results)

    bs_agg = None
    if also_blank:
        bs_results = [
            eval_episode(model, tokenizer, ep, device, seg_len,
                         blank_state=True, track_confusion=False)
            for ep in episodes
        ]
        bs_agg = aggregate(bs_results)

    return ws_agg, bs_agg, len(episodes)


def eval_value_breakdown_fast(model, tokenizer, data_path, device, seg_len=256, max_episodes=50):
    """训练期轻量评估: 返回 {"VALUE": acc, "FRAME": acc, "TELL": acc}"""
    ws_agg, _, _ = run_eval_on_path(
        model, tokenizer, data_path, device, seg_len, max_episodes,
        also_blank=False, track_confusion=False,
    )
    counts, _ = ws_agg
    return {
        CLASS_NAMES[cls]: counts[cls][0] / max(counts[cls][1], 1)
        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL)
    }


def main():
    parser = argparse.ArgumentParser(description="VALUE/FRAME/TELL 分类准确率 + Blank-state 消融")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data", default=None,
                        help="val jsonl 路径,默认用 config.val_path")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--turn-max-tokens", type=int, default=None)
    parser.add_argument("--no-blank", action="store_true",
                        help="跳过 blank-state 消融(节省一半时间)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")
    del checkpoint

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seg_len = args.turn_max_tokens or config.turn_max_tokens
    data_path = args.data or config.val_path

    print("=== VALUE/FRAME/TELL 分类评估 ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  data: {data_path}")
    print(f"  max_episodes: {args.max_episodes} | seg_len: {seg_len}")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    ws_agg, bs_agg, n = run_eval_on_path(
        model, tokenizer, data_path, device, seg_len, args.max_episodes,
        also_blank=not args.no_blank, track_confusion=True,
    )
    format_table(f"{Path(args.checkpoint).name} (n={n})", ws_agg, bs_agg)

    if args.output:
        ws_counts, ws_extras = ws_agg
        out = {
            "checkpoint": args.checkpoint,
            "with_state": {CLASS_NAMES[c]: list(v) for c, v in ws_counts.items()},
            "value_err_breakdown": {
                "crossed": ws_extras["value_err_crossed"],
                "offcat": ws_extras["value_err_offcat"],
                "total": ws_extras["value_err_total"],
            },
        }
        if bs_agg is not None:
            bs_counts, _ = bs_agg
            out["blank_state"] = {CLASS_NAMES[c]: list(v) for c, v in bs_counts.items()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
