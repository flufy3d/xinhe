"""
VALUE / FRAME / TELL 分类准确率评估 + Blank-state 消融

按 token 作用分三类:
- VALUE: recall 轮 assistant 里 `value` 字段对齐的 token (真正的 state 考试)
- FRAME: recall 轮里 value 以外的 assistant token (标点等, 从模板可预测)
- TELL:  tell 轮的所有 assistant token (重复用户输入即可)

⚠️ v5c TODO: 本脚本里 Per-slot 利用率 / 同类混淆 诊断还是 v5b slot 口径
    （state shape=(1,32,1024) 的 per-slot L2）。v5c 的 state 是 Delta Rule
    W: (1,H,d_v,d_k)，应改为 per-head W_norm 和 W_effective_rank。
    slot_norms 相关逻辑遇到新 shape 时会自动 early-skip（见 track_slot_norms）。
    VALUE/FRAME/TELL 主准确率部分与 state shape 无关，仍可用。

用法:
    # 单点评估
    python scripts/eval_value_breakdown.py --checkpoint xxx.pt

    # Sweep 模式 (用不同 same_category 生成 val 集)
    python scripts/eval_value_breakdown.py --checkpoint xxx.pt \
        --sweep-same-cat 0,0.3,0.5,0.7,1.0 --max-episodes 200
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

# 活跃 slot 阈值: L2 norm > 这个值才算活跃 (相对于空白 state ~sqrt(state_dim)*0.01 ≈ 0.32)
SLOT_ACTIVE_THRESHOLD = 0.5


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

    # 统一 value_str 为 list[str]（向后兼容 str / None / 空列表）
    if value_str is None:
        values = []
    elif isinstance(value_str, str):
        values = [value_str]
    else:
        values = [v for v in value_str if v]

    is_recall = bool(values)
    default_asst_class = CLS_FRAME if is_recall else CLS_TELL
    token_class = [CLS_IGNORE if lab == -100 else default_asst_class for lab in labels]

    # recall 轮: 用 offset_mapping 把每个 value 子串区间内的 token 标为 VALUE
    if is_recall:
        encoded = tokenizer(full_text, add_special_tokens=False,
                            return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for v in values:
            v_start = full_text.find(v, len(prefix_text))
            if v_start < 0:
                v_start = full_text.find(v)
            if v_start < 0:
                continue
            v_end = v_start + len(v)
            for i, (cs, ce) in enumerate(offsets):
                if i >= len(token_class):
                    break
                if token_class[i] != CLS_IGNORE and cs < v_end and ce > v_start:
                    token_class[i] = CLS_VALUE

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


def _collect_episode_value_tokens(tokenizer, episode) -> set:
    """采集 episode 里所有 recall 轮的 value token ids (去重, 用于同类混淆检测)。
    兼容 value 为 str / list[str]。"""
    all_value_tokens = set()
    for msg in episode.get("conversations", []):
        v = msg.get("value")
        if not v:
            continue
        items = [v] if isinstance(v, str) else list(v)
        for it in items:
            if it:
                toks = tokenizer.encode(it, add_special_tokens=False)
                all_value_tokens.update(toks)
    return all_value_tokens


@torch.no_grad()
def eval_episode(model, tokenizer, episode, device, segment_length=256,
                 blank_state=False, track_slot_norms=False, track_confusion=False):
    """评估一个 episode。

    返回:
        counts: {cls: (correct, total)}
        extras: {
            "slot_norms": list[float],              # 所有 turn 的 slot L2 norms (concat)
            "value_err_crossed": int,               # VALUE 错误且预测 token 来自本 episode 其他 value
            "value_err_offcat": int,                # VALUE 错误且预测 token 不在任何 value 内
            "value_err_total": int,                 # VALUE 总错误数
        }
    """
    conversations = episode.get("conversations", [])
    counts = {c: [0, 0] for c in CLASS_NAMES}
    extras = {
        "slot_norms": [],
        "value_err_crossed": 0,
        "value_err_offcat": 0,
        "value_err_total": 0,
    }
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
        compute_loss = asst_entry.get("train_loss", True)
        value_str = asst_entry.get("value")

        ids, labels, token_class = tokenize_turn_with_class(
            tokenizer, user_msg, assistant_msg, segment_length,
            compute_loss=compute_loss, value_str=value_str,
        )

        if not compute_loss:
            ids = ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            if blank_state:
                state = model.init_state(1).to(device)
            result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
            state = result["state_next"]
            if track_slot_norms and state.dim() == 3:
                # v5b slot 口径：(1, n_state, D) → per-slot L2
                # v5c state 是 4D (1,H,d_v,d_k)，跳过避免产生垃圾数据（TODO v5c 诊断）
                extras["slot_norms"].extend(state[0].norm(dim=-1).tolist())
            continue

        ids = ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        token_class = token_class.to(device)

        if blank_state:
            state = model.init_state(1).to(device)

        result = model(ids, state, labels=labels, pad_token_id=tokenizer.pad_token_id)
        state = result["state_next"]
        logits = result["logits"]  # (1, T, V)

        if track_slot_norms:
            extras["slot_norms"].extend(state[0].norm(dim=-1).tolist())

        shift_logits = logits[0, :-1]  # (T-1, V)
        shift_labels = labels[0, 1:]   # (T-1,)
        shift_class = token_class[1:]

        preds = shift_logits.argmax(dim=-1)
        correct_mask = (preds == shift_labels)

        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL):
            cls_mask = (shift_class == cls) & (shift_labels != -100)
            counts[cls][1] += cls_mask.sum().item()
            counts[cls][0] += (correct_mask & cls_mask).sum().item()

        # 同类混淆: 对 VALUE 错误, 判断预测 token 是否来自本 episode 其他 value
        if track_confusion and value_str is not None:
            # 兼容 value_str 为 str / list[str]
            _items = [value_str] if isinstance(value_str, str) else [v for v in value_str if v]
            curr_value_tokens = set()
            for _it in _items:
                curr_value_tokens.update(tokenizer.encode(_it, add_special_tokens=False))
            other_value_tokens = episode_value_tokens - curr_value_tokens
            value_mask = (shift_class == CLS_VALUE) & (shift_labels != -100)
            wrong_mask = value_mask & ~correct_mask
            wrong_preds = preds[wrong_mask].tolist()
            for tok in wrong_preds:
                extras["value_err_total"] += 1
                if tok in other_value_tokens:
                    extras["value_err_crossed"] += 1
                elif tok not in episode_value_tokens:
                    extras["value_err_offcat"] += 1
                # 既不在 other 也不在 curr (不可能出现), 忽略

    return {c: tuple(v) for c, v in counts.items()}, extras


def aggregate(all_results):
    """合并多个 (counts, extras) tuple。"""
    agg_counts = {c: [0, 0] for c in CLASS_NAMES}
    agg_extras = {
        "slot_norms": [],
        "value_err_crossed": 0,
        "value_err_offcat": 0,
        "value_err_total": 0,
    }
    for counts, extras in all_results:
        for c, (correct, total) in counts.items():
            agg_counts[c][0] += correct
            agg_counts[c][1] += total
        agg_extras["slot_norms"].extend(extras["slot_norms"])
        agg_extras["value_err_crossed"] += extras["value_err_crossed"]
        agg_extras["value_err_offcat"] += extras["value_err_offcat"]
        agg_extras["value_err_total"] += extras["value_err_total"]
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

    # Per-slot 利用率
    slot_norms = ws_extras["slot_norms"]
    if slot_norms:
        import statistics
        mean_norm = statistics.mean(slot_norms)
        active_ratio = sum(1 for n in slot_norms if n > SLOT_ACTIVE_THRESHOLD) / len(slot_norms)
        print(f"Slot 利用率: mean_norm={mean_norm:.3f} 活跃占比={active_ratio:.2%} "
              f"(阈值 {SLOT_ACTIVE_THRESHOLD})")

    # 同类混淆
    if ws_extras["value_err_total"] > 0:
        crossed = ws_extras["value_err_crossed"]
        offcat = ws_extras["value_err_offcat"]
        tot = ws_extras["value_err_total"]
        other = tot - crossed - offcat
        print(f"VALUE 错误分解: routing(同类预测另一 value)={crossed}/{tot} ({crossed/tot:.1%}) | "
              f"off-cat={offcat}/{tot} ({offcat/tot:.1%}) | 其他={other}/{tot} ({other/tot:.1%})")


def run_eval_on_path(model, tokenizer, data_path, device, seg_len, max_episodes,
                     also_blank=False, track_confusion=True, track_slot_norms=True):
    """在指定 val jsonl 上跑完整评估, 返回 (with_state_agg, blank_state_agg_or_None)"""
    episodes = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
            if len(episodes) >= max_episodes:
                break

    ws_results = []
    for ep in episodes:
        ws_results.append(eval_episode(
            model, tokenizer, ep, device, seg_len,
            blank_state=False,
            track_slot_norms=track_slot_norms,
            track_confusion=track_confusion,
        ))
    ws_agg = aggregate(ws_results)

    bs_agg = None
    if also_blank:
        bs_results = []
        for ep in episodes:
            bs_results.append(eval_episode(
                model, tokenizer, ep, device, seg_len,
                blank_state=True, track_slot_norms=False, track_confusion=False,
            ))
        bs_agg = aggregate(bs_results)

    return ws_agg, bs_agg, len(episodes)


def eval_value_breakdown_fast(model, tokenizer, data_path, device, seg_len=256, max_episodes=50):
    """训练期轻量评估: 返回 {"VALUE": acc, "FRAME": acc, "TELL": acc}"""
    ws_agg, _, _ = run_eval_on_path(
        model, tokenizer, data_path, device, seg_len, max_episodes,
        also_blank=False, track_confusion=False, track_slot_norms=False,
    )
    counts, _ = ws_agg
    return {
        CLASS_NAMES[cls]: counts[cls][0] / max(counts[cls][1], 1)
        for cls in (CLS_VALUE, CLS_FRAME, CLS_TELL)
    }


def sweep_same_cat(model, tokenizer, device, seg_len, values, max_episodes, out_base="data/_sweep"):
    """对每个 same_category 值生成一组 val 数据, 跑 eval, 汇总 VALUE 曲线。"""
    from xinhe.data.generate_memory_data import generate_data

    print(f"\n=== Sweep same_category ∈ {values} ===")
    print(f"每点生成 {max_episodes} eval episodes (num_facts=2, entity_ratio=1.0, max_turns=7)")
    results = []
    for v in values:
        out_dir = Path(out_base) / f"same_cat_{v}"
        print(f"\n[sweep {v}] 生成数据 → {out_dir}")
        _, val_path = generate_data(
            out_dir=str(out_dir),
            num_train=10,
            num_val=max_episodes,
            min_distance=1,
            max_distance=3,
            max_turns=7,
            num_facts=2,
            no_pre_filler=False,
            max_pre_filler=1,
            no_overwrite=True,
            entity_ratio=1.0,
            same_category=v,
            seed=42,
        )
        ws_agg, _, n = run_eval_on_path(
            model, tokenizer, val_path, device, seg_len, max_episodes,
            also_blank=False, track_confusion=True, track_slot_norms=True,
        )
        format_table(f"same_category={v} (n={n})", ws_agg, None)
        counts, extras = ws_agg
        results.append({
            "same_category": v,
            "VALUE": counts[CLS_VALUE][0] / max(counts[CLS_VALUE][1], 1),
            "FRAME": counts[CLS_FRAME][0] / max(counts[CLS_FRAME][1], 1),
            "TELL": counts[CLS_TELL][0] / max(counts[CLS_TELL][1], 1),
            "routing_err_ratio": (
                extras["value_err_crossed"] / max(extras["value_err_total"], 1)
            ),
        })

    # 汇总曲线
    print(f"\n=== VALUE sweep summary ===")
    print(f"{'same_cat':<10} {'VALUE':<10} {'routing_err%':<14}")
    print("-" * 36)
    for r in results:
        print(f"{r['same_category']:<10} {r['VALUE']:>7.2%}   {r['routing_err_ratio']:>7.1%}")
    return results


def main():
    parser = argparse.ArgumentParser(description="VALUE/FRAME/TELL 分类准确率评估 + v5a 诊断")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, default=None,
                        help="val jsonl 路径, 默认用 config 里的 val_path (sweep 模式下忽略)")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sweep-same-cat", type=str, default=None,
                        help="逗号分隔同类值, 对每个值生成 val 跑 eval, 画 VALUE 曲线。例: 0,0.3,0.5,0.7,1.0")
    parser.add_argument("--no-blank", action="store_true",
                        help="跳过 blank-state 消融 (sweep 模式默认跳过)")
    args = parser.parse_args()

    # 加载 config, 允许 checkpoint 内置 config 覆盖
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, _ = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")
    del checkpoint

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seg_len = args.segment_length or config.segment_length

    print(f"=== VALUE/FRAME/TELL 分类评估 ===")
    print(f"  checkpoint: {args.checkpoint}")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    # Sweep 模式
    if args.sweep_same_cat:
        values = [float(x) for x in args.sweep_same_cat.split(",")]
        results = sweep_same_cat(
            model, tokenizer, device, seg_len, values, args.max_episodes,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump({"checkpoint": args.checkpoint, "sweep": results}, f, indent=2)
            print(f"\n结果已保存到 {args.output}")
        return

    # 单点模式
    data_path = args.data or config.val_path
    print(f"  data: {data_path}")
    print(f"  max_episodes: {args.max_episodes}")

    ws_agg, bs_agg, n = run_eval_on_path(
        model, tokenizer, data_path, device, seg_len, args.max_episodes,
        also_blank=not args.no_blank, track_confusion=True, track_slot_norms=True,
    )
    format_table(f"{Path(args.checkpoint).name} (n={n})", ws_agg, bs_agg)

    if args.output:
        ws_counts, ws_extras = ws_agg
        out = {
            "checkpoint": args.checkpoint,
            "with_state": {CLASS_NAMES[c]: list(v) for c, v in ws_counts.items()},
            "slot_norms_mean": (
                sum(ws_extras["slot_norms"]) / len(ws_extras["slot_norms"])
                if ws_extras["slot_norms"] else 0.0
            ),
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
