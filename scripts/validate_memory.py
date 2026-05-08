"""
strict 记忆能力验证脚本(独立于 evaluate.py)。

三口径(per value_span,assistant turn 有 value_span 时):
  1. first-token argmax —— value_span 第一个 token 在 user prompt + asst-prefix 之后的 P(token|context)
     条件里没有任何 value 信息(prefix 没到 value 起点),纯测 memory 召回
  2. free-gen greedy —— user prompt 后 greedy decode,跟 gold value 字符串比对(startswith)
     模拟 chat.py 真实场景
  3. NM-zero ablation —— 上面两个口径都跑两次:正常 + mem_alpha_override=0.0
     (xinhe_model.py:258 把 fresh_proj 乘 0 → MAC 注入彻底关)

每个 value_span 标 is_recall = (value 字符串不在当前 user_msg 里):
  - is_recall=False:write/echo turn(用户给值,模型 copy),不真正测记忆
  - is_recall=True:read turn,模型必须依赖 NM 召回(真实记忆测试)

输出 per-skeleton × distance × {NM-on / NM-zero} × {recall/write} 表格(stdout + json)。

CLI:
  python scripts/validate_memory.py \
      --checkpoint checkpoints/xinhe_step_2500.pt \
      --config configs/pcap.yaml \
      --val data/skeleton/val.jsonl \
      --max-episodes 200 \
      --output checkpoints/validate_memory_step_2500.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.conversation import ensure_chat_template, tokenize_turn
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from scripts.evaluate import load_model_and_tokenizer


def _locate_value_token(tokenizer, user_msg: str, asst_text: str, char_span: tuple[int, int]):
    """
    返回 (full_input_ids, target_token_id, target_token_pos)
    full_input_ids[:target_token_pos] 是模型看到的 prefix,target_token_pos 处是 value 首 token,
    forward(prefix) 取 last_logits.argmax 跟 target_token_id 比较即测 first-token。
    定位失败返回 (None, None, None)。
    """
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg},
         {"role": "assistant", "content": asst_text}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    asst_offset = full.find(asst_text, len(prefix))
    if asst_offset < 0:
        asst_offset = full.find(asst_text)
    if asst_offset < 0:
        return None, None, None

    s_full = asst_offset + char_span[0]
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    ids = encoded["input_ids"]

    for i, (cs, ce) in enumerate(offsets):
        if cs <= s_full < ce or (cs == s_full and ce == s_full):
            return ids, ids[i], i
        if cs > s_full:
            # 第一个起点 > s_full 的 token,说明 s_full 落在前一个 token 内或边界,取它前一个
            if i > 0:
                return ids, ids[i - 1], i - 1
            return None, None, None

    return None, None, None


@torch.no_grad()
def _check_first_token(model, tokenizer, state, user_msg, asst_text, char_span,
                       device, mem_alpha_override):
    """
    返回 first_token_correct (True/False) 或 None(定位失败)。
    """
    full_ids, target_tok, target_pos = _locate_value_token(
        tokenizer, user_msg, asst_text, char_span,
    )
    if full_ids is None or target_pos is None or target_pos == 0:
        return None

    prefix_ids = full_ids[:target_pos]
    input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(input_tensor, state, pad_token_id=tokenizer.pad_token_id,
                    mem_alpha_override=mem_alpha_override)
    pred = out["logits"][0, -1].argmax().item()
    return pred == target_tok


@torch.no_grad()
def _check_free_gen(model, tokenizer, state, user_msg, value_str,
                    device, mem_alpha_override, max_new_tokens=32):
    """
    user prompt 后 greedy decode,看 value_str 是否出现在生成内容里。

    口径:value_str.strip() in decoded(子串)。模型自然会有"嗯,xxx 是 公安部"这种
    preamble,所以 startswith 太严;substring 命中即算 chat.py 用户能拿到答案。
    模型说够 target_chars + 24 个字符后停(够看完答案 + buffer 退出),或 EOS 早停。
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    target = value_str.strip()
    target_chars = max(len(target), 1)
    char_budget = target_chars + 24  # 给模型 preamble + answer + tail 余量

    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(generated, state, pad_token_id=tokenizer.pad_token_id,
                    mem_alpha_override=mem_alpha_override)
    next_logits = out["logits"][:, -1, :]

    for _ in range(max_new_tokens):
        next_tok = next_logits.argmax(dim=-1, keepdim=True)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        generated = torch.cat([generated, next_tok], dim=1)
        decoded = tokenizer.decode(
            generated[0, len(input_ids):], skip_special_tokens=True,
        )
        # 已命中则提前停
        if target in decoded:
            return True
        if len(decoded) >= char_budget:
            break
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            out = model(generated, state, pad_token_id=tokenizer.pad_token_id,
                        mem_alpha_override=mem_alpha_override)
        next_logits = out["logits"][:, -1, :]

    decoded = tokenizer.decode(
        generated[0, len(input_ids):], skip_special_tokens=True,
    )
    return target in decoded


@torch.no_grad()
def evaluate_episode_strict(model, tokenizer, episode, device, seg_len, mem_alpha_override):
    """
    跑一条 episode,返回每个 value_span 命中记录:
      [{
        "skeleton_id", "distance", "value_tier", "is_recall",
        "first_token_correct" (None / True / False),
        "free_gen_correct" (None / True / False),
        "value": str,
      }, ...]

    state 通过 teacher-forced forward 跨 turn 演化(模拟训练分布)。
    每个 value_span 在 turn forward 之前测(state 是写之前的状态)。
    """
    convs = episode.get("conversations", [])
    skeleton_id = episode.get("skeleton_id")
    distance = (episode.get("meta") or {}).get("distance_bucket")
    state = model.init_state(1).to(device)
    results = []

    for i in range(0, len(convs) - 1, 2):
        user_msg = convs[i].get("content", "")
        asst_entry = convs[i + 1] if i + 1 < len(convs) else {}
        assistant_msg = asst_entry.get("content", "")
        train_loss = asst_entry.get("train_loss", "true")
        value_spans = asst_entry.get("value_span") or []
        value_tier = asst_entry.get("value_tier")

        # 在 forward 之前测每个 value_span(state 是当前进入该 turn 的状态)
        for s, e in value_spans:
            value_str = assistant_msg[s:e]
            is_recall = value_str not in user_msg

            ft = _check_first_token(
                model, tokenizer, state, user_msg, assistant_msg, (int(s), int(e)),
                device, mem_alpha_override,
            )
            # free-gen 是 N 倍 forward 的开销,只在 recall turn(真测记忆)上跑;
            # write turn first-token 已经能反映"模型在已知前文条件下能否吐 value 首 token",
            # write turn 的 free-gen 主要测"模型生成路径下能否复述 prompt",信号弱开销大
            if is_recall:
                fg = _check_free_gen(
                    model, tokenizer, state, user_msg, value_str,
                    device, mem_alpha_override,
                )
            else:
                fg = None

            results.append({
                "skeleton_id": skeleton_id,
                "distance": distance,
                "value_tier": value_tier,
                "is_recall": is_recall,
                "first_token_correct": ft,
                "free_gen_correct": fg,
                "value": value_str,
            })

        # 推进 state(teacher-forced 用 ground-truth assistant 让 state 跨 turn 演化)
        weight_per_span = float(asst_entry.get("weight_per_span", 0.0) or 0.0)
        ids, labels, _w = tokenize_turn(
            tokenizer, user_msg, assistant_msg, seg_len,
            train_loss=train_loss,
            value_spans=value_spans,
            weight_per_span=weight_per_span,
        )
        ids_dev = ids.unsqueeze(0).to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            out = model(ids_dev, state, pad_token_id=tokenizer.pad_token_id,
                        mem_alpha_override=mem_alpha_override)
        state = out["state_next"]

    return results


def _aggregate(records: list[dict]) -> dict:
    """records → per-skeleton × distance × is_recall 的 acc dict。"""
    overall = {"first_token": [], "free_gen": []}
    by_skeleton = defaultdict(lambda: {"first_token": [], "free_gen": []})
    by_distance = defaultdict(lambda: {"first_token": [], "free_gen": []})
    by_skeleton_distance = defaultdict(lambda: {"first_token": [], "free_gen": []})
    by_recall = defaultdict(lambda: {"first_token": [], "free_gen": []})  # write / read

    for r in records:
        for metric in ("first_token", "free_gen"):
            v = r[f"{metric}_correct"]
            if v is None:
                continue
            overall[metric].append(v)
            sk = r.get("skeleton_id")
            dist = r.get("distance")
            rec = r.get("is_recall")
            if sk:
                by_skeleton[sk][metric].append(v)
            if dist:
                by_distance[dist][metric].append(v)
            if sk and dist:
                by_skeleton_distance[f"{sk}/{dist}"][metric].append(v)
            by_recall["read" if rec else "write"][metric].append(v)

    def _summ(lst):
        return {
            "acc": sum(lst) / len(lst) if lst else 0.0,
            "n": len(lst),
        }

    def _flatten(d):
        return {k: {m: _summ(vs) for m, vs in metrics.items()} for k, metrics in d.items()}

    return {
        "overall": {m: _summ(vs) for m, vs in overall.items()},
        "by_skeleton": _flatten(by_skeleton),
        "by_distance": _flatten(by_distance),
        "by_skeleton_distance": _flatten(by_skeleton_distance),
        "by_recall": _flatten(by_recall),
    }


def _print_table(label: str, agg_on: dict, agg_off: dict):
    """打印 NM-on / NM-zero 对照表。agg_on/off 都是 _aggregate 输出的 by_* dict。"""
    print(f"\n[{label}]")
    print(f"  {'key':<22}  {'first-token (NM-on / zero / Δ / n)':<38}  {'free-gen (NM-on / zero / Δ / n)':<38}")
    keys = sorted(set(agg_on.keys()) | set(agg_off.keys()), key=lambda x: str(x))
    for k in keys:
        on = agg_on.get(k, {})
        off = agg_off.get(k, {})

        ft_on = on.get("first_token", {}).get("acc", 0.0)
        ft_off = off.get("first_token", {}).get("acc", 0.0)
        ft_n = on.get("first_token", {}).get("n", 0)
        fg_on = on.get("free_gen", {}).get("acc", 0.0)
        fg_off = off.get("free_gen", {}).get("acc", 0.0)
        fg_n = on.get("free_gen", {}).get("n", 0)

        ft_delta = (ft_on - ft_off) * 100
        fg_delta = (fg_on - fg_off) * 100

        print(f"  {str(k):<22}  "
              f"{ft_on*100:5.1f}% / {ft_off*100:5.1f}% / {ft_delta:+5.1f}pp / n={ft_n:<5}  "
              f"{fg_on*100:5.1f}% / {fg_off*100:5.1f}% / {fg_delta:+5.1f}pp / n={fg_n}")


def _run_pass(model, tokenizer, val_path, device, seg_len, max_episodes,
              mem_alpha_override, label):
    """Run one full pass over val.jsonl with given mem_alpha_override. Returns list of records."""
    records = []
    n_eps = 0
    print(f"  [{label}] start: max_episodes={max_episodes}", flush=True)
    with open(val_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if n_eps >= max_episodes:
                break
            ep = json.loads(line)
            try:
                ep_records = evaluate_episode_strict(
                    model, tokenizer, ep, device, seg_len, mem_alpha_override,
                )
            except Exception:
                if n_eps == 0:
                    raise
                continue
            records.extend(ep_records)
            n_eps += 1
            if n_eps % 20 == 0:
                done_recall = [r for r in records if r["is_recall"]]
                ft_acc = sum(r["first_token_correct"] for r in done_recall
                             if r["first_token_correct"] is not None) / max(
                    sum(1 for r in done_recall if r["first_token_correct"] is not None), 1)
                print(f"  [{label}] ep {n_eps}/{max_episodes} "
                      f"recall_n={len(done_recall)} recall_first_token={ft_acc:.2%}",
                      flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="心核 strict 记忆能力验证")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/pcap.yaml")
    parser.add_argument("--val", default="data/skeleton/val.jsonl")
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--seg-len", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        print(f"  使用 ckpt 内置配置: backbone={config.backbone_type}")

    # validate_memory 跑变长序列 free-gen,torch.compile 每个长度都会 recompile 触顶 cache 限制。
    # 关 compile 跟 chat.py 一致,纯 eager 跑。
    config.compile_backbone_layers = False

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seg_len = args.seg_len or getattr(config, "turn_max_tokens", 256)

    print("=== strict memory validation ===")
    print(f"  ckpt: {args.checkpoint}")
    print(f"  val:  {args.val}")
    print(f"  max_episodes: {args.max_episodes} | seg_len: {seg_len}")

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    print("\n--- pass 1: NM-on (mem_alpha_override=None) ---")
    records_on = _run_pass(
        model, tokenizer, args.val, device, seg_len, args.max_episodes,
        mem_alpha_override=None, label="NM-on",
    )

    print("\n--- pass 2: NM-zero (mem_alpha_override=0.0) ---")
    records_off = _run_pass(
        model, tokenizer, args.val, device, seg_len, args.max_episodes,
        mem_alpha_override=0.0, label="NM-zero",
    )

    agg_on = _aggregate(records_on)
    agg_off = _aggregate(records_off)

    # 总体
    print("\n=== overall ===")
    print(f"first-token  NM-on  {agg_on['overall']['first_token']['acc']*100:5.1f}%  "
          f"NM-zero {agg_off['overall']['first_token']['acc']*100:5.1f}%  "
          f"Δ {(agg_on['overall']['first_token']['acc'] - agg_off['overall']['first_token']['acc'])*100:+5.1f}pp  "
          f"(n={agg_on['overall']['first_token']['n']})")
    print(f"free-gen     NM-on  {agg_on['overall']['free_gen']['acc']*100:5.1f}%  "
          f"NM-zero {agg_off['overall']['free_gen']['acc']*100:5.1f}%  "
          f"Δ {(agg_on['overall']['free_gen']['acc'] - agg_off['overall']['free_gen']['acc'])*100:+5.1f}pp  "
          f"(n={agg_on['overall']['free_gen']['n']})")

    # 各维度对照
    _print_table("by_recall (True=read turn 真测记忆)",
                 agg_on["by_recall"], agg_off["by_recall"])
    _print_table("by_skeleton", agg_on["by_skeleton"], agg_off["by_skeleton"])
    _print_table("by_distance", agg_on["by_distance"], agg_off["by_distance"])
    _print_table("by_skeleton_distance", agg_on["by_skeleton_distance"],
                 agg_off["by_skeleton_distance"])

    # JSON 落盘
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "checkpoint": args.checkpoint,
            "val": args.val,
            "max_episodes": args.max_episodes,
            "seg_len": seg_len,
            "NM_on": agg_on,
            "NM_zero": agg_off,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {args.output}")

    print("\n验证完成。")


if __name__ == "__main__":
    main()
