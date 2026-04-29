"""
v8 事件级评估。

输入：v8 schema JSONL 验证集（带 skeleton_id / distance_bucket / value_tier / substream）。
输出：按多维度聚合的 token-level argmax 准确率字典。

判定规则：
  - 对每个 assistant turn 的 value_span，通过 tokenizer offset_mapping 定位 token
  - 对每个 token，检查 model.argmax == label
  - 该 turn 全 token 正确才算 "value 命中"
  - 多 value 同时全对才算整 turn 命中

按维度聚合:
  - skeleton_id (S1..S11)：Stage 0 的事件骨架
  - distance_bucket (near/mid/far/very_far)
  - value_tier (hard / soft)
  - substream (1A / 1B)：Stage 1 的子流来源
  - relation：从 meta.canonical_facts 或 ctx 抽取（粗粒度，目前主要用于 1A）
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import torch

from xinhe.data.conversation import ensure_chat_template, tokenize_turn


@torch.no_grad()
def _forward_episode(
    model,
    tokenizer,
    episode: dict,
    *,
    device: str,
    seg_len: int = 256,
):
    """跑一条 episode，返回每个 assistant turn 的 (preds, labels, user, asst, span_info)。"""
    ensure_chat_template(tokenizer)
    convs = episode.get("conversations", [])
    state = model.init_state(1).to(device)
    out_records = []

    for i in range(0, len(convs) - 1, 2):
        user_msg = convs[i].get("content", "")
        asst_entry = convs[i + 1] if i + 1 < len(convs) else {}
        assistant_msg = asst_entry.get("content", "")
        train_loss = asst_entry.get("train_loss", "true")
        value_spans = asst_entry.get("value_span") or []
        weight_per_span = float(asst_entry.get("weight_per_span", 0.0) or 0.0)

        ids, labels, _w = tokenize_turn(
            tokenizer, user_msg, assistant_msg, seg_len,
            train_loss=train_loss,
            value_spans=value_spans,
            weight_per_span=weight_per_span,
        )
        ids_dev = ids.unsqueeze(0).to(device)
        labels_dev = labels.unsqueeze(0).to(device)
        out = model(ids_dev, state, labels=labels_dev, pad_token_id=tokenizer.pad_token_id)
        state = out["state_next"]
        logits = out["logits"][0]
        shift_logits = logits[:-1]
        shift_labels = labels[1:]
        preds = shift_logits.argmax(dim=-1).cpu()

        out_records.append({
            "preds": preds,
            "labels": shift_labels,
            "user": user_msg,
            "asst": assistant_msg,
            "value": asst_entry.get("value"),
            "value_span": value_spans,   # 仍是 char span（assistant_content 坐标系）
            "value_tier": asst_entry.get("value_tier"),
            "train_loss": train_loss,
        })
    return out_records


def _token_accuracy_on_char_span(
    preds, labels, tokenizer,
    user_text: str,
    asst_text: str,
    char_span_in_asst: tuple[int, int],
) -> tuple[int, int]:
    """通过 offset_mapping，把 (start, end) char span (相对 asst_text) 映射到 token，比对 argmax。

    Returns: (correct, total)
    """
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": asst_text}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )
    asst_offset = full_text.find(asst_text, len(prefix_text))
    if asst_offset < 0:
        asst_offset = full_text.find(asst_text)
    if asst_offset < 0:
        return 0, 0

    s_full = asst_offset + char_span_in_asst[0]
    e_full = asst_offset + char_span_in_asst[1]

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    correct = total = 0
    for i in range(len(preds)):
        j = i + 1
        if j >= len(offsets):
            break
        cs, ce = offsets[j]
        if labels[i].item() == -100:
            continue
        if cs < e_full and ce > s_full:
            total += 1
            if preds[i].item() == labels[i].item():
                correct += 1
    return correct, total


def _turn_value_fullmatch(turn_record: dict, tokenizer) -> Optional[bool]:
    """对一个 assistant turn，检查 value spans 是否全部 token-level 命中。

    Returns:
      None: turn 无 value（不参与统计）
      True: 全部 value 全对
      False: 至少一个 value 错或定位失败
    """
    spans = turn_record.get("value_span") or []
    if not spans:
        return None
    for s, e in spans:
        c, t = _token_accuracy_on_char_span(
            turn_record["preds"], turn_record["labels"], tokenizer,
            turn_record["user"], turn_record["asst"],
            (int(s), int(e)),
        )
        if t == 0 or c < t:
            return False
    return True


def eval_event_jsonl(
    model,
    tokenizer,
    val_path: str | Path,
    *,
    device: str,
    seg_len: int = 256,
    max_episodes: int = 50,
) -> dict:
    """跑一个 jsonl 数据集，按多维度聚合命中率。

    Returns dict with keys:
      overall_acc
      n_episodes, n_value_turns, n_correct
      by_skeleton: {S1: {acc, n}, ...}
      by_distance: {...}
      by_tier: {hard: {...}, soft: {...}}
      by_substream: {1A: {...}, 1B: {...}}
    """
    p = Path(val_path)
    if not p.exists():
        return {"overall_acc": 0.0, "n_episodes": 0, "n_value_turns": 0, "n_correct": 0}

    model.eval()
    n_eps = 0
    total_value_turns = 0
    total_correct = 0

    bucket_by_skeleton: dict[str, list[bool]] = defaultdict(list)
    bucket_by_distance: dict[str, list[bool]] = defaultdict(list)
    bucket_by_tier: dict[str, list[bool]] = defaultdict(list)
    bucket_by_substream: dict[str, list[bool]] = defaultdict(list)

    with open(p, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            if n_eps >= max_episodes:
                break
            ep = json.loads(line)
            skeleton_id = ep.get("skeleton_id")
            meta = ep.get("meta") or {}
            distance = meta.get("distance_bucket")
            substream = meta.get("substream")

            try:
                turn_records = _forward_episode(
                    model, tokenizer, ep, device=device, seg_len=seg_len,
                )
            except Exception as e:
                continue

            n_eps += 1

            for tr in turn_records:
                ok = _turn_value_fullmatch(tr, tokenizer)
                if ok is None:
                    continue
                total_value_turns += 1
                if ok:
                    total_correct += 1
                if skeleton_id:
                    bucket_by_skeleton[skeleton_id].append(ok)
                if distance:
                    bucket_by_distance[distance].append(ok)
                tier = tr.get("value_tier")
                if tier in ("hard", "soft"):
                    bucket_by_tier[tier].append(ok)
                if substream:
                    bucket_by_substream[substream].append(ok)

    def _summarize(buckets: dict[str, list[bool]]) -> dict[str, dict]:
        out = {}
        for k, vals in buckets.items():
            if not vals:
                continue
            out[k] = {"acc": sum(vals) / len(vals), "n": len(vals)}
        return out

    overall = total_correct / max(1, total_value_turns)
    return {
        "overall_acc": overall,
        "n_episodes": n_eps,
        "n_value_turns": total_value_turns,
        "n_correct": total_correct,
        "by_skeleton": _summarize(bucket_by_skeleton),
        "by_distance": _summarize(bucket_by_distance),
        "by_tier": _summarize(bucket_by_tier),
        "by_substream": _summarize(bucket_by_substream),
    }


def eval_joint_v8(
    model,
    tokenizer,
    config,
    *,
    device: str,
    max_episodes: int = 50,
) -> dict:
    """对 config 中声明的 val 路径全部跑一遍，扁平化为 metric 名 → 准确率的字典。

    config 可携带 val_sets: list[{"name": str, "path": str}]
    若没显式声明，回退到默认结构（data/val/stage{0,1}/*.jsonl）。

    Returns flat dict like:
      {
        "stage0_seen_overall": 0.92,
        "stage0_seen_S1": 0.95,
        "stage0_seen_S5": 0.83,
        "stage0_seen_distance_near": 0.95,
        ...
        "stage1_substream_1A": 0.88,
        "stage1_substream_1B": 0.79,
      }
    """
    val_sets = getattr(config, "val_sets", None) or _default_val_sets()
    seg_len = getattr(config, "turn_max_tokens", 256)

    flat: dict[str, float] = {}
    for vset in val_sets:
        name = vset.get("name", "unknown")
        path = vset.get("path")
        if not path:
            continue
        res = eval_event_jsonl(
            model, tokenizer, path,
            device=device, seg_len=seg_len, max_episodes=max_episodes,
        )
        flat[f"{name}_overall"] = res["overall_acc"]
        for k, v in res.get("by_skeleton", {}).items():
            flat[f"{name}_{k}"] = v["acc"]
        for k, v in res.get("by_distance", {}).items():
            flat[f"{name}_distance_{k}"] = v["acc"]
        for k, v in res.get("by_tier", {}).items():
            flat[f"{name}_tier_{k}"] = v["acc"]
        for k, v in res.get("by_substream", {}).items():
            flat[f"{name}_substream_{k}"] = v["acc"]
    return flat


def _default_val_sets() -> list[dict]:
    """val_sets 缺失时的 fallback 默认。"""
    candidates = [
        ("stage0_val", "data/v8/stage0/val.jsonl"),
        ("stage1_val", "data/v8/stage1/val.jsonl"),
    ]
    out = []
    for name, path in candidates:
        if Path(path).exists():
            out.append({"name": name, "path": path})
    return out
