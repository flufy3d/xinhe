"""
Persona 联合评估 (v7.1)

指标:
    WorldQA            — 世界常识 Q&A token argmax 准确率
    RefusalRate        — 空态问未披露 → 应含 refusal 关键词（不 fabricate）
    Compositional      — 多 fact 单 utterance 全 value 命中率
    RapidOverwrite     — 末轮应召回最新值
    Verbatim           — 随机 phrase 原样复述率（exact span match）
    ReferenceBack      — user quote-back 后 recall 已披露 slot
    ContextFollowup    — assistant 引用 persona 回答开放问题
    TopicContinuation  — 主题链末轮 sub-fact recall
    EntityTracking     — 代词消解 recall 第三方 entity 属性
    IrrelevantForget   — reveal + distractor 后 recall 原事实（stress_retention val）
    MultiSlotRetention — 多槽 retention（观察）

所有指标基于 teacher-forcing argmax + span match（和 VALUE 口径一致）。

v7.1 相对 v7 变化：
  删 eval_decay_refusal（decay 系列整体废弃）
  删 FORGET_REGEX / forget_detection_regex 依赖
  加 6 个新 eval（全部基于 _value_span_fullmatch）+ registry 注入
"""
import json
import re
from pathlib import Path

import torch

from xinhe.data.conversation import ensure_chat_template
from xinhe.data.refusal_templates import refusal_detection_regex
from xinhe.data.registry import VAL_FNS


REFUSAL_REGEX = re.compile(refusal_detection_regex())


@torch.no_grad()
def _forward_episode(model, tokenizer, episode, device, seg_len=256):
    """遍历 episode，state 跨 turn 传递，返回每 turn 的 preds/labels/user/asst/value/train_loss。"""
    ensure_chat_template(tokenizer)

    from xinhe.data.conversation import tokenize_turn
    conversations = episode.get("conversations", [])
    state = model.init_state(1).to(device)
    results = []

    for i in range(0, len(conversations) - 1, 2):
        user_msg = conversations[i].get("content", "")
        asst_entry = conversations[i + 1] if i + 1 < len(conversations) else {}
        assistant_msg = asst_entry.get("content", "")
        compute_loss = asst_entry.get("train_loss", True)
        value_str = asst_entry.get("value")

        ids, labels, _weights = tokenize_turn(
            tokenizer, user_msg, assistant_msg, seg_len,
            compute_loss=compute_loss,
            value_str=value_str,
        )
        ids = ids.unsqueeze(0).to(device)
        labels_dev = labels.unsqueeze(0).to(device)
        out = model(ids, state, labels=labels_dev, pad_token_id=tokenizer.pad_token_id)
        state = out["state_next"]
        logits = out["logits"][0]
        shift_logits = logits[:-1]
        shift_labels = labels[1:]
        preds = shift_logits.argmax(dim=-1).cpu()

        results.append({
            "preds": preds,
            "labels": shift_labels,
            "user": user_msg,
            "asst": assistant_msg,
            "value": value_str,
            "train_loss": compute_loss,
        })
    return results


def _token_accuracy_on_span(preds, labels, tokenizer, target_text: str,
                             full_text: str, search_from: int = 0) -> tuple[int, int]:
    """用 offset mapping 定位 target_text 区间 → argmax 准确率。"""
    if not target_text or not full_text:
        return 0, 0
    start = full_text.find(target_text, search_from)
    if start < 0:
        start = full_text.find(target_text)
        if start < 0:
            return 0, 0
    end = start + len(target_text)
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
        if cs < end and ce > start:
            total += 1
            if preds[i].item() == labels[i].item():
                correct += 1
    return correct, total


def _value_span_fullmatch(tr, tokenizer, value) -> bool:
    """对 tr，检查 value 在 assistant span 的 token argmax 是否全对。value 是 str 或 list。"""
    if isinstance(value, list):
        values_to_check = value
    else:
        values_to_check = [value]
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": tr["user"]},
         {"role": "assistant", "content": tr["asst"]}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": tr["user"]}],
        tokenize=False, add_generation_prompt=True,
    )
    search_from = len(prefix_text)
    for v in values_to_check:
        if not v:
            continue
        c, t = _token_accuracy_on_span(
            tr["preds"], tr["labels"], tokenizer, v, full_text,
            search_from=search_from,
        )
        if t == 0 or c < t:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 经典指标
# ═══════════════════════════════════════════════════════════════════

def eval_world_qa(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """单轮 QA 的 token argmax 准确率。"""
    path = Path(val_path)
    if not path.exists():
        return 0.0, 0
    correct = total = 0
    episodes_done = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if episodes_done >= max_episodes:
                break
            ep = json.loads(ln)
            turn_results = _forward_episode(model, tokenizer, ep, device, seg_len)
            for tr in turn_results:
                if not tr["train_loss"]:
                    continue
                mask = tr["labels"] != -100
                if mask.sum().item() == 0:
                    continue
                preds = tr["preds"][mask]
                tgts = tr["labels"][mask]
                correct += (preds == tgts).sum().item()
                total += tgts.numel()
            episodes_done += 1
    return correct / max(total, 1), episodes_done


def eval_refusal(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """末轮应拒答（不 fabricate）。"""
    path = Path(val_path)
    if not path.exists():
        return 0.0, 0
    refused = fabricated = 0
    episodes_done = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if episodes_done >= max_episodes:
                break
            ep = json.loads(ln)
            turn_results = _forward_episode(model, tokenizer, ep, device, seg_len)
            refusal_turn = None
            for tr in reversed(turn_results):
                if tr["train_loss"] and not tr["value"]:
                    refusal_turn = tr
                    break
            if refusal_turn is None:
                continue
            mask = refusal_turn["labels"] != -100
            if mask.sum().item() == 0:
                continue
            pred_ids = refusal_turn["preds"][mask].tolist()
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            if REFUSAL_REGEX.search(pred_text):
                refused += 1
            else:
                fabricated += 1
            episodes_done += 1
    total = refused + fabricated
    return refused / max(total, 1), episodes_done


def eval_compositional(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """末轮多 fact 单 utterance，所有 value token 全对。"""
    path = Path(val_path)
    if not path.exists():
        return 0.0, 0
    all_right = total_eps = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if total_eps >= max_episodes:
                break
            ep = json.loads(ln)
            turn_results = _forward_episode(model, tokenizer, ep, device, seg_len)
            comp_turn = None
            for tr in reversed(turn_results):
                v = tr["value"]
                if tr["train_loss"] and isinstance(v, list) and len(v) >= 2:
                    comp_turn = tr
                    break
            if comp_turn is None:
                continue
            total_eps += 1
            if _value_span_fullmatch(comp_turn, tokenizer, comp_turn["value"]):
                all_right += 1
    return all_right / max(total_eps, 1), total_eps


# ═══════════════════════════════════════════════════════════════════
# 通用末轮 value match eval（retention / continuity / verbatim 等）
# ═══════════════════════════════════════════════════════════════════

def eval_last_turn_value_match(
    model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100,
):
    """通用 eval：取最后一个 value-bearing turn，检查 value span token argmax 全对率。"""
    path = Path(val_path)
    if not path.exists():
        return 0.0, 0
    hit = total = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if total >= max_episodes:
                break
            ln = ln.strip()
            if not ln:
                continue
            ep = json.loads(ln)
            trs = _forward_episode(model, tokenizer, ep, device, seg_len)
            last = trs[-1] if trs else None
            if last is None or not last["train_loss"] or not last["value"]:
                continue
            total += 1
            if _value_span_fullmatch(last, tokenizer, last["value"]):
                hit += 1
    return hit / max(total, 1), total


# 兼容别名（trainer / chat_smoke 可能调用）
eval_rapid_overwrite = eval_last_turn_value_match


# ═══════════════════════════════════════════════════════════════════
# Registry patching：给所有 val 生成器附上 eval_fn
# 这是 persona_joint.py 最后一步，避免 patterns/ 里的循环 import
# ═══════════════════════════════════════════════════════════════════

def _patch_val_eval_fns():
    """把通用 eval_last_turn_value_match 注入所有注册的 val（没有自定义 eval 的）。"""
    for name, (gen_fn, eval_fn) in list(VAL_FNS.items()):
        if eval_fn is None:
            VAL_FNS[name] = (gen_fn, eval_last_turn_value_match)


_patch_val_eval_fns()


# ═══════════════════════════════════════════════════════════════════
# Joint eval 入口
# ═══════════════════════════════════════════════════════════════════

# val 集 → (eval_fn, config 路径字段名) 映射
_JOINT_METRIC_SPECS = [
    ("world_qa",           eval_world_qa,        "val_worldqa_path"),
    ("refusal",            eval_refusal,         "val_refusal_path"),
    ("compositional",      eval_compositional,   "val_compositional_path"),
    ("rapid_overwrite",    eval_rapid_overwrite, "val_rapid_overwrite_path"),
    ("verbatim",           eval_last_turn_value_match, "val_verbatim_path"),
    ("reference_back",     eval_last_turn_value_match, "val_reference_back_path"),
    ("context_followup",   eval_last_turn_value_match, "val_context_followup_path"),
    ("topic_continuation", eval_last_turn_value_match, "val_topic_continuation_path"),
    ("entity_tracking",    eval_last_turn_value_match, "val_entity_tracking_path"),
    ("irrelevant_forget",  eval_last_turn_value_match, "val_irrelevant_forget_path"),
    ("multi_slot_retention", eval_last_turn_value_match, "val_multi_slot_retention_path"),
]


def eval_persona_joint(model, tokenizer, config, device, max_episodes: int = 50) -> dict:
    """一次调用跑所有 joint 指标。路径缺失或不存在 → 对应指标返回 0.0。"""
    seg_len = getattr(config, "segment_length", 256)
    result = {}
    for name, eval_fn, path_attr in _JOINT_METRIC_SPECS:
        path = getattr(config, path_attr, "") or ""
        if not path:
            result[name] = 0.0
            continue
        score, _ = eval_fn(model, tokenizer, path, device, seg_len, max_episodes)
        result[name] = score
    return result
