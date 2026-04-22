"""
Persona 统一训练的 3 个新指标联合评估（配合 eval_value_breakdown_fast 形成 4 指标联合早停）。

指标:
    WorldQA        — 世界常识 Q&A 准确率（token-level argmax match）
    RefusalRate    — "问未披露槽 → 应拒答" 的识别率（含 fabrication 扫描）
    Compositional  — 多 fact 单 utterance / 跨槽组合的全 fact VALUE 准确率

设计：所有指标都基于 teacher-forcing argmax（和 VALUE/FRAME/TELL 一致），
而非 greedy generation，这样：
    (1) 计算成本低（一次 forward vs N 次自回归）
    (2) 和 train loss 口径一致
    (3) 无需采样随机性

每个 val 集是标准的 conversations 格式 JSONL，由 generate_persona_data 产出。
"""
import json
import re
from pathlib import Path

import torch

from xinhe.data.conversation import ensure_chat_template
from xinhe.data.refusal_templates import refusal_detection_regex


REFUSAL_REGEX = re.compile(refusal_detection_regex())


@torch.no_grad()
def _forward_episode(model, tokenizer, episode, device, seg_len=256):
    """遍历一个 episode，state 跨 turn 传递，返回每个 turn 的 (preds, labels, asst_text, user_text, value) 序列。

    每 turn 返回:
        - preds: (T,) 预测 token ids（argmax）
        - labels: (T,) 真实 labels（-100 为 ignore）
        - asst_text: assistant 实际字符串
        - user_text: user 字符串
        - value: "value" 字段（可能是 str / list / None）
        - train_loss: bool
    """
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
        logits = out["logits"][0]     # (T, V)
        shift_logits = logits[:-1]
        shift_labels = labels[1:]     # CPU
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
    """用 offset mapping 定位 target_text 在 full_text 里的 token 区间，统计 argmax 准确率。

    关键：value 字符串通常在 user 部分和 assistant 部分**都出现**（用户说 + 助手回重复）。
    user 部分 labels 全 -100，匹配上也算不到分，所以必须从 assistant 起点往后找。
    search_from 是 assistant 内容在 full_text 里的 char offset 下限。

    shift-by-1 对齐：
    - `preds` / `labels` 都已经 shift 过（长度 T-1）
    - `preds[i]` 预测的是原始 input_ids 位置 i+1 的 token
    - 于是 offsets 用 i+1 索引
    返回 (correct, total)。"""
    if not target_text or not full_text:
        return 0, 0
    start = full_text.find(target_text, search_from)
    if start < 0:
        # fallback: 全文找（某些情况下 value 只出现在 assistant）
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


def eval_world_qa(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """WorldQA: 逐条 single-turn QA，统计 assistant answer 的整体 token argmax 准确率。

    val 集：每行一个 episode，只有一轮（user 问题 + assistant 答案），train_loss=true。
    """
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

    acc = correct / max(total, 1)
    return acc, episodes_done


def eval_refusal(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """RefusalRate:
        每个 episode 最后一轮是"问未披露槽"，模型应产生含 refusal 关键词的回复且不 fabricate。

    衡量方式（保持 teacher-forcing 口径，不做 generation）:
        - 取最后一个 train_loss=true 的 turn，其 value=None（标记为 refusal 轮）
        - 用模型预测的 token argmax 拼成 predicted answer
        - 如果预测里包含 refusal regex 关键词 → refused=1
        - 否则 → 认为是 fabrication
    """
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

            # 找最后一个 train_loss=true 且 value 为空的 turn（refusal 轮）
            refusal_turn = None
            for tr in reversed(turn_results):
                if tr["train_loss"] and not tr["value"]:
                    refusal_turn = tr
                    break
            if refusal_turn is None:
                continue

            # 从 preds 拼 predicted assistant answer 文本
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
    rate = refused / max(total, 1)
    return rate, episodes_done


def eval_compositional(model, tokenizer, val_path: str, device, seg_len=256, max_episodes=100):
    """Compositional: 多 fact 单 utterance 准确率。

    val 集: 每 episode 最后一轮是多 fact 单 utterance，value=list[str]。
    衡量: 所有 value 的 token 都预测对才算一个 episode "全对"，返回全对率。
    """
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

            # 找最后一个 train_loss=true 且 value 是 list 的 turn
            comp_turn = None
            for tr in reversed(turn_results):
                v = tr["value"]
                if tr["train_loss"] and isinstance(v, list) and len(v) >= 2:
                    comp_turn = tr
                    break
            if comp_turn is None:
                continue

            total_eps += 1
            asst = comp_turn["asst"]
            user = comp_turn["user"]
            # 重建 full_text 用于 offset 定位（和 tokenize_turn 逻辑一致）
            full_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user},
                 {"role": "assistant", "content": asst}],
                tokenize=False, add_generation_prompt=False,
            )
            # 找 assistant 部分起点（跳过 user 部分，避免 find 撞到 user 里的 value 重复）
            prefix_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            asst_char_start = len(prefix_text)

            ep_all_right = True
            for v in comp_turn["value"]:
                c, t = _token_accuracy_on_span(
                    comp_turn["preds"], comp_turn["labels"], tokenizer, v, full_text,
                    search_from=asst_char_start,
                )
                if t == 0 or c < t:
                    ep_all_right = False
                    break
            if ep_all_right:
                all_right += 1

    rate = all_right / max(total_eps, 1)
    return rate, total_eps


def eval_persona_joint(model, tokenizer, config, device, max_episodes: int = 50) -> dict:
    """一次调用跑 3 个新指标，返回 dict {"world_qa", "refusal", "compositional"}。

    每个指标对应自己的 val jsonl 路径（config.val_worldqa_path 等）。
    路径为空或文件不存在时指标返回 0.0（trainer 会报告 "未达标"）。
    """
    seg_len = getattr(config, "segment_length", 256)
    result = {}

    wq_path = getattr(config, "val_worldqa_path", "") or ""
    rf_path = getattr(config, "val_refusal_path", "") or ""
    cp_path = getattr(config, "val_compositional_path", "") or ""

    wq_acc, _ = eval_world_qa(model, tokenizer, wq_path, device, seg_len, max_episodes)
    rf_rate, _ = eval_refusal(model, tokenizer, rf_path, device, seg_len, max_episodes)
    cp_rate, _ = eval_compositional(model, tokenizer, cp_path, device, seg_len, max_episodes)

    result["world_qa"] = wq_acc
    result["refusal"] = rf_rate
    result["compositional"] = cp_rate
    return result
