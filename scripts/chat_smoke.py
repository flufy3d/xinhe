"""
人工验收脚本 — 测 persona_unified.pt 是否修好三大问题。

用法:
    python scripts/chat_smoke.py --ckpt checkpoints/curriculum/persona_unified.pt

测试场景:
    [A] 世界知识（Problem 3）: 单轮 QA，无状态
    [B] 拒答（Problem 1）: 空状态 + 问未告知信息 → 应 refuse
    [C] 多 fact 单句（Problem 2）: 一句话 3 fact → 依次召回
    [D] 跨轮 overwrite: 事实覆写 → 召回最新值
"""
import argparse
import re
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.data.conversation import ensure_chat_template
from xinhe.data.refusal_templates import refusal_detection_regex


REFUSAL_RE = re.compile(refusal_detection_regex())


def load_model(config_path: str, ckpt: str, device):
    from scripts.evaluate import load_model_and_tokenizer
    config, _ = XinheConfig.from_yaml(config_path)
    model, tokenizer = load_model_and_tokenizer(config, ckpt, device)
    ensure_chat_template(tokenizer)
    return model, tokenizer, config


@torch.no_grad()
def generate(model, tokenizer, state, user_msg: str, max_new_tokens=128,
             temperature=0.1):
    """单轮对话生成，返回 (assistant_text, new_state)。"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=torch.long,
    ).unsqueeze(0).to(state.device)
    # 生成
    generated, new_state = model.generate_with_state(
        ids, state, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.9, repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 解码生成部分
    new_tokens = generated[0, ids.shape[1]:].tolist()
    if tokenizer.eos_token_id in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(tokenizer.eos_token_id)]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # 若遇到 ChatML 标记就截断
    for stop in ["<|im_end|>", "<|im_start|>"]:
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text, new_state


def run_turn(model, tokenizer, state, user, label="", expect=None):
    """跑一轮 + 打印。expect 是可选的关键词列表，命中任一算过。"""
    resp, state = generate(model, tokenizer, state, user)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}U: {user}")
    print(f"{prefix}A: {resp}")
    if expect is not None:
        hit = any(e in resp for e in expect)
        ok = "✓" if hit else "✗"
        print(f"{prefix}  expect {expect}: {ok}")
    print()
    return resp, state


def test_A_world_knowledge(model, tokenizer, device):
    """世界知识恢复：不依赖 state。"""
    print("=" * 60)
    print("[A] 世界知识（Problem 3: 原来答'新加拿地'）")
    print("=" * 60)
    cases = [
        ("巴黎是哪个国家的首都？", ["法国"]),
        ("豆腐是哪国发明的？", ["中国"]),
        ("珠穆朗玛峰有多高？", ["8848", "8844", "8000"]),
        ("太阳系里最大的行星是哪个？", ["木星"]),
        ("水的化学式是什么？", ["H2O", "H₂O", "水是"]),
    ]
    hits = 0
    for q, expects in cases:
        state = model.init_state(1).to(device)
        r, _ = run_turn(model, tokenizer, state, q, "A", expects)
        if any(e in r for e in expects):
            hits += 1
    print(f"[A] 世界知识: {hits}/{len(cases)} 正确")
    return hits, len(cases)


def test_B_refusal(model, tokenizer, device):
    """拒答：空状态被问个人信息应拒答，不 fabricate。"""
    print("=" * 60)
    print("[B] 拒答（Problem 1: 原来会编造）")
    print("=" * 60)
    cases = [
        "我叫什么名字？",
        "我多大了？",
        "我住在哪？",
        "我的爱好是什么？",
        "我养了什么宠物？",
    ]
    refused = 0
    for q in cases:
        state = model.init_state(1).to(device)
        r, _ = run_turn(model, tokenizer, state, q, "B")
        if REFUSAL_RE.search(r):
            refused += 1
            print(f"  → 判定: 拒答 ✓\n")
        else:
            print(f"  → 判定: 未拒答 ✗（fabricate 风险）\n")
    print(f"[B] 拒答率: {refused}/{len(cases)}")
    return refused, len(cases)


def test_C_multi_fact(model, tokenizer, device):
    """多 fact 单句（Problem 2）: 一句话讲 3 件事，依次召回。"""
    print("=" * 60)
    print("[C] 多 fact 单句（Problem 2: 原来崩盘）")
    print("=" * 60)
    state = model.init_state(1).to(device)
    # 告知
    _, state = run_turn(
        model, tokenizer, state,
        "我叫陈杰，今年35岁，爱弹吉他。", "C",
    )
    # 分别召回
    cases = [
        ("我叫什么名字？", ["陈杰"]),
        ("我多大了？", ["35"]),
        ("我的爱好是什么？", ["吉他"]),
    ]
    hits = 0
    for q, expects in cases:
        r, state = run_turn(model, tokenizer, state, q, "C", expects)
        if any(e in r for e in expects):
            hits += 1
    print(f"[C] 多 fact 召回: {hits}/{len(cases)}")
    return hits, len(cases)


def test_E_retention(model, tokenizer, device):
    """穿插场景：告知事实 → N 轮世界 QA → 召回事实。用户实测失败的场景。"""
    print("=" * 60)
    print("[E] 穿插召回（用户实测失败的场景）")
    print("=" * 60)
    scenarios = [
        # (reveal, [chat_prompts], recall_q, expect_in_answer)
        ("我叫陈杰。", ["你好", "今天天气不错"], "我叫什么？", ["陈杰"]),
        ("我叫陈杰。", ["豆腐是哪国发明的", "巴黎在哪", "木星有多大"], "我叫啥？", ["陈杰"]),
        ("我叫李雷，今年30岁。", ["你会弹吉他吗", "推荐本书", "周末想旅行"], "我叫啥？", ["李雷"]),
        ("我住在杭州。", ["什么是AI", "Python 简单吗"], "我住哪？", ["杭州"]),
        ("我叫陈杰。", ["你好"] + ["介绍一下自己"] * 2, "我叫什么？", ["陈杰"]),
    ]
    hits = 0
    for reveal, chats, recall_q, expects in scenarios:
        state = model.init_state(1).to(device)
        _, state = run_turn(model, tokenizer, state, reveal, "E")
        for c in chats:
            _, state = run_turn(model, tokenizer, state, c, "E")
        r, state = run_turn(model, tokenizer, state, recall_q, "E", expects)
        if any(e in r for e in expects):
            hits += 1
        print(f"  scenario 小结 ({len(chats)} 轮 chat): {'✓' if any(e in r for e in expects) else '✗'}\n")
    print(f"[E] 穿插召回: {hits}/{len(scenarios)}")
    return hits, len(scenarios)


def test_F_multi_slot_retention(model, tokenizer, device):
    """多槽 retention：用户实测失败的 pattern（先后告知 2-3 槽，chat，依次 recall）。"""
    print("=" * 60)
    print("[F] 多槽 retention（用户 turn 6 忘年龄的 pattern）")
    print("=" * 60)
    scenarios = [
        # (reveals, chats, recall_list)
        (["我叫陈杰", "我朋友叫王林", "我35岁"],
         ["巴黎在哪"],
         [("我朋友叫啥", ["王林"]),
          ("我叫什么", ["陈杰"]),
          ("我多大了", ["35"])]),
        (["我叫李雷", "我住在杭州"],
         ["什么是Python", "周末想旅行"],
         [("我叫啥", ["李雷"]),
          ("我住哪", ["杭州"])]),
        (["我今年40岁", "我最喜欢吃火锅", "我养了只猫"],
         ["巴黎在哪"],
         [("我多大", ["40"]),
          ("我爱吃啥", ["火锅"]),
          ("我养啥宠物", ["猫"])]),
    ]
    total_hits = 0
    total_queries = 0
    for reveals, chats, recalls in scenarios:
        state = model.init_state(1).to(device)
        print(f"\n--- 场景 ({len(reveals)} 槽, {len(chats)} chat, {len(recalls)} recall) ---")
        for rev in reveals:
            _, state = run_turn(model, tokenizer, state, rev, "F")
        for c in chats:
            _, state = run_turn(model, tokenizer, state, c, "F")
        hits_here = 0
        for q, expects in recalls:
            r, state = run_turn(model, tokenizer, state, q, "F", expects)
            if any(e in r for e in expects):
                hits_here += 1
                total_hits += 1
            total_queries += 1
        print(f"  该场景 {hits_here}/{len(recalls)}")
    print(f"[F] 多槽 retention: {total_hits}/{total_queries}")
    return total_hits, total_queries


def test_D_overwrite(model, tokenizer, device):
    """Overwrite: 先告诉，再纠正，问最新值。"""
    print("=" * 60)
    print("[D] 覆写（Delta Rule 原生能力，应该稳）")
    print("=" * 60)
    state = model.init_state(1).to(device)
    _, state = run_turn(model, tokenizer, state, "我叫小明。", "D")
    _, state = run_turn(model, tokenizer, state, "哦不对，我其实叫李雷。", "D")
    r, state = run_turn(model, tokenizer, state, "我叫什么名字？", "D", ["李雷"])
    hit = int("李雷" in r)
    print(f"[D] 覆写召回: {hit}/1")
    return hit, 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="checkpoints/curriculum/persona_unified.pt")
    p.add_argument("--config", type=str, default="configs/qwen3.5-0.8b.yaml")
    args = p.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"加载 {args.ckpt} ...")
    from scripts.evaluate import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config, args.ckpt, device)
    ensure_chat_template(tokenizer)
    model.eval()

    a_hit, a_total = test_A_world_knowledge(model, tokenizer, device)
    b_hit, b_total = test_B_refusal(model, tokenizer, device)
    c_hit, c_total = test_C_multi_fact(model, tokenizer, device)
    d_hit, d_total = test_D_overwrite(model, tokenizer, device)
    e_hit, e_total = test_E_retention(model, tokenizer, device)
    f_hit, f_total = test_F_multi_slot_retention(model, tokenizer, device)

    print("=" * 60)
    print("  人工验收总结")
    print("=" * 60)
    print(f"  [A] 世界知识:         {a_hit}/{a_total}")
    print(f"  [B] 拒答:             {b_hit}/{b_total}")
    print(f"  [C] 多 fact 召回:      {c_hit}/{c_total}")
    print(f"  [D] 覆写召回:         {d_hit}/{d_total}")
    print(f"  [E] 单槽穿插召回:      {e_hit}/{e_total}")
    print(f"  [F] 多槽 retention:   {f_hit}/{f_total}   ← 用户 turn 6 丢年龄的 pattern")
    total_hit = a_hit + b_hit + c_hit + d_hit + e_hit + f_hit
    total = a_total + b_total + c_total + d_total + e_total + f_total
    print(f"  合计:                 {total_hit}/{total} ({total_hit/total:.1%})")


if __name__ == "__main__":
    main()
