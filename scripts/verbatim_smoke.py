"""0b 训练 ckpt 的 verbatim smoke：teacher forcing + greedy 双路测试

用法: python scripts/verbatim_smoke.py --ckpt checkpoints/curriculum/0b_verbatim_rw.pt

对 N 个 verbatim episode:
  1. teacher forcing: 跑 setup → recall，看 recall turn value span token argmax 准确率
  2. greedy: setup 正常跑，recall user 后 autoregressive 生成，看是否输出 phrase
"""
import argparse
import sys
import random
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.config import XinheConfig
from xinhe.data.conversation import ensure_chat_template, tokenize_turn
from xinhe.data.curriculum_data import load_cache
from xinhe.data.patterns.phrase import generate_verbatim_recall_episode


@torch.no_grad()
def teacher_forcing_value_acc(model, tokenizer, episode, device, seg_len):
    """返回 recall turn 的 value span per-token 正确率 (0-1)"""
    ensure_chat_template(tokenizer)
    state = model.init_state(1).to(device)

    # turn 0: setup
    setup = episode[0]
    ids, labels, _ = tokenize_turn(tokenizer, setup["user"], setup["assistant"], seg_len,
                                    compute_loss=False)
    ids = ids.unsqueeze(0).to(device)
    out = model(ids, state, pad_token_id=tokenizer.pad_token_id)
    state = out["state_next"]

    # turn 1: recall (teacher forcing)
    recall = episode[1]
    value = recall["value"]
    ids, labels, _ = tokenize_turn(tokenizer, recall["user"], recall["assistant"], seg_len,
                                    compute_loss=True, value_str=value)
    ids_dev = ids.unsqueeze(0).to(device)
    labels_dev = labels.unsqueeze(0).to(device)
    out = model(ids_dev, state, labels=labels_dev, pad_token_id=tokenizer.pad_token_id)
    logits = out["logits"][0]  # (T, V)

    # 定位 value span token
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": recall["user"]},
         {"role": "assistant", "content": recall["assistant"]}],
        tokenize=False, add_generation_prompt=False,
    )
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": recall["user"]}],
        tokenize=False, add_generation_prompt=True,
    )
    v_start = full_text.find(value, len(prefix_text))
    if v_start < 0:
        return None
    v_end = v_start + len(value)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    preds = shift_logits.argmax(dim=-1).cpu()

    correct = total = 0
    for i in range(len(preds)):
        j = i + 1
        if j >= len(offsets):
            break
        cs, ce = offsets[j]
        if shift_labels[i].item() == -100:
            continue
        if cs < v_end and ce > v_start:
            total += 1
            if preds[i].item() == shift_labels[i].item():
                correct += 1
    return correct / max(total, 1), correct, total


@torch.no_grad()
def greedy_generation(model, tokenizer, episode, device, seg_len, max_gen=50):
    """setup 正常跑，recall user 后 greedy 生成，返回 (generated_text, phrase)"""
    ensure_chat_template(tokenizer)
    state = model.init_state(1).to(device)

    setup = episode[0]
    ids, _, _ = tokenize_turn(tokenizer, setup["user"], setup["assistant"], seg_len,
                                compute_loss=False)
    ids = ids.unsqueeze(0).to(device)
    out = model(ids, state, pad_token_id=tokenizer.pad_token_id)
    state = out["state_next"]

    recall = episode[1]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": recall["user"]}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text, add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    gen_ids = prompt_ids.clone()
    for _ in range(max_gen):
        out = model(gen_ids, state)
        next_id = out["logits"][0, -1].argmax().item()
        if next_id == tokenizer.eos_token_id:
            break
        gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=1)
        decoded = tokenizer.decode(gen_ids[0, prompt_ids.shape[1]:].tolist())
        if "<|im_end|>" in decoded:
            break

    generated = tokenizer.decode(gen_ids[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=False)
    generated = generated.split("<|im_end|>")[0].strip()
    return generated, recall["value"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="configs/persona_unified_0.8b.yaml")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    config, _ = XinheConfig.from_yaml(args.config)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    from scripts.evaluate import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(config, args.ckpt, device)
    model.eval()

    cache = load_cache("data/cache")
    rng = random.Random(args.seed)

    print(f"\n{'='*72}")
    print(f"  Verbatim Smoke: 0b ckpt = {args.ckpt}")
    print(f"  测试 {args.n} 个 2-turn episode (setup+recall)")
    print(f"{'='*72}")

    tf_total_c, tf_total_t = 0, 0
    gen_hit = 0
    for i in range(args.n):
        ep = generate_verbatim_recall_episode(rng, None, cache)
        if not ep:
            continue
        phrase = ep[-1]["value"]
        print(f"\n━━━ Episode {i+1} ━━━")
        print(f"  setup user : {ep[0]['user'][:60]}")
        print(f"  setup asst : {ep[0]['assistant'][:60]}")
        print(f"  recall user: {ep[1]['user'][:60]}")
        print(f"  target asst: {ep[1]['assistant'][:80]}")
        print(f"  phrase     : {phrase!r}")

        # teacher forcing
        tf = teacher_forcing_value_acc(model, tokenizer, ep, device, config.segment_length)
        if tf is not None:
            acc, c, t = tf
            tf_total_c += c
            tf_total_t += t
            print(f"  [TF] value span acc: {c}/{t} = {acc:.2%}")

        # greedy
        gen, val = greedy_generation(model, tokenizer, ep, device, config.segment_length)
        hit = phrase in gen
        gen_hit += int(hit)
        mark = "✓" if hit else "✗"
        print(f"  [Gen] {mark} generated: {gen!r}")

    print(f"\n{'='*72}")
    print(f"  汇总：")
    print(f"  TF value span: {tf_total_c}/{tf_total_t} = {tf_total_c/max(tf_total_t,1):.2%}")
    print(f"  Greedy 命中 phrase: {gen_hit}/{args.n} = {gen_hit/args.n:.2%}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
