#!/usr/bin/env python
"""精确复现 validate_memory 的两遍 pass:对同一批 episode,分别用 override=None 和 0.0
跑完整 evaluate_episode_strict,逐 recall turn 打印 first_token_correct 是否不同。
若两遍逐 turn 相同 -> 真 bug 在 forward 的 override 路径;若不同 -> 落盘 JSON 是旧的/缓存。
"""
import sys, json, torch
sys.path.insert(0, ".")
from scripts.evaluate import load_model_and_tokenizer
from scripts.validate_memory import evaluate_episode_strict

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/xinhe_step_1500.pt"
VAL = "data/skeleton/val.jsonl"
N_EP = 8

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
config = ckpt["config"]
config.compile_backbone_layers = False
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
seg_len = getattr(config, "turn_max_tokens", 256)
model, tokenizer = load_model_and_tokenizer(config, CKPT, device)
model.eval()

eps = []
with open(VAL, encoding="utf-8") as fp:
    for line in fp:
        if len(eps) >= N_EP:
            break
        eps.append(json.loads(line))

n_diff = 0
n_total = 0
for idx, ep in enumerate(eps):
    rec_on = evaluate_episode_strict(model, tokenizer, ep, device, seg_len, None)
    rec_ze = evaluate_episode_strict(model, tokenizer, ep, device, seg_len, 0.0)
    assert len(rec_on) == len(rec_ze)
    for a, b in zip(rec_on, rec_ze):
        if not a["is_recall"]:
            continue
        n_total += 1
        ft_on = a["first_token_correct"]
        ft_ze = b["first_token_correct"]
        if ft_on != ft_ze:
            n_diff += 1
            print(f"  ep#{idx} recall '{a['value'][:15]}': first_token NM-on={ft_on} NM-zero={ft_ze}  <-- 不同!")

print(f"\n===== 汇总 =====")
print(f"  recall turn 总数 = {n_total}")
print(f"  两遍 first_token_correct 不同的 turn 数 = {n_diff}")
print(f"  -> {'两遍确有差异(JSON 落盘是旧/缓存数据)' if n_diff>0 else '两遍逐 turn 完全相同(真 bug:override 在多 turn forward 未生效)'}")
