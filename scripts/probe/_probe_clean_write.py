#!/usr/bin/env python
"""v18 判定探针 —— clean-write oracle。
假设:read=0% 根因是 M 被 distractor turn 污染(150:1 线性稀释)。
做法:对每个 recall turn,对比两种 M 下读正确 value token 的 rank:
  (A) polluted M:正常跨 turn thread(写入所有 turn,含 distractor)—— 现状
  (B) clean   M:只把"首次陈述该 value 的 fact turn"写进 fresh state —— 无 distractor
若 clean rank << polluted rank(甚至冲到 0)→ 稀释是瓶颈,v18=write-selectivity 铁证。
若 clean rank 仍很大 → 不是稀释,是 retrieve/容量问题。
forward-only no_grad。
"""
import sys, json, torch
sys.path.insert(0, ".")
from scripts.evaluate import load_model_and_tokenizer
from scripts.validate_memory import tokenize_turn, _locate_value_token

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/xinhe_step_3000.pt"
VAL = "data/skeleton/val.jsonl"
N_EP = 15

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
config = ckpt["config"]
config.compile_backbone_layers = False
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
seg_len = getattr(config, "turn_max_tokens", 256)
model, tokenizer = load_model_and_tokenizer(config, CKPT, device)
model.eval()
print(f"ckpt={CKPT}  mac_disabled={getattr(config,'mac_disabled',None)}")


def advance(state, user_msg, asst_msg, value_spans, train_loss, wps):
    ids, labels, _w = tokenize_turn(tokenizer, user_msg, asst_msg, seg_len,
                                    train_loss=train_loss, value_spans=value_spans,
                                    weight_per_span=wps)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(ids.unsqueeze(0).to(device), state,
                    pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    return out["state_next"]


def read_rank(state, user_msg, asst_text, char_span):
    full_ids, target_tok, target_pos = _locate_value_token(tokenizer, user_msg, asst_text, char_span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return None, None
    prefix_ids = full_ids[:target_pos]
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(inp, state, pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    logits = out["logits"][0, -1].detach().float()
    rank = int((logits > logits[target_tok]).sum().item())
    return rank, target_tok


rows = []
with open(VAL, encoding="utf-8") as fp:
    for line in fp:
        if len([r for r in rows]) >= N_EP:
            break
        ep = json.loads(line)
        convs = ep.get("conversations", [])
        # 预扫:turn 列表 (user, asst, value_spans, train_loss, wps)
        turns = []
        for i in range(0, len(convs) - 1, 2):
            a = convs[i]; b = convs[i + 1] if i + 1 < len(convs) else {}
            turns.append({
                "u": a.get("content", ""), "a": b.get("content", ""),
                "vs": b.get("value_span") or [],
                "tl": b.get("train_loss", "true"),
                "wps": float(b.get("weight_per_span", 0.0) or 0.0),
            })
        # 正常 thread,记录每个 turn 之前的 polluted state
        state = model.init_state(1).to(device)
        polluted_states = []
        for t in turns:
            polluted_states.append(state)
            state = advance(state, t["u"], t["a"], t["vs"], t["tl"], t["wps"])

        # 找第一个 recall turn + 其 fact turn
        for ti, t in enumerate(turns):
            done = False
            for s, e in t["vs"]:
                vstr = t["a"][s:e]
                if vstr in t["u"]:
                    continue  # write turn,不是 recall
                # fact turn = 最早一个 content 含 vstr 的更早 turn
                fact_idx = None
                for fj in range(ti):
                    if vstr in turns[fj]["u"] or vstr in turns[fj]["a"]:
                        fact_idx = fj
                        break
                if fact_idx is None:
                    continue
                # (A) polluted:用进入 recall turn 之前的完整 state
                rank_pol, tgt = read_rank(polluted_states[ti], t["u"], t["a"], (int(s), int(e)))
                # (B) clean:fresh state 只写 fact turn
                cs = model.init_state(1).to(device)
                ft = turns[fact_idx]
                cs = advance(cs, ft["u"], ft["a"], ft["vs"], ft["tl"], ft["wps"])
                rank_cl, _ = read_rank(cs, t["u"], t["a"], (int(s), int(e)))
                if rank_pol is None or rank_cl is None:
                    continue
                n_distract = ti - 1  # recall 前的 turn 数(粗略)
                rows.append((ep.get("skeleton_id"), vstr[:14], rank_pol, rank_cl, n_distract))
                print(f"  {ep.get('skeleton_id'):4s} '{vstr[:14]:14s}' polluted_rank={rank_pol:<6d} clean_rank={rank_cl:<6d} "
                      f"(clean命中={rank_cl==0}) Δrank={rank_pol-rank_cl:+d}  #turns_before={ti}")
                done = True
                break
            if done:
                break

# 汇总
print(f"\n===== 汇总 (n={len(rows)}) =====")
pol_hit = sum(1 for r in rows if r[2] == 0)
cl_hit = sum(1 for r in rows if r[3] == 0)
import statistics as st
print(f"  polluted M: first-token 命中(rank=0) = {pol_hit}/{len(rows)}   中位 rank = {st.median([r[2] for r in rows]):.0f}")
print(f"  clean    M: first-token 命中(rank=0) = {cl_hit}/{len(rows)}   中位 rank = {st.median([r[3] for r in rows]):.0f}")
better = sum(1 for r in rows if r[3] < r[2])
print(f"  clean rank < polluted rank 的比例 = {better}/{len(rows)}")
print(f"  -> {'稀释是瓶颈:清洁 write 显著改善 recall,v18=write-selectivity' if cl_hit > pol_hit or better > len(rows)*0.6 else '清洁 write 没救:不是稀释,是 retrieve/容量问题'}")
