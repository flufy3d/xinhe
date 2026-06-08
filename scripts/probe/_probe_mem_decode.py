#!/usr/bin/env python
"""v18 最终判定 —— 纯 memory decode。
绕开 backbone,只看 memory readout 经 nm_aux 那条路 lm_head(W_mac(mem_out)) 能否解出正确 value。
对每个 recall turn:正常 thread state 到该 turn,跑 read forward 抓 mem_out,
计算 logits_mem = lm_head(W_mac(mem_out).mean(1)),报正确 value token 在 logits_mem 里的 rank。
对照:同一 mem_out 但用 clean M(只写 fact turn)。
若 rank 小 → memory 编码了 value,瓶颈是注入;若 rank 大 → memory 没编码 value。
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
lm_head_w = model.lm_head.weight if hasattr(model, "lm_head") else model.backbone.get_lm_head().weight
print(f"ckpt={CKPT}  mac_disabled={getattr(config,'mac_disabled',None)}  lm_head={tuple(lm_head_w.shape)}")

# 抓 mem_out:patch W_mac 记录 input
rec = {}
_orig = model.W_mac.forward
def _spy(x):
    rec["mem_out"] = x.detach()
    return _orig(x)
model.W_mac.forward = _spy


def advance(state, t):
    ids, _l, _w = tokenize_turn(tokenizer, t["u"], t["a"], seg_len,
                                train_loss=t["tl"], value_spans=t["vs"], weight_per_span=t["wps"])
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(ids.unsqueeze(0).to(device), state,
                    pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    return out["state_next"]


def mem_decode_rank(state, user_msg, asst_text, char_span):
    """跑 read forward 抓 mem_out → lm_head(W_mac(mem_out).mean) → 正确 value token rank。"""
    full_ids, target_tok, target_pos = _locate_value_token(tokenizer, user_msg, asst_text, char_span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return None, None
    inp = torch.tensor([full_ids[:target_pos]], dtype=torch.long, device=device)
    rec.clear()
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        model(inp, state, pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    mem_out = rec.get("mem_out")
    if mem_out is None:
        return None, None
    # mac_tokens = W_mac(mem_out) (B,n_q,hidden);nm_aux 是每个 query token 独立预测 value
    mac = model.W_mac(mem_out.float())[0]                      # (n_q, hidden)
    logits_all = mac.float() @ lm_head_w.float().T            # (n_q, vocab)
    # mean-pool rank
    lm_mean = logits_all.mean(dim=0)
    rank_mean = int((lm_mean > lm_mean[target_tok]).sum().item())
    # 每个 query token 各自 rank,取最好(min)
    ranks_pt = [(logits_all[i] > logits_all[i, target_tok]).sum().item() for i in range(mac.shape[0])]
    rank_best = int(min(ranks_pt))
    return (rank_mean, rank_best), target_tok


rows = []
with open(VAL, encoding="utf-8") as fp:
    for line in fp:
        if len(rows) >= N_EP:
            break
        ep = json.loads(line)
        convs = ep.get("conversations", [])
        turns = []
        for i in range(0, len(convs) - 1, 2):
            a = convs[i]; b = convs[i + 1] if i + 1 < len(convs) else {}
            turns.append({"u": a.get("content", ""), "a": b.get("content", ""),
                          "vs": b.get("value_span") or [], "tl": b.get("train_loss", "true"),
                          "wps": float(b.get("weight_per_span", 0.0) or 0.0)})
        state = model.init_state(1).to(device)
        pol_states = []
        for t in turns:
            pol_states.append(state)
            state = advance(state, t)
        for ti, t in enumerate(turns):
            done = False
            for s, e in t["vs"]:
                vstr = t["a"][s:e]
                if vstr in t["u"]:
                    continue
                fact_idx = next((fj for fj in range(ti)
                                 if vstr in turns[fj]["u"] or vstr in turns[fj]["a"]), None)
                if fact_idx is None:
                    continue
                r_pol, tgt = mem_decode_rank(pol_states[ti], t["u"], t["a"], (int(s), int(e)))
                cs = advance(model.init_state(1).to(device), turns[fact_idx])
                r_cl, _ = mem_decode_rank(cs, t["u"], t["a"], (int(s), int(e)))
                # 隔离测:纯 decode 读 fact turn 自己(query 与 stored key 同源,retrieve 必命中)+ clean M
                ft = turns[fact_idx]
                fspan = next((sp for sp in ft["vs"] if ft["a"][sp[0]:sp[1]] == vstr), None)
                r_fact = mem_decode_rank(cs, ft["u"], ft["a"], fspan)[0] if fspan else None
                if r_pol is None:
                    continue
                rows.append((ep.get("skeleton_id"), vstr[:12], r_pol[0], r_pol[1],
                             r_cl[1] if r_cl else None, r_fact[1] if r_fact else None))
                print(f"  {ep.get('skeleton_id'):4s} '{vstr[:12]:12s}' recall_best={r_pol[1]:<6d}  "
                      f"fact_self_best={str(r_fact[1]) if r_fact else '-':<6s} <== 同源retrieve,若小则存储OK问题在query")
                done = True
                break
            if done:
                break

print(f"\n===== 汇总 (n={len(rows)}) =====")
import statistics as st
def med(xs): xs=[x for x in xs if x is not None]; return st.median(xs) if xs else float('nan')
def hits(xs,k): return sum(1 for x in xs if x is not None and x<k)
recall_ranks=[r[3] for r in rows]; fact_ranks=[r[5] for r in rows]
print(f"  纯 memory decode (lm_head∘W_mac∘mem_out), best-of-16 query token:")
print(f"    recall turn(召回 query): 中位 rank={med(recall_ranks):.0f}  top100={hits(recall_ranks,100)}/{len(rows)}")
print(f"    fact turn 自己(同源 query,retrieve 必命中): 中位 rank={med(fact_ranks):.0f}  top10={hits(fact_ranks,10)}/{len(rows)}  top100={hits(fact_ranks,100)}/{len(rows)}")
mf = med(fact_ranks)
if mf < 100:
    print(f"  -> 存储+decode OK(同源 retrieve rank {mf:.0f} 小):瓶颈是召回 query 不匹配 stored key → v18 攻 query-key 对齐")
else:
    print(f"  -> 存储/decode 坏(连同源 retrieve 都 rank {mf:.0f}):value 没被可解码地存进 M → v18 攻存储(d_value 不压缩/embedding-tied)")
