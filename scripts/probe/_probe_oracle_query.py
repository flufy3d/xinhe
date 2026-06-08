#!/usr/bin/env python
"""v18 判定探针 —— oracle query。
clean-write 已证伪稀释假设(clean M 仍 rank~366)。现分清:
  (a) query-key 不对齐:召回 prompt 产生的 q 选不中 fact 的 key
  (b) storage/decode 坏:retrieve+decode 本身就出不来正确 value
做法:对每个 recall turn,clean M 只写 fact turn。然后 read 该 recall turn 时,
把 model.query_head 的输出强制替换为"在 fact prompt 上算出的 q_fact"(oracle query,
本应完美匹配 fact key)。比较三种 query 下正确 value token 的 rank:
  - q_recall(现状,召回 prompt 的 query)
  - q_fact  (oracle:fact prompt 的 query)
  - q_self  (sanity:直接读 fact turn 自己,q 与 key 同源)
若 q_fact rank << q_recall rank(冲到 0)→ bug=query-key 对齐;v18 攻对齐。
若 q_fact rank 仍大 → storage/decode 坏;v18 攻 retrieve/decode。
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

# ---- query_head 录制/回放 monkeypatch ----
_orig_qh = model.query_head.forward
_qh_mode = {"mode": "normal", "captured": None}
def _qh_patched(h_last):
    if _qh_mode["mode"] == "replay" and _qh_mode["captured"] is not None:
        return _qh_mode["captured"]
    out = _orig_qh(h_last)
    if _qh_mode["mode"] == "record":
        _qh_mode["captured"] = out.detach()
    return out
model.query_head.forward = _qh_patched


def advance(state, t):
    ids, _l, _w = tokenize_turn(tokenizer, t["u"], t["a"], seg_len,
                                train_loss=t["tl"], value_spans=t["vs"], weight_per_span=t["wps"])
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(ids.unsqueeze(0).to(device), state,
                    pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    return out["state_next"]


def read_rank(state, user_msg, asst_text, char_span, override=None):
    full_ids, target_tok, target_pos = _locate_value_token(tokenizer, user_msg, asst_text, char_span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return None, None
    inp = torch.tensor([full_ids[:target_pos]], dtype=torch.long, device=device)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(inp, state, pad_token_id=tokenizer.pad_token_id, mem_alpha_override=override)
    logits = out["logits"][0, -1].detach().float()
    return int((logits > logits[target_tok]).sum().item()), target_tok


def capture_q_on_prompt(state, user_msg, asst_text, char_span):
    """在 fact prompt 上跑一次 forward,record 模式抓 q_fact。"""
    full_ids, _t, target_pos = _locate_value_token(tokenizer, user_msg, asst_text, char_span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return False
    inp = torch.tensor([full_ids[:target_pos]], dtype=torch.long, device=device)
    _qh_mode["mode"] = "record"; _qh_mode["captured"] = None
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        model(inp, state, pad_token_id=tokenizer.pad_token_id, mem_alpha_override=None)
    _qh_mode["mode"] = "normal"
    return _qh_mode["captured"] is not None


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
                ft = turns[fact_idx]
                # clean M:fresh 只写 fact turn
                cs = advance(model.init_state(1).to(device), ft)
                # fact turn 的 value span(用于 q_fact / q_self 定位)
                fspan = next((sp for sp in ft["vs"] if ft["a"][sp[0]:sp[1]] == vstr), None)

                _qh_mode["mode"] = "normal"
                r_recall, tgt = read_rank(cs, t["u"], t["a"], (int(s), int(e)))

                r_fact = None
                if fspan is not None and capture_q_on_prompt(model.init_state(1).to(device), ft["u"], ft["a"], fspan):
                    _qh_mode["mode"] = "replay"
                    r_fact, _ = read_rank(cs, t["u"], t["a"], (int(s), int(e)))
                    _qh_mode["mode"] = "normal"

                # q_self sanity:直接读 fact turn 自己(q 与 key 同 prompt)
                # 同时测 mem 关闭(override=0)→ 区分"真用 memory" vs "backbone 从 prefix 泄漏 value"
                r_self = r_self_z = None
                if fspan is not None:
                    r_self, _ = read_rank(cs, ft["u"], ft["a"], fspan, override=None)
                    r_self_z, _ = read_rank(cs, ft["u"], ft["a"], fspan, override=0.0)

                if r_recall is None:
                    continue
                rows.append((ep.get("skeleton_id"), vstr[:12], r_recall, r_fact, r_self, r_self_z))
                print(f"  {ep.get('skeleton_id'):4s} '{vstr[:12]:12s}' q_recall={r_recall:<6d} "
                      f"q_fact={str(r_fact):<7s} q_self(mem)={str(r_self):<6s} q_self(mem off)={str(r_self_z):<6s}")
                done = True
                break
            if done:
                break

print(f"\n===== 汇总 (n={len(rows)}) =====")
import statistics as st
def med(xs): xs=[x for x in xs if x is not None]; return st.median(xs) if xs else float('nan')
def hits(xs): return sum(1 for x in xs if x==0)
print(f"  q_recall    : 命中={hits([r[2] for r in rows])}/{len(rows)}  中位rank={med([r[2] for r in rows]):.0f}")
print(f"  q_fact      : 命中={hits([r[3] for r in rows])}/{len(rows)}  中位rank={med([r[3] for r in rows]):.0f}  <== oracle query")
print(f"  q_self(mem) : 命中={hits([r[4] for r in rows])}/{len(rows)}  中位rank={med([r[4] for r in rows]):.0f}  <== sanity")
print(f"  q_self(关mem): 命中={hits([r[5] for r in rows])}/{len(rows)}  中位rank={med([r[5] for r in rows]):.0f}  <== 关键:若仍0则是backbone泄漏,非memory")
self_mem = med([r[4] for r in rows]); self_z = med([r[5] for r in rows])
if self_z > 20 and self_mem < 5:
    print(f"  -> q_self 靠 memory(关 mem 后 rank {self_z:.0f}>>0):存储/decode 真的工作,瓶颈是 recall 上下文里 mem 注入太弱")
else:
    print(f"  -> q_self 是 backbone 泄漏(关 mem 仍 rank {self_z:.0f}):memory decode 未验证,需另查")
