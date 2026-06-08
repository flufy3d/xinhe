#!/usr/bin/env python
"""v17 诊断:为什么 eval 里 NM-on == NM-zero 精确为零?
逐项测量 read 通路:σ(mal_alpha_logit)、‖W_mal‖、‖mem_out‖、MAL delta、NM-on/zero logits 差。
forward-only no_grad,跑前 N 个 val episode 的首个 recall turn。
"""
import sys, json, torch
sys.path.insert(0, ".")
from scripts.evaluate import load_model_and_tokenizer
from scripts.validate_memory import tokenize_turn, _locate_value_token
from xinhe.model.config import XinheConfig

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/xinhe_step_1500.pt"
VAL = "data/skeleton/val.jsonl"
N_EP = 12

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
config = ckpt["config"]
config.compile_backbone_layers = False
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
seg_len = getattr(config, "turn_max_tokens", 256)
model, tokenizer = load_model_and_tokenizer(config, CKPT, device)
model.eval()

print(f"\n===== 静态权重诊断 =====")
print(f"  mac_disabled        = {getattr(config, 'mac_disabled', 'MISSING')}")
print(f"  _mal_target_idx     = {model._mal_target_idx} ({model._mal_target_layer_type})")
print(f"  _global_write_idx   = {model._global_write_idx}")
print(f"  mal_alpha_logit     = {model.mal_alpha_logit.item():+.4f}  -> sigmoid = {torch.sigmoid(model.mal_alpha_logit).item():.4f}")
print(f"  ||W_mal.weight||    = {model.W_mal.weight.norm().item():.4e}")
print(f"  ||W_mac.weight||    = {model.W_mac.weight.norm().item():.4e}")
print(f"  ||query_head proj|| = {sum(p.norm().item() for p in model.query_head.parameters()):.4e}")

# monkeypatch W_mal 记录 input(mem_out) / output norm
rec = {}
_orig_wmal = model.W_mal.forward
def _wmal_spy(x):
    rec["mem_out_norm"] = x.detach().float().norm().item()
    y = _orig_wmal(x)
    rec["wmal_out_norm"] = y.detach().float().norm().item()
    return y
model.W_mal.forward = _wmal_spy


def read_logits(state, user_msg, asst_text, char_span, override):
    """复刻 _check_first_token 的 forward,但返回 (logits, rec, target_tok)。"""
    full_ids, target_tok, target_pos = _locate_value_token(tokenizer, user_msg, asst_text, char_span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return None, None, None
    prefix_ids = full_ids[:target_pos]
    input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    rec.clear()
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(input_tensor, state, pad_token_id=tokenizer.pad_token_id,
                    mem_alpha_override=override)
    logits = out["logits"][0, -1].detach().float()
    return logits, dict(rec), target_tok


def rank_of(logits, tok):
    """target token 在 logits 里的排名(0=argmax)。"""
    return int((logits > logits[tok]).sum().item())


def advance(state, user_msg, asst_msg, value_spans, train_loss, wps, override):
    ids, labels, _w = tokenize_turn(tokenizer, user_msg, asst_msg, seg_len,
                                    train_loss=train_loss, value_spans=value_spans,
                                    weight_per_span=wps)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(ids.unsqueeze(0).to(device), state,
                    pad_token_id=tokenizer.pad_token_id, mem_alpha_override=override)
    return out["state_next"]


n_checked = 0
with open(VAL, encoding="utf-8") as fp:
    for line in fp:
        if n_checked >= N_EP:
            break
        ep = json.loads(line)
        convs = ep.get("conversations", [])
        state = model.init_state(1).to(device)
        found = False
        for i in range(0, len(convs) - 1, 2):
            user_msg = convs[i].get("content", "")
            asst = convs[i + 1] if i + 1 < len(convs) else {}
            asst_msg = asst.get("content", "")
            vspans = asst.get("value_span") or []
            train_loss = asst.get("train_loss", "true")
            wps = float(asst.get("weight_per_span", 0.0) or 0.0)
            for s, e in vspans:
                vstr = asst_msg[s:e]
                is_recall = vstr not in user_msg
                if is_recall and not found:
                    lg_on, rec_on, tgt = read_logits(state, user_msg, asst_msg, (int(s), int(e)), None)
                    lg_ze, rec_ze, _ = read_logits(state, user_msg, asst_msg, (int(s), int(e)), 0.0)
                    if lg_on is None:
                        continue
                    diff = (lg_on - lg_ze).abs().max().item()
                    r_on, r_ze = rank_of(lg_on, tgt), rank_of(lg_ze, tgt)
                    lt_on, lt_ze = lg_on[tgt].item(), lg_ze[tgt].item()
                    print(f"\n--- ep#{n_checked} recall (skeleton={ep.get('skeleton_id')}) value='{vstr[:16]}' target_tok={tgt} ---")
                    print(f"  max|logit_on - logit_zero| = {diff:.3f}   argmax翻转={lg_on.argmax().item()!=lg_ze.argmax().item()}")
                    print(f"  正确token logit:  NM-on={lt_on:+.3f}  NM-zero={lt_ze:+.3f}  Δ={lt_on-lt_ze:+.3f}  <== >0 则 mem 抬高了正确 token")
                    print(f"  正确token rank :  NM-on={r_on:<6d} NM-zero={r_ze:<6d} (0=argmax命中)  <== on<zero 则 mem 帮忙")
                    found = True
                    n_checked += 1
            # 推进 state(用 NM-on override=None 口径)
            state = advance(state, user_msg, asst_msg, vspans, train_loss, wps, None)
            if found:
                break

print("\n诊断完成。")
