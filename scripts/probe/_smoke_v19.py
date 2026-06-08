#!/usr/bin/env python
"""v19 per_layer_delta smoke:build + forward + write 生效 + NM-zero ablation + backward + ckpt 往返。"""
import sys, torch
sys.path.insert(0, ".")
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel

cfg, _ = XinheConfig.from_yaml("configs/pcap_skeleton_5080_v19.yaml")
print(f"read_mode={cfg.read_mode} head_dim={cfg.head_dim} lora_rank={cfg.lora_rank}")
dev = torch.device("cuda")
m = XinheModel(cfg); m.setup_device(dev)
assert m.read_mode == "per_layer_delta" and hasattr(m, "hippocampus"), "per_layer_delta 未生效"

n_train = m.get_trainable_param_count()
hippo_params = sum(p.numel() for p in m.hippocampus.parameters())
print(f"trainable={n_train:,}  hippocampus={hippo_params:,}")
assert hippo_params > 0 and n_train >= hippo_params

st = m.init_state(1).to(dev)
ids1 = torch.randint(5, 1000, (1, 40), device=dev)
labels1 = ids1.clone()
weights1 = torch.ones_like(ids1, dtype=torch.float32)

# turn1:write 生效?
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out1 = m(ids1, st, labels=labels1, pad_token_id=0, weights=weights1, compute_logits=True)
W1 = out1["state_next"].get(m._pld_key).hippo.M
print(f"turn1 loss={out1['loss'].item():.3f}  ce={out1['ce_loss'].item():.3f}  ||W_next||={W1.float().norm().item():.3e}")
assert W1.float().norm().item() > 0, "write 没生效(W 仍为 0)"
assert torch.isfinite(out1["loss"]), "loss NaN/Inf"

# turn2:NM-on vs NM-zero read 是否不同(W 已populated)
st1 = out1["state_next"]
ids2 = torch.randint(5, 1000, (1, 30), device=dev)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    on = m(ids2, st1, pad_token_id=0, compute_logits=True, mem_alpha_override=None)["logits"][0, -1].float()
    ze = m(ids2, st1, pad_token_id=0, compute_logits=True, mem_alpha_override=0.0)["logits"][0, -1].float()
diff = (on - ze).abs().max().item()
print(f"turn2 NM-on vs NM-zero max|Δlogit|={diff:.4f}  (>0 则 read 注入生效)")
assert diff > 1e-3, "read 注入对 logits 无影响(NM-on==NM-zero)"

# backward:2-turn TBPTT(turn1 write → turn2 read → loss),梯度才能流到 read+write 投影
# (单 turn 时 read 见 W=0 恒为 0、write 只影响未来 → 都拿不到梯度,需跨 turn)
m.train()
st = m.init_state(1).to(dev)
labels2 = ids2.clone(); weights2 = torch.ones_like(ids2, dtype=torch.float32)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    o1 = m(ids1, st, labels=labels1, pad_token_id=0, weights=weights1, compute_logits=False)
    o2 = m(ids2, o1["state_next"], labels=labels2, pad_token_id=0, weights=weights2, compute_logits=False)
    loss = o1["loss"] + o2["loss"]
loss.backward()
def gn(p): return None if p.grad is None else p.grad.norm().item()
g_read = gn(m.hippocampus.q_projs[0].weight)
g_write = gn(m.hippocampus.k_proj.weight)
g_scale = gn(m.hippocampus.read_scale)
print(f"grad q_projs[0]={g_read}  k_proj={g_write}  read_scale={g_scale}")
assert g_write is not None and g_write > 0, "写侧无梯度"
assert g_read is not None and g_read > 0, "读侧无梯度(TBPTT 没接通?)"

# ckpt 往返
ad = m.addon_state_dict()
print(f"addon_state read_mode={ad['read_mode']} keys={list(ad.keys())}")
m.load_addon_state_dict(ad)
print("SMOKE PASS")
