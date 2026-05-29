"""最小架构 smoke —— 上真数据训练前一键确认单全局架构没问题。

验 QueryHead + HippoDelta 单全局的:整体 forward → state 跨 turn 演化 →
2-turn TBPTT backward(梯度到 QueryHead / W_mac / delta W_k)→ NM-zero forward → M 谱范数。
本地 backbone:CPU 走 fp32(慢但够验逻辑),GPU 走 bf16 autocast(接近真实训练)。

  uv run python scripts/probe/smoke_arch.py                 # CPU
  uv run python scripts/probe/smoke_arch.py --device cuda   # GPU bf16
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


def run_one(config_path: str, device: str):
    cfg, _ = XinheConfig.from_yaml(config_path)
    cfg.use_query_head = True
    cfg.d_key, cfg.d_value = 256, 128
    cfg.compile_backbone_layers = False

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(0)
    model = XinheModel(cfg)
    if dev.type == "cpu":
        model = model.float()          # CPU 统一 fp32(无 autocast)
    model.to(dev).eval()
    gk = model._global_write_idx

    B, T = 1, 8
    ids1 = torch.randint(5, 2000, (B, T), device=dev)
    ids2 = torch.randint(5, 2000, (B, T), device=dev)
    ac = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda"
          else contextlib.nullcontext())

    # 2-turn TBPTT(都 grad on),loss 在 turn2(read 上一 turn write 的 M)→ 反传到 write 投影
    with ac:
        o1 = model(ids1, model.init_state(B).to(dev), pad_token_id=0)
        o2 = model(ids2, o1["state_next"], labels=ids2.clone(), pad_token_id=0)
    o2["loss"].backward()

    hs = o1["state_next"].layers[gk].hippo
    g_qh = sum((p.grad.norm().item() for p in model.query_head.parameters() if p.grad is not None), 0.0)
    g_mac = model.W_mac.weight.grad.norm().item() if model.W_mac.weight.grad is not None else 0.0
    print(f"[delta] loss={o2['loss'].item():.3f} | hippo={type(hs).__name__} "
          f"| logits={tuple(o1['logits'].shape)} | QueryHead_grad={g_qh:.2f} | W_mac_grad={g_mac:.2f}")
    assert g_qh > 0 and g_mac > 0, "read / MAC-R 梯度通路断"
    assert gk in o1["state_next"].layers, "write 层未写入 state"
    g_wk = model.global_hippo.W_k.weight.grad.norm().item()
    sn = model.global_hippo.last_M_specnorm.item()
    print(f"        delta W_k_grad={g_wk:.2f} | M_specnorm={sn:.2f}")
    assert g_wk > 0, "delta W_k 无梯度(write→read TBPTT 没接上)"
    assert sn < model.global_hippo.spectral_norm_cap * 5, "M 谱范数异常(可能爆炸)"

    # NM-zero:mem 通路置零,forward 不崩(read≈0 的 sanity 由 probe 在真权重上验)
    with ac, torch.no_grad():
        oz = model(ids1, model.init_state(B).to(dev), pad_token_id=0, mem_alpha_override=0.0)
    assert oz["logits"].shape == o1["logits"].shape
    print(f"        NM-zero forward ok | OK")


def main():
    ap = argparse.ArgumentParser(description="心核单全局架构 smoke")
    ap.add_argument("--config", default="configs/pcap_skeleton.yaml")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    run_one(args.config, args.device)
    print("\n=== 架构 smoke 全过 → 可上真数据训练 ===")


if __name__ == "__main__":
    main()
