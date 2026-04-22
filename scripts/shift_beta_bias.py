"""
把一个 checkpoint 的 plugin state 里 `beta_proj.bias` 整体平移一个 delta 值。

用途（可选）: 如果 T2 probe 发现 chat-turn β 偏高（污染 W），
用这个脚本把 bias 平移 -2 到 -4 降低写入默认强度（sigmoid(-4)≈0.018）。
训练过程中 beta_proj 会重新学到每个 head 应有的级别，但初始起点更保守。

用法:
    python scripts/shift_beta_bias.py \
        --in checkpoints/curriculum/11_all_mix.pt \
        --out checkpoints/curriculum/11_all_mix_beta_shifted.pt \
        --delta -2.0

相当于 new_bias = old_bias + delta（delta 为负 → 写强度下调）。
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--delta", type=float, default=-2.0,
                   help="加到 beta_proj.bias 上的值（默认 -2.0 = 写强度大幅下调）")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    if not in_path.exists():
        print(f"错误: 输入不存在 {in_path}")
        sys.exit(1)

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    if "plugin_state" not in ckpt:
        print("错误: checkpoint 缺少 plugin_state")
        sys.exit(1)

    bias_key = "beta_proj.bias"
    if bias_key not in ckpt["plugin_state"]:
        # 也许有别的命名
        cand = [k for k in ckpt["plugin_state"] if "beta_proj" in k and "bias" in k]
        if not cand:
            print(f"错误: plugin_state 里找不到 beta_proj.bias，实际 keys: {list(ckpt['plugin_state'].keys())}")
            sys.exit(1)
        bias_key = cand[0]

    old = ckpt["plugin_state"][bias_key].clone()
    new = old + args.delta
    ckpt["plugin_state"][bias_key] = new

    print(f"=== beta_bias shift ===")
    print(f"  key: {bias_key}")
    print(f"  old: mean={old.mean().item():+.4f} min={old.min().item():+.4f} max={old.max().item():+.4f}")
    print(f"  Δ  : {args.delta:+.4f}")
    print(f"  new: mean={new.mean().item():+.4f} min={new.min().item():+.4f} max={new.max().item():+.4f}")
    print(f"  sigmoid(old.mean) = {torch.sigmoid(old.mean()).item():.4f}")
    print(f"  sigmoid(new.mean) = {torch.sigmoid(new.mean()).item():.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
