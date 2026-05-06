"""离线人格注入协议(零号纪元)训练 CLI。

用法:
    python scripts/train_persona.py --config configs/persona_inject.yaml \
        --novel-stem <novel_stem>

novel_stem 与 generate_persona_data.py 输出 <out_dir>/<novel_stem>/ 一致;省略时
自动用 persona_inject.out_dir 下唯一子目录(若多个则报错要求显式指定)。

不走 trainer.py / curriculum,完全独立子系统:
  - 不加载 backbone(协议明确"训练时免前向")
  - 仅训 PersonaExpertStack,落 ckpt,后续 PR 再加载到 XinheModel forward 路径
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.model.persona_expert import PersonaExpertStack


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


class PersonaHiddenDataset(Dataset):
    """读 hidden_states.pt + pairs.json,产 (h_n, h_n+1) 对。"""

    def __init__(self, persona_dir: Path):
        self.dir = persona_dir
        self.manifest = json.loads((persona_dir / "manifest.json").read_text(encoding="utf-8"))
        self.pairs: list[tuple[int, int]] = [
            tuple(p) for p in json.loads((persona_dir / "pairs.json").read_text(encoding="utf-8"))
        ]
        self.hidden = torch.load(persona_dir / "hidden_states.pt", map_location="cpu", weights_only=True)
        # hidden: (N_blocks, n_layers_kept, hidden_dim)
        if self.hidden.dim() != 3:
            raise ValueError(f"hidden_states.pt 形状错: {tuple(self.hidden.shape)}")
        n_blocks, n_layers, hidden_dim = self.hidden.shape
        if n_blocks != self.manifest["n_blocks"]:
            raise ValueError(
                f"manifest.n_blocks={self.manifest['n_blocks']} 与张量 n_blocks={n_blocks} 不一致"
            )
        if not self.pairs:
            raise ValueError("pairs.json 为空,无可训练对")
        print(
            f"  [dataset] {persona_dir.name}: {n_blocks} 块 / {n_layers} 层 / "
            f"hidden={hidden_dim} / {len(self.pairs)} 对"
        )

    @property
    def n_layers(self) -> int:
        return self.hidden.size(1)

    @property
    def hidden_dim(self) -> int:
        return self.hidden.size(2)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.pairs[i]
        return self.hidden[a], self.hidden[b]    # 都 (n_layers, hidden_dim)


def _resolve_persona_dir(out_dir: Path, novel_stem: str | None) -> Path:
    """novel_stem 缺省时:out_dir 下唯一子目录;多于一个则报错。"""
    if novel_stem:
        d = out_dir / novel_stem
        if not d.exists():
            raise FileNotFoundError(f"persona 目录不存在: {d}(先跑 generate_persona_data.py)")
        return d
    if not out_dir.exists():
        raise FileNotFoundError(
            f"out_dir 不存在: {out_dir}(先跑 generate_persona_data.py 生成数据)"
        )
    subs = [p for p in out_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not subs:
        raise FileNotFoundError(f"{out_dir} 下无 persona 子目录")
    if len(subs) > 1:
        names = [p.name for p in subs]
        raise ValueError(
            f"{out_dir} 下有多个 persona 子目录 ({names}),用 --novel-stem 显式指定"
        )
    return subs[0]


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #


def cosine_delta_loss(
    pred: torch.Tensor,
    delta: torch.Tensor,
    layer_weights: torch.Tensor,
    *,
    l2_reg: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """协议公式: L = 1 - cos(Expert(x), ΔH),per-layer 加权;加 L2 防 ||pred|| 爆掉。

    bf16 下 normalize 分母可能下溢,显式上浮 fp32 算 cos。
    """
    pred_f = pred.float()
    delta_f = delta.float()
    pred_n = F.normalize(pred_f, dim=-1, eps=eps)
    delta_n = F.normalize(delta_f, dim=-1, eps=eps)
    cos = (pred_n * delta_n).sum(-1)                       # (B, N)
    cos_per_layer = cos.mean(0)                            # (N,)
    cos_loss = ((1.0 - cos) * layer_weights).mean()
    l2 = pred_f.pow(2).sum(-1).mean() if l2_reg > 0 else pred_f.new_zeros(())
    loss = cos_loss + l2_reg * l2
    return loss, {
        "loss": loss.detach().item(),
        "cos_loss": cos_loss.detach().item(),
        "l2": l2.detach().item(),
        "cos_mean": cos.mean().detach().item(),
        "cos_per_layer": cos_per_layer.detach().cpu().tolist(),
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _load_yaml_with_base(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in raw:
        base_path = path.parent / raw.pop("base")
        base = _load_yaml_with_base(base_path)
        for k, v in raw.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
        raw = base
    return raw


def main():
    p = argparse.ArgumentParser(description="离线人格注入协议训练")
    p.add_argument("--config", required=True, help="configs/persona_inject.yaml")
    p.add_argument("--novel-stem", default=None,
                   help="data_dir/<stem>/;省略时自动选唯一子目录")
    p.add_argument("--resume", default=None, help="resume ckpt 路径")
    args = p.parse_args()

    cfg = _load_yaml_with_base(Path(args.config))
    pi  = cfg.get("persona_inject", {})
    pe  = cfg.get("persona_expert", {})
    pt  = cfg.get("persona_training", {})
    if not pt:
        raise ValueError(f"{args.config}: persona_training 段缺失")

    data_dir = Path(pt.get("data_dir", pi.get("out_dir", "data/persona")))
    persona_dir = _resolve_persona_dir(data_dir, args.novel_stem)

    out_root = Path(pt.get("out_dir", "checkpoints/persona_expert")) / persona_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    ds = PersonaHiddenDataset(persona_dir)

    batch_size = int(pt.get("batch_size", 64))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=int(pt.get("num_workers", 0)),
        drop_last=True,
    )

    # ---- 模型形参从数据/manifest 自动推 ----
    hidden_dim = ds.hidden_dim
    n_layers = ds.n_layers

    # intermediate_dim: 0 / 缺省 → 用 backbone 自身(写在 manifest 里)
    inter_cfg = int(pe.get("intermediate_dim", 0))
    if inter_cfg <= 0:
        inter_dim = int(ds.manifest.get("intermediate_size_backbone", 0))
        if inter_dim <= 0:
            raise ValueError(
                "manifest 缺 intermediate_size_backbone;请显式设置 persona_expert.intermediate_dim"
            )
    else:
        inter_dim = inter_cfg

    init_scale = float(pe.get("init_scale", 0.02))
    print(
        f"  [model] PersonaExpertStack n_layers={n_layers} hidden={hidden_dim} "
        f"intermediate={inter_dim} init_scale={init_scale}"
    )

    expert = PersonaExpertStack(n_layers, hidden_dim, inter_dim, init_scale=init_scale)

    # ---- device / dtype ----
    device = pt.get("device", "cuda")
    dtype_str = pt.get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    expert.to(device=device, dtype=dtype)

    # ---- 优化器 ----
    opt = torch.optim.AdamW(
        expert.parameters(),
        lr=float(pt.get("learning_rate", 5e-4)),
        weight_decay=float(pt.get("weight_decay", 0.01)),
        betas=(0.9, 0.95),
    )

    max_steps = int(pt.get("max_steps", 5000))
    warmup_steps = int(pt.get("warmup_steps", 100))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay 到 10%
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    # ---- 层级权重(linspace,深层更重) ----
    lw_min = float(pt.get("layer_weight_min", 0.5))
    lw_max = float(pt.get("layer_weight_max", 1.5))
    layer_weights = torch.linspace(lw_min, lw_max, n_layers, device=device, dtype=dtype)

    dropout_p = float(pt.get("dropout_p", 0.4))
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p 越界: {dropout_p}")
    l2_reg = float(pt.get("l2_reg", 1e-3))
    grad_clip = float(pt.get("grad_clip", 1.0))
    log_every = int(pt.get("log_every", 50))
    save_every = int(pt.get("save_every", 500))

    # ---- resume ----
    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device, weights_only=True)
        expert.load_state_dict(sd["model"])
        opt.load_state_dict(sd["optimizer"])
        start_step = int(sd.get("step", 0))
        print(f"  [resume] from step {start_step} ({args.resume})")

    log_path = out_root / "training_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)
    print(f"  [log] {log_path}")
    print(
        f"  [train] max_steps={max_steps} batch_size={batch_size} lr={pt.get('learning_rate', 5e-4)} "
        f"dropout_p={dropout_p} l2_reg={l2_reg}"
    )

    # ---- 训练循环 ----
    step = start_step
    t0 = time.time()
    expert.train()

    def run_one_batch(h_n: torch.Tensor, h_next: torch.Tensor) -> dict:
        h_n = h_n.to(device=device, dtype=dtype, non_blocking=True)
        h_next = h_next.to(device=device, dtype=dtype, non_blocking=True)
        delta = h_next - h_n
        x_in = F.dropout(h_n, p=dropout_p, training=True)
        pred = expert(x_in)
        loss, info = cosine_delta_loss(
            pred, delta, layer_weights=layer_weights, l2_reg=l2_reg,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(expert.parameters(), grad_clip).item()
        # lr schedule
        cur_lr = lr_at(step) * float(pt.get("learning_rate", 5e-4))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.step()
        info["lr"] = cur_lr
        info["grad_norm"] = gnorm
        return info

    while step < max_steps:
        for h_n, h_next in loader:
            if step >= max_steps:
                break
            info = run_one_batch(h_n, h_next)
            step += 1
            if step % log_every == 0 or step == 1:
                elapsed = time.time() - t0
                cos_str = " ".join(f"{c:+.3f}" for c in info["cos_per_layer"])
                print(
                    f"  step {step:6d} | loss {info['loss']:.4f} "
                    f"cos_loss {info['cos_loss']:.4f} cos_mean {info['cos_mean']:+.4f} "
                    f"l2 {info['l2']:.3f} | grad {info['grad_norm']:.2f} "
                    f"lr {info['lr']:.2e} | t {elapsed:.0f}s | per-layer cos: {cos_str}",
                    flush=True,
                )
                log_f.write(json.dumps({"step": step, **info}) + "\n")
            if step % save_every == 0:
                _save_ckpt(out_root / f"step_{step}.pt", expert, opt, step, ds.manifest, pe, pt)

    final = out_root / "persona_expert.pt"
    _save_ckpt(final, expert, opt, step, ds.manifest, pe, pt)
    log_f.close()
    print(f"\n  [done] {step} 步 / {time.time() - t0:.0f}s | ckpt → {final}")


def _save_ckpt(
    path: Path,
    expert: PersonaExpertStack,
    opt: torch.optim.Optimizer,
    step: int,
    manifest: dict,
    pe_cfg: dict,
    pt_cfg: dict,
) -> None:
    payload = {
        "model": expert.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "n_layers": expert.n_layers,
        "hidden_dim": expert.hidden_dim,
        "intermediate_dim": expert.intermediate_dim,
        "manifest": {
            "novel_stem": manifest.get("novel_stem"),
            "novel_sha256": manifest.get("novel_sha256"),
            "layer_indices": manifest.get("layer_indices"),
            "block_size": manifest.get("block_size"),
        },
        "config": {"persona_expert": pe_cfg, "persona_training": pt_cfg},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
