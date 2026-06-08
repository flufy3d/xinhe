"""最小架构正确性测试 —— 不依赖 ckpt + 不跑真数据(单全局架构)。

经典 ML debug:**fresh-init 模型 overfit 单个 NIH 样本**,能 overfit 才证明架构能学。
绕过大数据训练 → 快速迭代改架构 / 调参 / 加 aux loss。

Verdict 矩阵:
  NM-on=✓ / NM-zero=✗ → ✅ mem 通路活,差异由 memory 提供(真训练失败=信号/规模问题)
  NM-on=✓ / NM-zero=✓ → ⚠ backbone 直接 overfit prompt,mem 通路死
  NM-on=✗              → ❌ 架构本身学不动单样本,回设计

跑(GPU 快,几分钟):
  uv run python scripts/probe/min_arch_test.py --steps 200
  uv run python scripts/probe/min_arch_test.py --nm-aux 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel
from xinhe.data.conversation import tokenize_turn, ensure_chat_template


def build_min_episode(entity: str, n_distract: int):
    """N distractors → target write → query read(同 targeted_probe 的 NIH 缩小版)。"""
    distractors = ["Bob", "Carol", "David", "Eve", "Frank"][:n_distract]
    turns = []
    for d in distractors:
        turns.append((f"我朋友叫{d}。", f"好的,你朋友叫{d},我记下了。", []))
    aw = f"嗯,你叫{entity},我记住了。"
    sw = aw.find(entity)
    turns.append((f"我叫{entity}。", aw, [[sw, sw + len(entity)]]))
    ar = f"{entity}。"
    sr = ar.find(entity)
    turns.append(("我叫什么名字?", ar, [[sr, sr + len(entity)]]))
    return turns


def first_token_hit(model, tokenizer, state, q_user, q_asst, span, device, override):
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": q_user}, {"role": "assistant", "content": q_asst}],
        tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": q_user}], tokenize=False, add_generation_prompt=True,
    )
    asst_off = full.find(q_asst, len(prefix))
    if asst_off < 0:
        asst_off = full.find(q_asst)
    s_full = asst_off + span[0]
    pos = next((i for i, (cs, ce) in enumerate(offsets) if cs <= s_full < ce), None)
    if pos is None or pos == 0:
        return None
    px = torch.tensor([ids[:pos]], dtype=torch.long, device=device)
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(px, state, pad_token_id=tokenizer.pad_token_id, mem_alpha_override=override)
    return out["logits"][0, -1].argmax().item() == ids[pos]


@torch.no_grad()
def eval_read(model, tokenizer, turn_data, q_user, q_asst, q_span, device):
    """NM-on + NM-zero 两次 evolve 到 query 前,各测 first-token。"""
    def evolve(override):
        st = model.init_state(1).to(device)
        for ids, _, _ in turn_data[:-1]:
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                out = model(ids, st, pad_token_id=tokenizer.pad_token_id,
                            mem_alpha_override=override)
            st = out["state_next"]
        return st
    model.eval()
    on = first_token_hit(model, tokenizer, evolve(None), q_user, q_asst, q_span, device, None)
    off = first_token_hit(model, tokenizer, evolve(0.0), q_user, q_asst, q_span, device, 0.0)
    model.train()
    return on, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pcap_delta.yaml")
    ap.add_argument("--steps", type=int, default=200, help="overfit step count")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-distract", type=int, default=2)
    ap.add_argument("--entity", default="Alice")
    ap.add_argument("--nm-aux", type=float, default=0.0)
    ap.add_argument("--lambda-div", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--no-kpers", action="store_true",
                    help="关 per-layer K/V(消除 backbone 死记 query→answer 捷径,逼 mem 通路工作)")
    ap.add_argument("--no-lora", action="store_true",
                    help="关 LoRA(极端 sanity:完全 frozen backbone,几乎必死)")
    ap.add_argument("--shortcut", action="store_true",
                    help="开 Margin-Based Shortcut Suppression:每 turn 跑 NM-on/NM-zero 双 forward,"
                         "loss_on 必须比 loss_zero 至少低 margin 否则给 penalty(逼 mem 通路有用)")
    ap.add_argument("--shortcut-margin", type=float, default=0.5)
    ap.add_argument("--shortcut-lambda", type=float, default=1.0)
    ap.add_argument("--mem-dropout", type=float, default=0.0,
                    help="训练时随机 NM-zero forward 比例(implicit curriculum,与 --shortcut 互补)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, _ = XinheConfig.from_yaml(args.config)
    cfg.use_query_head = True
    cfg.compile_backbone_layers = False
    cfg.nm_aux_weight = args.nm_aux
    cfg.lambda_div = args.lambda_div
    if args.no_kpers:
        cfg.n_persistent_per_layer = 0
    if args.no_lora:
        cfg.lora_rank = 0
    cfg.shortcut_suppression = args.shortcut
    cfg.shortcut_margin = args.shortcut_margin
    cfg.shortcut_lambda = args.shortcut_lambda
    cfg.memory_dropout = args.mem_dropout

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"=== min_arch_test: hippo=delta steps={args.steps} lr={args.lr} "
          f"nm_aux={args.nm_aux} λ_div={args.lambda_div} n_distract={args.n_distract} "
          f"lora={cfg.lora_rank} kpers={cfg.n_persistent_per_layer} "
          f"shortcut={'on' if args.shortcut else 'off'}(m={args.shortcut_margin},λ={args.shortcut_lambda}) "
          f"mem_drop={args.mem_dropout} dev={dev} ===")

    torch.manual_seed(args.seed)
    model = XinheModel(cfg).to(dev)
    tok = AutoTokenizer.from_pretrained(
        str(Path(cfg.backbone_model_path).resolve()), trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)

    turns = build_min_episode(args.entity, args.n_distract)
    print(f"  episode: {len(turns)} turns, target={args.entity!r}")
    seg_len = cfg.turn_max_tokens
    turn_data = []
    for u, a, spans in turns:
        ids, lbl, wts = tokenize_turn(
            tok, u, a, seg_len,
            train_loss="true" if spans else "lm_only",
            value_spans=spans, weight_per_span=1.0 if spans else 0.0,
        )
        turn_data.append((
            ids.unsqueeze(0).to(dev),
            lbl.unsqueeze(0).to(dev),
            wts.unsqueeze(0).to(dev),
        ))
    q_user, q_asst, q_span = turns[-1]
    q_span0 = q_span[0]

    opt = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)
    on_hit, off_hit = None, None
    last_gap = float("nan")
    last_penalty = float("nan")
    for step in range(args.steps + 1):
        model.train()
        state = model.init_state(1).to(dev)
        total = None
        for ids, lbl, wts in turn_data:
            # Memory Dropout(implicit curriculum):随机让本 turn 走 NM-zero,backbone
            # 不能假设 mem 总在。与 shortcut 互斥(本 turn 已是 NM-zero,无需 baseline)
            do_drop = (cfg.memory_dropout > 0
                       and float(torch.rand(()).item()) < cfg.memory_dropout)
            this_override = 0.0 if do_drop else None

            with torch.amp.autocast(dev.type, dtype=torch.bfloat16):
                out = model(ids, state, labels=lbl, weights=wts,
                            pad_token_id=tok.pad_token_id,
                            mem_alpha_override=this_override)
            state = out["state_next"]
            loss = out["loss"]

            # Margin-Based Shortcut Suppression:NM-zero forward **带 grad**(关键!)
            # 加 hinge:penalty = max(0, loss_on - loss_zero + margin),active 时:
            #   d(loss_on)/d(params) 推主路径优化(NM-on 更准)
            #   d(-loss_zero)/d(params) **推 backbone 在 NM-zero 模式下变差**(因 mem_alpha=0
            #     时 W_mac/W_mal/mem_out 全被乘 0,梯度只能落 backbone 的 LoRA / K_pers)
            # → 直接 attack backbone shortcut:不让 backbone 学到"不靠 mem 也能答对"
            # 代价:dual backward,GPU mem ~1.5x。
            if cfg.shortcut_suppression and not do_drop:
                with torch.amp.autocast(dev.type, dtype=torch.bfloat16):
                    out_zero = model(ids, state, labels=lbl, weights=wts,
                                     pad_token_id=tok.pad_token_id, mem_alpha_override=0.0)
                # ★ 关键修复:gap 必须用纯 CE,不能用 result["loss"](含 aux_loss)。
                # NM-zero forward 里 _any_override=True → nm_aux 被跳过,loss_zero 没 aux,
                # loss_on 有 aux(nm_aux_weight × nm_ce),gap 天然偏负,
                # penalty 实际在惩罚 nm_aux 项的存在,不是惩罚 NM shortcut。
                ce_on = out["ce_loss"]
                ce_zero = out_zero["ce_loss"]            # 不 detach!grad 通到 backbone
                penalty = torch.clamp(ce_on - ce_zero + cfg.shortcut_margin, min=0.0)
                last_gap = float((ce_zero.detach() - ce_on.detach()).item())
                last_penalty = float(penalty.detach().item())
                loss = loss + cfg.shortcut_lambda * penalty

            total = loss if total is None else total + loss
        opt.zero_grad()
        total.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            on_hit, off_hit = eval_read(model, tok, turn_data, q_user, q_asst, q_span0, dev)
            mn = "—"
            if hasattr(model, "global_hippo"):
                mn = f"{model.global_hippo.last_M_specnorm.item():.2f}"
            extra = ""
            if cfg.shortcut_suppression:
                extra = f" | gap={last_gap:+.3f} pen={last_penalty:.3f}"
            print(f"  [step {step:3d}] loss={total.item():6.3f} | "
                  f"NM-on={on_hit} NM-zero={off_hit} | M_spec={mn}{extra}")

    print(f"\n=== verdict ===")
    if on_hit is True and off_hit is False:
        print("  ✅ mem 通路活:NM-on 命中、NM-zero 没命中 → 架构能学,差异由 memory 提供。")
        print("     真训练 read=0% → 信号/规模问题,加 nm_aux/λ_div 或调数据混合即可。")
    elif on_hit is True and off_hit is True:
        print("  ⚠ NM-on == NM-zero:backbone 直接 overfit prompt,没靠 memory。")
        print("     mem 通路死(W_mac/MAL/QueryHead 学到零,或 backbone 忽略 prefix)。")
        print("     → 架构 bug:mem 输出对 logits 无贡献。改 mem 通路设计。")
    else:
        print(f"  ❌ overfit 失败(NM-on={on_hit}):架构本身没法学单样本。")
        print("     → 回设计阶段,架构有根本问题。")


if __name__ == "__main__":
    main()
