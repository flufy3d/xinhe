"""Phase 1 Targeted Probe —— 把 read=0% 解构成三个独立失败模式。

诊断(forward 单步,no BPTT):
  write_probe   = retrieve(M_after_target, q=target_key_emb) 经 lm_head 是否预测出 value 首 token
                  —— 绕过 read 接口,直接验"W 是否持有 key→value"(镜像 nm_aux 路由)
  read_probe    = first_token(query 轮输出) == value 首 token —— 走完整 backbone,验读接口
  leakage_score = query 轮 free-gen 里出现 distractor 实体的比例 —— 验 distract 污染

数据在内存生成(不入 jsonl)。模型/ckpt 不改。
任务:N 个 distract(同类型不同实体的 tell 轮)→ 1 target → 1 query,严格 distract×N→target→query。

NM-zero 口径(mem_alpha_override=0.0)必须让 read≈0%(验证不是 backbone 偷记)。

Phase 1 Pass = 在当前 v9.5 上 baseline 复现 read_probe ≈ 0%(≤~5%)。
  read_probe 偏高 → 是 probe 设计太易/泄漏,修 probe,不改架构。

跑(需 GPU + v9.5 ckpt):
  uv run python scripts/probe/targeted_probe.py --checkpoint <v9.5.pt> --n-distract 3
  uv run python scripts/probe/targeted_probe.py --checkpoint <v9.5.pt> --n-distract 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 只在模块顶层 import 轻量依赖(generate_recall_probe 纯数据,无 torch);
# 模型栈(scripts.evaluate / validate_memory / xinhe.model)在用到的函数里 lazy import,
# 这样 make_nih_episode / count_leakage 的 CPU 自测无需拉起整个模型/transformers 栈。
from scripts.generate_recall_probe import ENTITY_PATTERNS

# value_type → 模板(复用 recall_probe 的实体池 + write/read 模板)
PATTERN_BY_TYPE = {p["type"]: p for p in ENTITY_PATTERNS}


# ───────────────────────── 内存 NIH 生成器 ─────────────────────────

def make_nih_episode(
    rng: random.Random,
    n_distract: int,
    value_type: str = "name_en",
    target_position: int | None = None,
) -> dict:
    """生成一条 distract×N → target → query 的 NIH episode(纯内存,不落盘)。

    distract 轮:N 个同类型不同实体的 tell 轮,与 target 竞争同一 key 几何 → flood M。
    target 轮:写入指定 entity(=gold)。target_position 控制它落在 N 个 distract 中的第几位
              (默认 = 全部 distract 之后,即紧邻 query,literal "distract×N→target→query")。
    query 轮:问目标关系,gold = target entity;复用 recall_probe 的 assert 保证 user_msg 不含 entity。
    """
    if value_type not in PATTERN_BY_TYPE:
        raise ValueError(f"未知 value_type={value_type!r},可选 {sorted(PATTERN_BY_TYPE)}")
    pat = PATTERN_BY_TYPE[value_type]
    pool = pat["pool"]
    if n_distract + 1 > len(pool):
        raise ValueError(
            f"value_type={value_type!r} 池大小 {len(pool)} 不够 n_distract+1={n_distract + 1};"
            f"换更大的池(如 name_en=40)或减小 --n-distract"
        )
    if target_position is None:
        target_position = n_distract  # 默认:target 在所有 distract 之后
    if not (0 <= target_position <= n_distract):
        raise ValueError(f"target_position 须在 [0, {n_distract}],得到 {target_position}")

    # 采样互不相同的实体:1 个 target + N 个 distractor
    entities = rng.sample(pool, n_distract + 1)
    target = entities[0]
    distractors = entities[1:]

    def _tell(entity: str) -> list[dict]:
        u = pat["user_write"].format(entity=entity)
        a = pat["asst_write"].format(entity=entity)
        s = a.find(entity)
        assert s >= 0, f"entity 未在 asst_write 找到: {entity} in {a}"
        return [
            {"role": "user", "content": u},
            {"role": "assistant", "content": a, "train_loss": "true",
             "value": [entity], "value_span": [[s, s + len(entity)]],
             "value_tier": "hard", "weight_per_span": 1.0},
        ]

    # 组装写入序列:distractor 与 target 按 target_position 交错
    write_turns: list[str] = list(distractors)
    write_turns.insert(target_position, target)

    convs: list[dict] = []
    for e in write_turns:
        convs.extend(_tell(e))

    # query 轮(gold = target)
    user_read = pat["user_read"]
    asst_read = pat["asst_read"].format(entity=target)
    assert target not in user_read, f"BUG: query user_msg 含 entity({target}) → 非真 recall"
    r_s = asst_read.find(target)
    assert r_s >= 0
    convs.extend([
        {"role": "user", "content": user_read},
        {"role": "assistant", "content": asst_read, "train_loss": "true",
         "value": [target], "value_span": [[r_s, r_s + len(target)]],
         "value_tier": "hard", "weight_per_span": 1.0},
    ])

    return {
        "skeleton_id": "PROBE_NIH",
        "meta": {
            "entity_type": value_type,
            "entity": target,
            "distractors": distractors,
            "n_distract": n_distract,
            "target_position": target_position,
            "distance_bucket": "far",
        },
        "conversations": convs,
    }


def count_leakage(text: str, distractors: list[str], target: str) -> float:
    """decoded 文本里出现的 distractor 实体占比(纯函数,可独立测)。"""
    if not distractors:
        return 0.0
    hit = sum(1 for d in distractors if d != target and d in text)
    return hit / len(distractors)


# ───────────────────────── 单步 forward 诊断 ─────────────────────────

@torch.no_grad()
def _evolve_one_turn(model, tokenizer, state, user_msg, asst_msg, value_spans,
                     weight, seg_len, device, mem_alpha_override):
    """teacher-forced 推进一个 turn,返回新 state(复用 validate_memory 演化口径)。"""
    from xinhe.data.conversation import tokenize_turn
    ids, _labels, _w = tokenize_turn(
        tokenizer, user_msg, asst_msg, seg_len,
        train_loss="true", value_spans=value_spans, weight_per_span=weight,
    )
    ids_dev = ids.unsqueeze(0).to(device)
    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(ids_dev, state, pad_token_id=tokenizer.pad_token_id,
                    mem_alpha_override=mem_alpha_override)
    return out["state_next"]


@torch.no_grad()
def _free_gen_text(model, tokenizer, state, user_msg, device, mem_alpha_override,
                   max_new_tokens=32, char_budget=48) -> str:
    """query 轮 greedy decode,返回 decoded 文本(复用 _check_free_gen 解码骨架)。"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        out = model(generated, state, pad_token_id=tokenizer.pad_token_id,
                    mem_alpha_override=mem_alpha_override)
    next_logits = out["logits"][:, -1, :]
    for _ in range(max_new_tokens):
        nxt = next_logits.argmax(dim=-1, keepdim=True)
        if eos_id is not None and nxt.item() == eos_id:
            break
        generated = torch.cat([generated, nxt], dim=1)
        decoded = tokenizer.decode(generated[0, len(input_ids):], skip_special_tokens=True)
        if len(decoded) >= char_budget:
            break
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            out = model(generated, state, pad_token_id=tokenizer.pad_token_id,
                        mem_alpha_override=mem_alpha_override)
        next_logits = out["logits"][:, -1, :]
    return tokenizer.decode(generated[0, len(input_ids):], skip_special_tokens=True)


@torch.no_grad()
def _write_probe(model, tokenizer, state_after_target, key_str, target_value_tok, device):
    """write_probe:q 从 key 的 embedding 派生(走 QueryHead.norm+proj),经 global_hippo retrieve
    → W_mac 投到 hidden → lm_head,看 argmax 是否命中 value 首 token。绕过 read 接口,
    直接验单全局 M 是否持有 key→value 关联。

    注:旧 per-layer NeuralMemoryPair 探针口径已迁到单全局架构:hits 现在只有一项(单 M)。
    """
    if not hasattr(model, "global_hippo") or not hasattr(model, "query_head"):
        return None
    key_ids = tokenizer.encode(key_str, add_special_tokens=False)
    if not key_ids:
        return None
    key_t = torch.tensor([key_ids], dtype=torch.long, device=device)
    gkey = getattr(model, "_global_write_idx", None)
    if gkey is None:
        return None
    old_layer = state_after_target.get(gkey, None)
    old_hippo = old_layer.hippo if old_layer is not None else None
    try:
        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
            key_emb = model.backbone.embed(key_t)              # (1, Lk, hidden)
            h_last = key_emb[:, -1]                            # (1, hidden) — 取 key 末位
            q = model.query_head(h_last)                       # (1, n_q, d_key)
            r_h = model._global_read(q, old_hippo)             # (1, n_q, d_value)
            mem_out = model.global_mem_rmsnorm(r_h)
            mac_proj = model.W_mac(mem_out).mean(dim=1)        # (1, hidden) 均值聚合
            logits = model.lm_head(mac_proj)
        hit = int(logits.argmax(dim=-1).item()) == target_value_tok
        return {"mean": float(hit), "max": float(hit), "n_pairs": 1}
    except Exception:
        return None


@torch.no_grad()
def run_episode(model, tokenizer, ep, device, seg_len, mem_alpha_override):
    """跑一条 episode:teacher-force 推进 distract+target,在 query 轮测 read(+write/leakage)。

    NM-on(mem_alpha_override=None)测 write/read/leakage;NM-zero(=0.0)只测 read。
    state 演化与诊断都带同一 override,口径一致(镜像 validate_memory._run_pass)。
    """
    from scripts.validate_memory import _check_first_token, _locate_value_token
    convs = ep["conversations"]
    state = model.init_state(1).to(device)
    # 推进 distract + target(最后一对是 query,留作探测)
    for i in range(0, len(convs) - 2, 2):
        asst = convs[i + 1]
        state = _evolve_one_turn(
            model, tokenizer, state, convs[i]["content"], asst["content"],
            asst.get("value_span") or [], float(asst.get("weight_per_span", 0.0) or 0.0),
            seg_len, device, mem_alpha_override,
        )

    q_user = convs[-2]["content"]
    q_asst = convs[-1]["content"]
    q_span = tuple(convs[-1]["value_span"][0])
    target = ep["meta"]["entity"]

    read = _check_first_token(model, tokenizer, state, q_user, q_asst, q_span,
                              device, mem_alpha_override)
    rec: dict = {"read": read}

    if mem_alpha_override is None:
        _ids, value_tok, _pos = _locate_value_token(tokenizer, q_user, q_asst, q_span)
        rec["write"] = (_write_probe(model, tokenizer, state, target, value_tok, device)
                        if value_tok is not None else None)
        text = _free_gen_text(model, tokenizer, state, q_user, device, None)
        rec["leakage"] = count_leakage(text, ep["meta"]["distractors"], target)
    return rec


# ───────────────────────── 聚合 + 主流程 ─────────────────────────

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_probe(model, tokenizer, episodes, device, seg_len) -> dict:
    if not episodes:
        raise ValueError("n_episodes == 0:没有 episode 可跑(实现规划 §3:n_eps=0 必 raise)")
    on = [run_episode(model, tokenizer, ep, device, seg_len, None) for ep in episodes]
    off = [run_episode(model, tokenizer, ep, device, seg_len, 0.0) for ep in episodes]

    read_on = _mean([r["read"] for r in on])
    read_off = _mean([r["read"] for r in off])
    write_mean = _mean([r["write"]["mean"] for r in on if r.get("write")])
    write_max = _mean([r["write"]["max"] for r in on if r.get("write")])
    leakage = _mean([r["leakage"] for r in on])
    return {
        "n_episodes": len(episodes),
        "write_probe_mean": write_mean,
        "write_probe_max": write_max,
        "read_probe_nm_on": read_on,
        "read_probe_nm_zero": read_off,
        "leakage_score": leakage,
    }


def _fmt(x):
    return "  N/A" if x is None else f"{x * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description="心核 Phase 1 Targeted Probe")
    ap.add_argument("--checkpoint", default=None, help="v9.5 ckpt;省略=fresh-init baseline")
    ap.add_argument("--config", default="configs/pcap_skeleton.yaml")
    ap.add_argument("--n-distract", type=int, default=3)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--value-type", default="name_en", choices=sorted(PATTERN_BY_TYPE))
    ap.add_argument("--target-position", type=int, default=None)
    ap.add_argument("--seg-len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", default=None, help="可选 JSON 落盘")
    args = ap.parse_args()

    if args.n_episodes <= 0:
        raise ValueError("n_episodes 必须 > 0")

    from xinhe.model.config import XinheConfig
    from scripts.evaluate import load_model_and_tokenizer

    config, _ = XinheConfig.from_yaml(args.config)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt.get("config"), XinheConfig):
            config = ckpt["config"]
            print(f"  使用 ckpt 内置配置: backbone={config.backbone_type}")
    config.compile_backbone_layers = False

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seg_len = args.seg_len or getattr(config, "turn_max_tokens", 256)

    print("=== Phase 1 Targeted Probe ===")
    print(f"  ckpt={args.checkpoint or '(fresh-init)'}  config={args.config}")
    print(f"  n_distract={args.n_distract}  n_episodes={args.n_episodes}  "
          f"value_type={args.value_type}  seg_len={seg_len}  device={device}")

    rng = random.Random(args.seed)
    episodes = [make_nih_episode(rng, args.n_distract, args.value_type, args.target_position)
                for _ in range(args.n_episodes)]

    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)
    res = run_probe(model, tokenizer, episodes, device, seg_len)

    print("\n--- results ---")
    print(f"  write_probe   mean={_fmt(res['write_probe_mean'])}  max={_fmt(res['write_probe_max'])}")
    print(f"  read_probe    NM-on={_fmt(res['read_probe_nm_on'])}  "
          f"NM-zero={_fmt(res['read_probe_nm_zero'])}")
    print(f"  leakage_score {_fmt(res['leakage_score'])}")
    print("\n  [Phase 1 Pass] baseline read_probe(NM-on) ≈ 0%(≤~5%)且 NM-zero ≈ 0% → probe 可信,进 Phase 2")
    print("  read_probe 偏高 → probe 太易/泄漏,修 probe 不改架构(实现规划 §74)")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"args": vars(args), "result": res}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  → {args.output}")


if __name__ == "__main__":
    main()
