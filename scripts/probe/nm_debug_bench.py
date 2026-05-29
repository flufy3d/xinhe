"""心核 NM write/read 通路本地 5080 GPU 白盒 debug bench(forward-only, no_grad)。

9 个子命令,各自独立诊断单全局 QueryHead + HippoDelta 架构的一块:
  1. dtype-walk      —— bf16 autocast 覆盖审计(6 个关键点 input dtype)
  2. mal-layer-trace —— MAL hook 真触发 + α=0/1 的 L2 信号比
  3. ablate-mac-mal  —— MAC/MAL 独立 ablate 4 配置 first-token 差异性
  4. state-trace     —— state 跨 turn 演化无 silent reset(seq_index 单增 / M ptr 变)
  5. write-fidelity  —— 绕过 backbone 直接测 write→retrieve 的 ideal cosine 衰减曲线
  6. query-diversity —— QueryHead 跨 episode + 单 episode 多样性(防 mode collapse)
  7. per-head-spec   —— 单步写入后每 head 谱范数,定位 dead head
  8. spec-evolution  —— 16-turn 演化的 spec_history ASCII 直方图
  9. backend-parity  —— FLA vs torch GPU bf16 forward 对齐(WSL only)

跑(本地 Windows native 5080):
  uv run python scripts/probe/nm_debug_bench.py all --device cuda
  uv run python scripts/probe/nm_debug_bench.py write-fidelity --device cuda --n-distract-sweep 0,1,3,8
  uv run python scripts/probe/nm_debug_bench.py mal-layer-trace --device cuda

工程纪律:全 forward + no_grad;autocast 包住;n_episodes=0 显式 raise;fail 时 dump 详细。
"""
from __future__ import annotations

import argparse
import contextlib
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


# ───────────────────────── 颜色 / 打印 helpers ─────────────────────────

GREEN, RED, YELLOW, CYAN, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m"


def color_pass(msg: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {msg}")


def color_fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def color_skip(msg: str) -> None:
    print(f"{YELLOW}[SKIP]{RESET} {msg}")


def color_info(msg: str) -> None:
    print(f"  {msg}")


# ───────────────────────── 模型 / 数据 helpers ─────────────────────────


def make_model_fresh(config_path: str, device: str, seed: int = 0):
    """fresh-init XinheModel,设备 + dtype 与 smoke_arch 一致。"""
    cfg, _ = XinheConfig.from_yaml(config_path)
    cfg.use_query_head = True
    cfg.compile_backbone_layers = False
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    model = XinheModel(cfg)
    if dev.type == "cpu":
        model = model.float()
    model.to(dev).eval()
    return model, cfg, dev


def maybe_load_ckpt(model: XinheModel, ckpt_path: str | None, device) -> bool:
    """可选 ckpt 加载(strict load 单全局 qhead_state;无则保持 fresh-init)。返回是否加载。"""
    if not ckpt_path:
        return False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "qhead_state" not in ckpt:
        raise RuntimeError(f"ckpt {ckpt_path} 缺 'qhead_state',只兼容单全局架构 ckpt")
    if "backbone_addons_state" in ckpt:
        addons = {k.replace("_orig_mod.", ""): v.to(device)
                  for k, v in ckpt["backbone_addons_state"].items()}
        model.backbone.load_state_dict(addons, strict=False)
    qh = ckpt["qhead_state"]
    model.query_head.load_state_dict(qh["query_head"])
    model.W_mac.load_state_dict(qh["W_mac"])
    model.W_mal.load_state_dict(qh["W_mal"])
    model.global_mem_rmsnorm.load_state_dict(qh["global_mem_rmsnorm"])
    with torch.no_grad():
        model.mal_alpha_logit.copy_(qh["mal_alpha_logit"].to(model.mal_alpha_logit.device))
    model.global_hippo.load_state_dict(qh["global_hippo"])
    print(f"  [ckpt] 已加载: {ckpt_path}")
    return True


def autocast_ctx(dev):
    """GPU bf16 autocast,CPU 直通(fresh-init CPU 走 fp32)。"""
    return (torch.amp.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda"
            else contextlib.nullcontext())


def make_random_ids(B: int, T: int, dev, vocab_size: int = 50000) -> torch.Tensor:
    return torch.randint(5, vocab_size, (B, T), device=dev)


def lazy_tokenizer(cfg: XinheConfig):
    """lazy import tokenizer + chat template。"""
    from transformers import AutoTokenizer
    from xinhe.data.conversation import ensure_chat_template
    tok = AutoTokenizer.from_pretrained(
        str(Path(cfg.backbone_model_path).resolve()), trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ensure_chat_template(tok)
    return tok


def build_nih_turns(entity: str, n_distract: int, pat: dict) -> list[tuple[str, str, list]]:
    """构造一条 NIH episode:N 个 distract → target write → query read。
    每个 turn = (user_msg, asst_msg, value_spans)。"""
    pool = [e for e in pat["pool"] if e != entity]
    distractors = random.sample(pool, min(n_distract, len(pool)))
    turns: list[tuple[str, str, list]] = []
    # N 个 distractor tell 轮(无 value span,纯背景)
    for d in distractors:
        u = pat["user_write"].format(entity=d)
        a = pat["asst_write"].format(entity=d)
        s = a.find(d)
        turns.append((u, a, [[s, s + len(d)]] if s >= 0 else []))
    # target write 轮
    uw = pat["user_write"].format(entity=entity)
    aw = pat["asst_write"].format(entity=entity)
    sw = aw.find(entity)
    turns.append((uw, aw, [[sw, sw + len(entity)]]))
    # query read 轮(留作 query;evolve 时不推到这条)
    ur = pat["user_read"]
    ar = pat["asst_read"].format(entity=entity)
    sr = ar.find(entity)
    turns.append((ur, ar, [[sr, sr + len(entity)]]))
    return turns


def evolve_turns(model, tokenizer, turns, dev, seg_len, **overrides):
    """teacher-force 推 turns(list 中除最后一条 query 全推),返回最终 state。"""
    from xinhe.data.conversation import tokenize_turn
    state = model.init_state(1).to(dev)
    with autocast_ctx(dev), torch.no_grad():
        for u, a, spans in turns:
            train_loss = "true" if spans else "lm_only"
            ids, _, _ = tokenize_turn(
                tokenizer, u, a, seg_len,
                train_loss=train_loss,
                value_spans=spans, weight_per_span=1.0 if spans else 0.0,
            )
            out = model(ids.unsqueeze(0).to(dev), state,
                        pad_token_id=tokenizer.pad_token_id, **overrides)
            state = out["state_next"]
    return state


def first_token_hit(model, tokenizer, state, q_user, q_asst, span, dev, **overrides) -> bool | None:
    """复用 validate_memory._check_first_token 的口径:prefix forward → 末位 argmax 与 value 首 token 比。"""
    from scripts.validate_memory import _locate_value_token
    full_ids, target_tok, target_pos = _locate_value_token(tokenizer, q_user, q_asst, span)
    if full_ids is None or target_pos is None or target_pos == 0:
        return None
    px = torch.tensor([full_ids[:target_pos]], dtype=torch.long, device=dev)
    with autocast_ctx(dev), torch.no_grad():
        out = model(px, state, pad_token_id=tokenizer.pad_token_id, **overrides)
    pred = out["logits"][0, -1].argmax().item()
    return pred == target_tok


# ───────────────────────── 子命令 1: dtype-walk ─────────────────────────


def cmd_dtype_walk(args) -> dict:
    """bf16 autocast 覆盖审计:6 个关键模块挂 forward_pre_hook 记录 input dtype。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)

    if dev.type == "cpu":
        return {"pass": True, "msg": "CPU 路径无 autocast(fp32 直通),跳过 dtype 审计",
                "details": {"device": "cpu"}}

    targets: dict[str, nn.Module] = {
        "query_head.proj": model.query_head.proj,
        "global_hippo.W_k": model.global_hippo.W_k,
        "global_hippo.W_v": model.global_hippo.W_v,
        "W_mac": model.W_mac,
        "W_mal": model.W_mal,
        "lm_head": model.lm_head,
    }
    seen: dict[str, list] = {k: [] for k in targets}

    def make_hook(name: str):
        def _hook(module, inputs):
            seen[name].append(inputs[0].dtype if inputs else None)
        return _hook

    handles = [m.register_forward_pre_hook(make_hook(name)) for name, m in targets.items()]
    try:
        B, T = 1, 64
        ids = make_random_ids(B, T, dev)
        state = model.init_state(B).to(dev)
        with autocast_ctx(dev), torch.no_grad():
            model(ids, state, pad_token_id=0)
    finally:
        for h in handles:
            h.remove()

    bf = torch.bfloat16
    bad: list[tuple[str, str]] = []
    details: dict = {}
    for name, dtypes in seen.items():
        if not dtypes:
            bad.append((name, "未触发"))
            details[name] = "miss"
            continue
        d = dtypes[0]
        details[name] = str(d)
        if d != bf:
            bad.append((name, str(d)))
    color_info("dtype audit:")
    for name, d in details.items():
        tag = GREEN if d == str(bf) else RED
        print(f"    {tag}{name:<22}{RESET}  {d}")
    if bad:
        return {"pass": False,
                "msg": f"{len(bad)} 处 input 不是 bf16: {bad}",
                "details": details}
    return {"pass": True, "msg": f"6/6 关键点 input==bf16(autocast 覆盖正常)",
            "details": details}


# ───────────────────────── 子命令 2: mal-layer-trace ─────────────────────────


def cmd_mal_layer_trace(args) -> dict:
    """MAL hook 触发 + 信号比:α=1 vs α=0 的 W_mal output L2-norm 差几个数量级。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)

    n_layers = model.backbone.get_num_layers()
    mal_idx = model._mal_target_idx
    gw_idx = model._global_write_idx
    mal_type = model._mal_target_layer_type
    # 计算 mal 与最后一层之间还有多少层 attention 让信号传播
    layers_after_mal = n_layers - mal_idx - 1
    full_indices = model._hook_layer_indices
    full_after_mal = [i for i in full_indices if i > mal_idx]
    color_info(f"backbone n_layers={n_layers}  full_attn_indices={full_indices}")
    color_info(f"MAL@L{mal_idx}({mal_type})  write@L{gw_idx}  "
               f"layers_after_mal={layers_after_mal}  full_after_mal={full_after_mal}")
    color_info(f"mal_alpha_logit={model.mal_alpha_logit.item():.4f} "
               f"→ σ={torch.sigmoid(model.mal_alpha_logit).item():.4f}")

    # fresh-init 时 W_mal weight 零 init(xinhe_model.py:131)→ W_mal output 必然 0。
    # 这里临时 xavier 重新 init(无 ckpt 时),跑完恢复;ckpt 加载时用 trained weight 不动。
    used_ckpt = bool(args.ckpt)
    original_state = None
    if not used_ckpt:
        original_state = {k: v.clone() for k, v in model.W_mal.state_dict().items()}
        nn.init.xavier_uniform_(model.W_mal.weight)
        color_info("(fresh-init)临时 xavier_uniform W_mal,跑完恢复")

    # 进一步 hook delta(MAL 真实注入位置):验 α 不同时的 hidden 改变量
    # monkey-patch W_mal 包一层记录 output L2;同时挂 forward_blocks pre-hook 截 hidden delta
    original_W_mal = model.W_mal
    captured_wmal: list[float] = []

    class _Probe(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            y = self.inner(x)
            captured_wmal.append(float(y.float().norm().item()))
            return y

    model.W_mal = _Probe(original_W_mal).to(dev)
    try:
        B, T = 1, 32
        ids_seed = make_random_ids(B, T, dev)
        ids_test = make_random_ids(B, T, dev)

        # 先跑一次 forward 让 M 非空(fresh-init 第 1 turn M=None → retrieve 返回 0 → W_mal(0)=0
        # 第 2 turn 后 M 才有内容,mem_out 才非零,W_mal output 才能验证信号传播)
        captured_wmal.clear()
        state = model.init_state(B).to(dev)
        with autocast_ctx(dev), torch.no_grad():
            r_seed = model(ids_seed, state, pad_token_id=0)
        state_after = r_seed["state_next"]
        color_info(f"seed turn 写入后:M_specnorm={model.global_hippo.last_M_specnorm.item():.4f}"
                   f"  (M 非空才能测 W_mal 信号)")

        # α=1 全开:state_after 已含 M,retrieve 出非零 mem_out → W_mal output 非零
        captured_wmal.clear()
        with autocast_ctx(dev), torch.no_grad():
            out_on = model(ids_test, state_after, pad_token_id=0, mem_mal_override=1.0)
        l2_wmal_on = max(captured_wmal) if captured_wmal else 0.0
        logits_on = out_on["logits"].float().norm().item()

        # α=0 全关
        captured_wmal.clear()
        with autocast_ctx(dev), torch.no_grad():
            out_off = model(ids_test, state_after, pad_token_id=0, mem_mal_override=0.0)
        l2_wmal_off = max(captured_wmal) if captured_wmal else 0.0
        logits_off = out_off["logits"].float().norm().item()

        # 两次 logits 之差(MAL 真信号到 LM head):α=1 vs α=0 的 hidden 经多层传播后 logits 变化
        delta_logits = (out_on["logits"].float() - out_off["logits"].float()).norm().item()
    finally:
        model.W_mal = original_W_mal
        if original_state is not None:
            model.W_mal.load_state_dict(original_state)

    color_info(f"W_mal output L2: α=1→{l2_wmal_on:.4f}, α=0→{l2_wmal_off:.4f} "
               f"(W_mal 与 α 无关,两次应近似相同)")
    color_info(f"final logits norm: α=1→{logits_on:.2f}, α=0→{logits_off:.2f}")
    color_info(f"‖logits(α=1)−logits(α=0)‖ = {delta_logits:.4f}  ← MAL 信号最终能否到 logits")

    # 判 1:W_mal 被调用(=memory_hook 到达 mal_target_idx 层)
    if not captured_wmal:
        return {"pass": False,
                "msg": f"W_mal 从未 forward → memory_hook 没走到 L{mal_idx}",
                "details": {"mal_idx": mal_idx, "gw_idx": gw_idx,
                            "hook_indices": full_indices,
                            "layers_after_mal": layers_after_mal}}
    # 判 2:W_mal output 非零(W_mal weight 非零)
    if l2_wmal_on < 1e-3:
        return {"pass": False,
                "msg": f"W_mal output L2={l2_wmal_on:.2e} 近零 → W_mal weight 是 0"
                       f"({'ckpt' if used_ckpt else 'fresh-init'} 状态)",
                "details": {"l2_wmal_on": l2_wmal_on}}
    # 判 3:MAL 真信号到 logits(α=1 vs α=0 经多层传播应有可观差异)
    # fresh-init 下 layers_after_mal 个 layer 会衰减,但应 > 1e-3
    if delta_logits < 1e-3:
        return {"pass": False,
                "msg": f"‖logits(α=1)−logits(α=0)‖={delta_logits:.2e} < 1e-3:"
                       f"MAL 信号未到 logits,可能 MAL 残差被后续 layer 完全吃掉,"
                       f"或 mal_target_idx={mal_idx}({mal_type}) 选错位置 "
                       f"(layers_after_mal={layers_after_mal})",
                "details": {"l2_wmal_on": l2_wmal_on, "delta_logits": delta_logits,
                            "mal_idx": mal_idx, "mal_layer_type": mal_type,
                            "layers_after_mal": layers_after_mal}}
    return {"pass": True,
            "msg": f"MAL@L{mal_idx}({mal_type}) hook 真触发,信号传到 logits "
                   f"(W_mal L2={l2_wmal_on:.3f}, Δlogits={delta_logits:.3f})",
            "details": {"l2_wmal_on": l2_wmal_on, "delta_logits": delta_logits,
                        "mal_idx": mal_idx, "mal_layer_type": mal_type,
                        "layers_after_mal": layers_after_mal,
                        "used_ckpt": used_ckpt}}


# ───────────────────────── 子命令 3: ablate-mac-mal ─────────────────────────


def cmd_ablate_mac_mal(args) -> dict:
    """4 配置独立 ablation:(mac, mal) ∈ {(None,None), (0,None), (None,0), (0,0)}。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)
    tok = lazy_tokenizer(cfg)

    from scripts.generate_recall_probe import ENTITY_PATTERNS
    pat = next(p for p in ENTITY_PATTERNS if p["type"] == "name_en")
    random.seed(args.seed)
    entity = random.choice(pat["pool"])
    turns = build_nih_turns(entity, n_distract=3, pat=pat)
    seg_len = cfg.turn_max_tokens
    write_turns = turns[:-1]
    q_user, q_asst, q_spans = turns[-1]
    q_span = tuple(q_spans[0])

    configs: list[tuple[str, dict]] = [
        ("(None, None)",  {}),
        ("(MAC=0, None)", {"mem_mac_override": 0.0}),
        ("(None, MAL=0)", {"mem_mal_override": 0.0}),
        ("(MAC=0, MAL=0)", {"mem_mac_override": 0.0, "mem_mal_override": 0.0}),
        ("(α_override=0)", {"mem_alpha_override": 0.0}),
    ]
    hits: dict[str, bool | None] = {}
    for label, overrides in configs:
        state = evolve_turns(model, tok, write_turns, dev, seg_len, **overrides)
        hit = first_token_hit(model, tok, state, q_user, q_asst, q_span, dev, **overrides)
        hits[label] = hit
        color_info(f"{label:<22}  first_token_hit = {hit}")

    on = hits["(None, None)"]
    mac0 = hits["(MAC=0, None)"]
    mal0 = hits["(None, MAL=0)"]
    both0 = hits["(MAC=0, MAL=0)"]
    alpha0 = hits["(α_override=0)"]

    differ = len({h for h in hits.values() if h is not None}) > 1
    both_eq_alpha = (both0 == alpha0)

    if both0 is None or alpha0 is None:
        return {"pass": False,
                "msg": "first-token 定位失败(value_span tokenize 不到位)",
                "details": hits}
    if not differ:
        # fresh-init 模型 first-token 大概率 chance,4 配置全 False 可能就是 hits 全相等但都正确
        # 我们要求至少二态(任一不同)说明 MAC/MAL 通路对 logits 有差异化贡献
        # 但 fresh-init 太弱时四态可能都 chance(同为 False);只做警告不 fail
        msg = (f"4 配置 first-token 全相等 (=hits): {hits} —— fresh-init "
               f"模型很可能 backbone chance,建议用 ckpt 验")
        return {"pass": True if args.ckpt is None else False,
                "msg": msg, "details": hits}
    if not both_eq_alpha:
        return {"pass": False,
                "msg": f"(MAC=0, MAL=0) hit={both0} ≠ (α=0) hit={alpha0} —— "
                       f"二者应等价(都关 MAC+MAL),路由不一致",
                "details": hits}
    return {"pass": True,
            "msg": f"4 配置有差异且 (MAC=0,MAL=0)≡(α=0) ok ({hits})",
            "details": hits}


# ───────────────────────── 子命令 4: state-trace ─────────────────────────


def cmd_state_trace(args) -> dict:
    """state 跨 turn 演化无 silent reset:seq_index 单调递增 / M 不是 None / data_ptr 每次变。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)
    tok = lazy_tokenizer(cfg)

    from scripts.generate_recall_probe import ENTITY_PATTERNS
    pat = next(p for p in ENTITY_PATTERNS if p["type"] == "name_en")
    random.seed(args.seed)
    entity = random.choice(pat["pool"])
    turns = build_nih_turns(entity, n_distract=2, pat=pat)[:-1]  # 4 个 turn:3 distract + 1 target
    seg_len = cfg.turn_max_tokens
    gkey = model._global_write_idx

    from xinhe.data.conversation import tokenize_turn
    state = model.init_state(1).to(dev)
    snapshots: list[dict] = []
    for i, (u, a, spans) in enumerate(turns):
        ids, _, _ = tokenize_turn(
            tok, u, a, seg_len,
            train_loss="true" if spans else "lm_only",
            value_spans=spans, weight_per_span=1.0 if spans else 0.0,
        )
        with autocast_ctx(dev), torch.no_grad():
            out = model(ids.unsqueeze(0).to(dev), state, pad_token_id=tok.pad_token_id)
        state = out["state_next"]
        layer = state.layers.get(gkey)
        hippo = layer.hippo if layer is not None else None
        snap = {
            "turn": i,
            "has_layer": layer is not None,
            "hippo_id": id(hippo) if hippo is not None else None,
            "M_is_none": hippo is None or hippo.M is None,
            "M_ptr": int(hippo.M.data_ptr()) if (hippo is not None and hippo.M is not None) else None,
            "seq_index": getattr(hippo, "seq_index", None) if hippo is not None else None,
            "M_shape": tuple(hippo.M.shape) if (hippo is not None and hippo.M is not None) else None,
        }
        snapshots.append(snap)
        color_info(f"turn {i}: seq_index={snap['seq_index']}  "
                   f"M_shape={snap['M_shape']}  M_ptr={snap['M_ptr']}")

    seqs = [s["seq_index"] for s in snapshots]
    ptrs = [s["M_ptr"] for s in snapshots]
    if any(s["M_is_none"] for s in snapshots):
        return {"pass": False, "msg": f"某 turn 后 M 仍为 None: {snapshots}",
                "details": {"snapshots": snapshots}}
    monotonic = all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))
    distinct_ptrs = len(set(ptrs)) == len(ptrs)
    if not monotonic:
        return {"pass": False, "msg": f"seq_index 非单调递增: {seqs}",
                "details": {"snapshots": snapshots}}
    if not distinct_ptrs:
        return {"pass": False, "msg": f"M data_ptr 重复(疑似 in-place 覆写): {ptrs}",
                "details": {"snapshots": snapshots}}
    return {"pass": True, "msg": f"{len(turns)} turn 演化:seq_index 单增 {seqs};"
                                 f"M 每 turn 新 tensor",
            "details": {"snapshots": snapshots}}


# ───────────────────────── 子命令 5: write-fidelity ─────────────────────────


def cmd_write_fidelity(args) -> dict:
    """**核心**:write→retrieve 数学保真度测试。**纯 HippoDelta unit**(不经 backbone/QueryHead)。

    构造:N 个随机 distract keys + 1 target,逐个 write 进同一 M;然后用 target 的 *同一个* k
    retrieve,验 cosine(retrieve(M, k_target), v_target)。
    delta rule 数学:M_t = M_{t-1}(I - β k kᵀ) + β v kᵀ → 同 key 重写覆盖,无 key flood 应保留。
    所以 fresh-init 下:n=0 cos→1.0(最纯净);N>0 后 cos 随 distract 单调下降(distract key
    与 target key 偶尔接近 → 部分覆盖),但绝不归零(delta 删旧关联机制有效)。

    经 backbone 的 end-to-end 保真度由 ablate-mac-mal / 真训练 ckpt 上的 targeted_probe 验。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)

    sweep = [int(x) for x in args.n_distract_sweep.split(",") if x.strip()]
    hippo = model.global_hippo
    d_model = hippo.d_model
    H, dk, dv = hippo.H, hippo.dk, hippo.dv
    d_key = hippo.d_key
    d_value = hippo.d_value

    color_info(f"HippoDelta: d_model={d_model} H={H} d_key={d_key}(dk/H={dk}) "
               f"d_value={d_value}(dv/H={dv})")

    results: list[dict] = []
    rng = torch.Generator(device=dev).manual_seed(args.seed)
    for n_distract in sweep:
        # 合成 (n_distract+1) 个 d_model 维 random 向量(模拟 _d_total_in 输出),逐个 write
        # 每个"虚拟 turn"长度 1 token,共 n_distract+1 个 turn,最后一个是 target
        # 用 _project 把 random x 投成 (k, v, β) 然后 HippoDelta.write
        x_seq = torch.randn(1, n_distract + 1, d_model, device=dev, generator=rng)
        x_seq = torch.nn.functional.normalize(x_seq, dim=-1) * (d_model ** 0.5)
        # target 是最后一个 turn
        state = None
        with autocast_ctx(dev), torch.no_grad():
            # 逐 turn write(长度 1)
            for t in range(n_distract + 1):
                xt = x_seq[:, t:t + 1, :]  # (1, 1, d_model)
                state = hippo.write(xt, state)
            M = state.M  # (1, H, dv, dk)

            # 用 target 的同一个 x 拿 ideal k/v(走 _project)
            x_target = x_seq[:, -1:, :]
            k_t, v_t, beta_t, _ = hippo._project(x_target)
            # k_t: (1, H, 1, dk), v_t: (1, H, 1, dv)
            # 构造 q_ideal 与 k_t 同形(B, n_q=1, d_key):reshape (1,H,1,dk) → (1,1,H*dk)
            q_ideal = k_t.transpose(1, 2).reshape(1, 1, d_key)
            r = hippo.retrieve(M, q_ideal)  # (1, 1, d_value)
            ideal_v = v_t.transpose(1, 2).reshape(1, 1, d_value)
            # per-head cosine 平均
            r_h = r.view(1, 1, H, dv).squeeze(0).squeeze(0).float()
            iv_h = ideal_v.view(1, 1, H, dv).squeeze(0).squeeze(0).float()
            cos = torch.nn.functional.cosine_similarity(r_h, iv_h, dim=-1).mean().item()
            # 每 head 也算,看是否有 head 完全 dead
            per_head_cos = torch.nn.functional.cosine_similarity(r_h, iv_h, dim=-1).tolist()
        results.append({"n_distract": n_distract, "cos": cos,
                        "per_head_cos": [round(c, 3) for c in per_head_cos],
                        "M_specnorm": float(hippo.last_M_specnorm.item())})
        color_info(f"n_distract={n_distract:<3}  cos(retrieved, ideal_v)={cos:+.4f}  "
                   f"M_spec={hippo.last_M_specnorm.item():.3f}  per_head_cos_range=["
                   f"{min(per_head_cos):+.2f},{max(per_head_cos):+.2f}]")

    cos_by_n = {r["n_distract"]: r["cos"] for r in results if r["cos"] is not None}
    fails: list[str] = []
    if 0 in cos_by_n and cos_by_n[0] < 0.95:
        fails.append(f"n=0 cos={cos_by_n[0]:.3f} < 0.95"
                     f"(空 M 单写单读都不准 → delta rule / β·‖k‖² 约束有 bug)")
    if 8 in cos_by_n and cos_by_n[8] < 0.5:
        fails.append(f"n=8 cos={cos_by_n[8]:.3f} < 0.5"
                     f"(delta 删旧关联机制弱,distract 已淹 target)")
    if 16 in cos_by_n and cos_by_n[16] < 0.2:
        fails.append(f"n=16 cos={cos_by_n[16]:.3f} < 0.2"
                     f"(归零,M 完全失忆 target)")
    if fails:
        return {"pass": False, "msg": "; ".join(fails),
                "details": {"results": results}}
    return {"pass": True,
            "msg": f"delta rule write→retrieve 保真 OK: cos by n_distract = {cos_by_n}",
            "details": {"results": results}}


# ───────────────────────── 子命令 6: query-diversity ─────────────────────────


def cmd_query_diversity(args) -> dict:
    """QueryHead 多样性:16 episode × n_query 个 q,跨/单 episode diversity。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)
    tok = lazy_tokenizer(cfg)

    from scripts.generate_recall_probe import ENTITY_PATTERNS
    pat = next(p for p in ENTITY_PATTERNS if p["type"] == "name_en")
    n_ep = 16
    if n_ep > len(pat["pool"]):
        n_ep = len(pat["pool"])
    if n_ep == 0:
        raise ValueError("n_episodes == 0:实体池为空")
    seg_len = cfg.turn_max_tokens
    entities = random.Random(args.seed).sample(pat["pool"], n_ep)

    all_q: list[torch.Tensor] = []
    intra_div: list[float] = []
    from xinhe.data.conversation import tokenize_turn
    from xinhe.model.query_head import QueryHead

    for e in entities:
        turns = build_nih_turns(e, n_distract=2, pat=pat)
        state = evolve_turns(model, tok, turns[:-1], dev, seg_len)
        # 在 query 前取 q(模拟 forward 内部第一步)
        ur, ar, _ = turns[-1]
        # 走到 read prefix 末位 token,取 q
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": ur}], tokenize=False, add_generation_prompt=True,
        )
        pids = torch.tensor([tok.encode(prefix, add_special_tokens=False)],
                            dtype=torch.long, device=dev)
        with autocast_ctx(dev), torch.no_grad():
            emb = model.backbone.embed(pids)
            valid = (pids != tok.pad_token_id).long().sum(dim=1).clamp(min=1) - 1
            h_last = emb[torch.arange(emb.shape[0], device=dev), valid]
            q = model.query_head(h_last)  # (1, n_q, d_key)
        all_q.append(q.float().cpu())
        intra = QueryHead.cosine_diversity(q).item()
        intra_div.append(intra)

    stacked = torch.cat([qq.reshape(-1, qq.shape[-1]) for qq in all_q], dim=0)
    cross = QueryHead.cosine_diversity(stacked).item()
    intra_mean = sum(intra_div) / len(intra_div)
    color_info(f"n_episodes={n_ep}  intra-episode mean diversity={intra_mean:.4f}")
    color_info(f"cross-episode diversity (over {stacked.shape[0]} q)={cross:.4f}")
    if cross < 0.3:
        return {"pass": False,
                "msg": f"cross-episode diversity {cross:.3f} < 0.3 → q mode collapse",
                "details": {"cross": cross, "intra_mean": intra_mean,
                            "intra_each": intra_div}}
    return {"pass": True, "msg": f"cross={cross:.3f} > 0.3 OK(intra mean={intra_mean:.3f})",
            "details": {"cross": cross, "intra_mean": intra_mean,
                        "n_q_total": int(stacked.shape[0])}}


# ───────────────────────── 子命令 7: per-head-spec ─────────────────────────


def cmd_per_head_spec(args) -> dict:
    """单步写入后每 head 谱范数 → 找 dead head(spec → 0 或爆 50)。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)
    tok = lazy_tokenizer(cfg)

    from scripts.generate_recall_probe import ENTITY_PATTERNS
    pat = next(p for p in ENTITY_PATTERNS if p["type"] == "name_en")
    random.seed(args.seed)
    entity = random.choice(pat["pool"])
    turns = build_nih_turns(entity, n_distract=8, pat=pat)[:-1]
    seg_len = cfg.turn_max_tokens
    state = evolve_turns(model, tok, turns, dev, seg_len)
    gkey = model._global_write_idx
    layer = state.layers.get(gkey)
    if layer is None or layer.hippo is None or layer.hippo.M is None:
        return {"pass": False, "msg": "写入后 M 仍为 None",
                "details": {"gkey": gkey}}
    M = layer.hippo.M[0].float()  # (H, dv, dk)
    H = M.shape[0]
    specs: list[float] = []
    for h in range(H):
        s = torch.linalg.matrix_norm(M[h], ord=2).item()
        specs.append(s)
    dead = [(h, s) for h, s in enumerate(specs) if s < 1e-3]
    explode = [(h, s) for h, s in enumerate(specs) if s > 50.0]

    color_info(f"H={H}  spec norm per head:")
    for h, s in enumerate(specs):
        tag = RED if (s < 1e-3 or s > 50.0) else GREEN
        bar = "▏" * min(int(s * 4), 40)
        print(f"    {tag}h{h:02d}  spec={s:7.4f}{RESET}  {bar}")

    if dead or explode:
        return {"pass": False,
                "msg": f"{len(dead)} dead, {len(explode)} explode(want all ∈ (1e-3, 50))",
                "details": {"specs": specs, "dead": dead, "explode": explode}}
    return {"pass": True, "msg": f"全 {H} head spec ∈ (1e-3, 50)  range=[{min(specs):.3f}, {max(specs):.3f}]",
            "details": {"specs": specs}}


# ───────────────────────── 子命令 8: spec-evolution ─────────────────────────


def cmd_spec_evolution(args) -> dict:
    """16-turn NIH 演化 M 谱范数;ASCII bar chart。"""
    model, cfg, dev = make_model_fresh(args.config, args.device, args.seed)
    maybe_load_ckpt(model, args.ckpt, dev)
    tok = lazy_tokenizer(cfg)

    from scripts.generate_recall_probe import ENTITY_PATTERNS
    pat = next(p for p in ENTITY_PATTERNS if p["type"] == "name_en")
    random.seed(args.seed)
    entity = random.choice(pat["pool"])
    # 至少 16 turn:用 16 distract + 1 target(共 17 个写 turn)
    n_distract = min(16, len(pat["pool"]) - 1)
    turns = build_nih_turns(entity, n_distract=n_distract, pat=pat)[:-1]
    seg_len = cfg.turn_max_tokens

    model.global_hippo._spec_history = []
    model.global_hippo._spec_log_enabled = True
    try:
        evolve_turns(model, tok, turns, dev, seg_len)
        history = list(model.global_hippo._spec_history)
    finally:
        model.global_hippo._spec_log_enabled = False
        model.global_hippo._spec_history = []

    if not history:
        return {"pass": False, "msg": "spec_history 空(write step 未触发)",
                "details": {"n_turns": len(turns)}}
    cap = cfg.spectral_norm_cap
    color_info(f"spec_history len={len(history)}  cap={cap}  "
               f"max={max(history):.4f}  min={min(history):.4f}")
    color_info("ASCII bar:")
    mx = max(max(history), 1e-6)
    for i, s in enumerate(history):
        bar_len = int(s / mx * 40)
        tag = RED if not (s == s and abs(s) < cap * 5) else GREEN
        print(f"    {tag}turn {i:02d}  spec={s:7.4f}{RESET}  {'▏' * bar_len}")

    all_finite = all((s == s and abs(s) < float("inf")) for s in history)
    under_cap = all(s < cap * 5 for s in history)
    enough = len(history) >= 8
    if not all_finite:
        return {"pass": False, "msg": "存在 NaN/Inf",
                "details": {"history": history}}
    if not under_cap:
        return {"pass": False, "msg": f"max(spec)={max(history):.2f} > cap×5={cap * 5}",
                "details": {"history": history}}
    if not enough:
        return {"pass": False,
                "msg": f"spec_history len={len(history)} < 8(写入触发不足)",
                "details": {"history": history, "n_turns": len(turns)}}
    return {"pass": True,
            "msg": f"{len(history)} 次写入,max spec={max(history):.3f} (cap={cap})",
            "details": {"history": history}}


# ───────────────────────── 子命令 9: backend-parity ─────────────────────────


def cmd_backend_parity(args) -> dict:
    """FLA vs torch 双后端 forward 对齐(WSL Linux+CUDA only)。"""
    from xinhe.model.delta_kernel import _FLA_AVAILABLE, torch_delta_chunk, _fla_write
    if not _FLA_AVAILABLE:
        return {"pass": True, "msg": "FLA 不可用(Windows native;Linux WSL 装 fla 后跑)",
                "details": {"fla_available": False}, "skip": True}
    if args.device != "cuda" or not torch.cuda.is_available():
        return {"pass": True, "msg": "需要 CUDA;CPU 跳过",
                "details": {"device": args.device}, "skip": True}

    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    B, H, T, dk, dv = 1, 4, 16, 64, 32
    # 模仿 _project 之后的输入:k 归一,β∈(0,1)
    k = torch.randn(B, H, T, dk, device=dev)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(B, H, T, dv, device=dev)
    beta = torch.sigmoid(torch.randn(B, H, T, device=dev)) * 0.5  # 保 β‖k‖²<1
    M_prev = torch.zeros(B, H, dv, dk, device=dev)

    M_torch = torch_delta_chunk(M_prev.float(), k.float(), v.float(), beta.float())
    # FLA 要 bf16 输入
    M_fla = _fla_write(
        M_prev.float(), k.to(torch.bfloat16),
        v.to(torch.bfloat16), beta.to(torch.bfloat16),
    )

    t_flat = M_torch.flatten().float()
    f_flat = M_fla.flatten().float()
    cos = torch.nn.functional.cosine_similarity(
        t_flat.unsqueeze(0), f_flat.unsqueeze(0)).item()
    diff = (t_flat - f_flat).abs().max().item()
    norm = t_flat.norm().item()
    rel = diff / max(norm, 1e-8)
    color_info(f"cos(M_torch, M_fla)={cos:.6f}  max|Δ|={diff:.4e}  rel={rel:.4e}")
    if cos < 0.99 or rel > 0.01:
        return {"pass": False,
                "msg": f"FLA/torch 对齐失败:cos={cos:.4f} (want>0.99) rel={rel:.4e} (want<1%)",
                "details": {"cos": cos, "max_abs_diff": diff,
                            "rel_diff": rel, "M_torch_norm": norm}}
    return {"pass": True, "msg": f"FLA/torch parity OK cos={cos:.4f} rel={rel:.4e}",
            "details": {"cos": cos, "max_abs_diff": diff, "rel_diff": rel}}


# ───────────────────────── all orchestrator ─────────────────────────


COMMANDS_ORDER = [
    "dtype-walk", "mal-layer-trace", "ablate-mac-mal", "state-trace",
    "write-fidelity", "query-diversity", "per-head-spec", "spec-evolution",
    "backend-parity",
]

COMMANDS = {
    "dtype-walk": cmd_dtype_walk,
    "mal-layer-trace": cmd_mal_layer_trace,
    "ablate-mac-mal": cmd_ablate_mac_mal,
    "state-trace": cmd_state_trace,
    "write-fidelity": cmd_write_fidelity,
    "query-diversity": cmd_query_diversity,
    "per-head-spec": cmd_per_head_spec,
    "spec-evolution": cmd_spec_evolution,
    "backend-parity": cmd_backend_parity,
}


def cmd_all(args) -> dict:
    n_pass = n_fail = n_skip = 0
    failed_at: str | None = None
    skipped: list[str] = []
    for name in COMMANDS_ORDER:
        if failed_at is not None:
            color_skip(f"{name}: stopped (前面失败于 {failed_at})")
            skipped.append(name)
            n_skip += 1
            continue
        print(f"\n{CYAN}=== {name} ==={RESET}")
        try:
            res = COMMANDS[name](args)
        except Exception as e:
            color_fail(f"{name}: 异常 {type(e).__name__}: {e}")
            n_fail += 1
            failed_at = name
            continue
        if res.get("skip"):
            color_skip(f"{name}: {res['msg']}")
            n_skip += 1
            continue
        if res["pass"]:
            color_pass(f"{name}: {res['msg']}")
            n_pass += 1
        else:
            color_fail(f"{name}: {res['msg']}")
            n_fail += 1
            failed_at = name
    print(f"\n{CYAN}=== summary ==={RESET}")
    print(f"  {GREEN}pass={n_pass}{RESET}  {RED}fail={n_fail}{RESET}  {YELLOW}skip={n_skip}{RESET}")
    return {"pass": n_fail == 0, "msg": f"{n_pass} pass / {n_fail} fail / {n_skip} skip",
            "details": {"failed_at": failed_at, "skipped": skipped}}


# ───────────────────────── 入口 ─────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="心核 NM debug bench(9 子命令 forward-only 白盒诊断)",
    )
    ap.add_argument("--config", default="configs/pcap_skeleton.yaml")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--ckpt", default=None, help="可选 ckpt 路径;None 用 fresh-init")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in COMMANDS_ORDER + ["all"]:
        sp = sub.add_parser(name)
        if name in ("write-fidelity", "all"):
            sp.add_argument("--n-distract-sweep", default="0,1,3,8,16",
                            help="逗号分隔,如 0,1,3,8,16")
    args = ap.parse_args()

    if args.cmd == "all":
        # 确保 sub-args 默认值传给各子命令(write-fidelity 需要 n_distract_sweep)
        if not hasattr(args, "n_distract_sweep"):
            args.n_distract_sweep = "0,1,3,8,16"
        res = cmd_all(args)
        return 0 if res["pass"] else 1

    print(f"{CYAN}=== {args.cmd} ==={RESET}")
    handler = COMMANDS[args.cmd]
    res = handler(args)
    if res.get("skip"):
        color_skip(f"{args.cmd}: {res['msg']}")
        return 0
    if res["pass"]:
        color_pass(f"{args.cmd}: {res['msg']}")
        return 0
    color_fail(f"{args.cmd}: {res['msg']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
