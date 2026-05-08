"""
评估入口 — token-level argmax 准确率,与训练目标完全一致。

两种模式:

  1. 单文件 / 目录扫描 (灵活探针):
       python scripts/evaluate.py --checkpoint A1.pt --val data/skeleton/val.jsonl
       python scripts/evaluate.py --checkpoint A1.pt --val-dir data/skeleton

  2. Stage joint 模式 (复用训练早停指标):
       python scripts/evaluate.py --checkpoint A1.pt \\
           --config configs/persona_unified_0.8b_A1.yaml \\
           --stage 0_atomic_skeletons

     该模式从 yaml 的 curriculum 中找指定 stage,读其 val_sets,
     然后调用 xinhe.evaluation.event_eval.eval_joint — 输出与训练
     时 trainer 早停判定完全相同的指标 dict (overall / S{1..11} /
     distance_* / tier_* / substream_*)。

  --output result.json 可把任一模式的结果落盘。
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.evaluation.event_eval import eval_event_jsonl, eval_joint
from xinhe.model.config import XinheConfig
from xinhe.model.xinhe_model import XinheModel


def load_model_and_tokenizer(config, checkpoint_path, device):
    model = XinheModel(config)

    # eval 关 NM 的 torch.compile:Dynamo 在 no_grad 下 trace NeuralMemState pytree
    # 触发 side_effects.codegen_save_tempvars AssertionError(cls_source 缺失)
    # 训练期 grad 路径无此问题,只 eval 关
    for pair in model.memory.values():
        pair.hippocampus.use_compile_chunk_loop = False

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "memory_pair_state" not in checkpoint:
            raise RuntimeError(
                "checkpoint 缺少 'memory_pair_state' 键。仅兼容 v9+ 格式。"
            )
        mem_state = dict(checkpoint["memory_pair_state"])
        # legacy ckpt 残留 alpha_logit(已被 Phase 0 alpha 清理移除),pop 掉避免 strict=True 失败
        legacy_alpha_keys = [k for k in mem_state if k.endswith(".alpha_logit")]
        for k in legacy_alpha_keys:
            mem_state.pop(k)
        if legacy_alpha_keys:
            print(f"  [legacy] 跳过 ckpt 中已移除的 alpha_logit 键 ({len(legacy_alpha_keys)} 个)")

        # ckpt 在 compile_backbone_layers=True 下保存时,neocortex 子模块被 OptimizedModule
        # 包了一层,key 带 `_orig_mod.` 前缀。eval 关 compile 时 strip 掉这层。
        orig_mod_keys = [k for k in mem_state if "._orig_mod." in k]
        if orig_mod_keys and not getattr(config, "compile_backbone_layers", False):
            for k in orig_mod_keys:
                mem_state[k.replace("._orig_mod.", ".")] = mem_state.pop(k)
            print(f"  [legacy] 剥离 _orig_mod. 前缀 ({len(orig_mod_keys)} 个,compile 关闭兼容)")

        model.memory.load_state_dict(mem_state, strict=True)
        # v9.5 MAC 参数(mem_token_init)可选加载
        for key in ("mem_token_init",):
            param = getattr(model, key, None)
            if param is not None and key in checkpoint:
                with torch.no_grad():
                    param.copy_(checkpoint[key].to(param.device))
        # v9.5 backbone addons(LoRA + per-layer K/V),strict=False 兼容旧 ckpt
        if "backbone_addons_state" in checkpoint:
            addons = {k: v.to(device) for k, v in checkpoint["backbone_addons_state"].items()}
            model.backbone.load_state_dict(addons, strict=False)

    model.to(device)
    model.eval()

    from transformers import AutoTokenizer
    from xinhe.data.conversation import ensure_chat_template
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(config.backbone_model_path).resolve()),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ensure_chat_template(tokenizer)

    return model, tokenizer


def _discover_val_jsonl(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".jsonl" else []
    return sorted(root.rglob("*.jsonl"))


def _print_dim(label: str, dim: dict | None):
    if dim:
        line = " ".join(f"{k}={v['acc']:.2%}({v['n']})" for k, v in dim.items())
        print(f"    {label}: {line}")


def run_single_files(model, tokenizer, config, args, device) -> dict:
    """模式 1: --val / --val-dir 的逐文件评估。"""
    val_files: list[Path] = []
    if args.val:
        val_files.append(Path(args.val))
    if args.val_dir:
        root = Path(args.val_dir)
        if root.exists():
            val_files.extend(_discover_val_jsonl(root))
    val_files = sorted(set(val_files))
    if not val_files:
        print("没有找到 val 文件。请通过 --val / --val-dir / --stage 指定。")
        return {}

    seg_len = args.seg_len or getattr(config, "turn_max_tokens", 256)
    results = {}
    for vf in val_files:
        name = vf.relative_to(project_root) if vf.is_absolute() else vf
        print(f"\n[{name}]")
        res = eval_event_jsonl(
            model, tokenizer, vf,
            device=device, seg_len=seg_len, max_episodes=args.max_episodes,
        )
        print(f"  episodes={res['n_episodes']}, value_turns={res['n_value_turns']}, "
              f"correct={res['n_correct']}, overall_acc={res['overall_acc']:.2%}")
        for label, key in (("skeleton", "by_skeleton"), ("distance", "by_distance"),
                           ("tier", "by_tier"), ("substream", "by_substream")):
            _print_dim(label, res.get(key))
        results[str(name)] = res
    return results


def run_joint_stage(model, tokenizer, config, curriculum, args, device) -> dict:
    """模式 2: --stage <name>，复用训练早停的 joint 评估。"""
    if not curriculum:
        raise SystemExit(
            f"--stage 模式要求 --config 指向带 curriculum 的 yaml,但 {args.config} 没有 curriculum。"
        )
    stage_names = [s["name"] for s in curriculum]
    stage = next((s for s in curriculum if s["name"] == args.stage), None)
    if stage is None:
        raise SystemExit(
            f"--stage '{args.stage}' 不在 curriculum 中。可用 stage: {stage_names}"
        )

    val_sets = stage.get("val_sets") or stage.get("data", {}).get("val_sets") or []
    if not val_sets:
        raise SystemExit(f"--stage '{args.stage}' 没有 val_sets,无法 joint 评估")

    # 把 stage 的 val_sets / turn_max_tokens 注入 config(eval_joint 消费)
    stage_training = stage.get("training", {})
    seg_len = args.seg_len or stage_training.get("turn_max_tokens", config.turn_max_tokens)
    config = replace(config, val_sets=val_sets, turn_max_tokens=seg_len)

    print(f"  stage: {args.stage}")
    print(f"  val_sets: {[v.get('name') for v in val_sets]}")
    print(f"  seg_len: {seg_len} | max_episodes: {args.max_episodes}")

    flat = eval_joint(
        model, tokenizer, config,
        device=device, max_episodes=args.max_episodes,
    )

    # 按 val_set 分组打印
    print("\n=== Joint 指标(与训练早停一致) ===")
    seen_groups: list[str] = []
    for v in val_sets:
        seen_groups.append(v.get("name", ""))
    for group in seen_groups:
        keys = sorted(k for k in flat if k.startswith(f"{group}_"))
        if not keys:
            continue
        print(f"\n  [{group}]")
        for k in keys:
            short = k[len(group) + 1:]
            print(f"    {short}: {flat[k]:.2%}")

    # 早停阈值复盘:配 yaml 里的 early_stop dict 时,顺手算"哪些已过线"
    es_thresholds = stage_training.get("early_stop") or {}
    if es_thresholds:
        print("\n  [早停阈值复盘]")
        all_pass = True
        for metric, thresh in es_thresholds.items():
            val = flat.get(metric)
            if val is None:
                print(f"    {metric}: (缺) ≥ {thresh:.2%} ✗")
                all_pass = False
            else:
                ok = val >= thresh
                all_pass = all_pass and ok
                print(f"    {metric}: {val:.2%} ≥ {thresh:.2%}  {'✓' if ok else '✗'}")
        print(f"  → AND 早停判定: {'PASS' if all_pass else 'FAIL'}")

    return flat


def main():
    parser = argparse.ArgumentParser(description="心核事件级评估")
    parser.add_argument("--checkpoint", required=True, help="ckpt 路径")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="模型/课程配置 yaml(joint 模式必需带 curriculum)")
    # 模式 1: 灵活探针
    parser.add_argument("--val", default=None, help="单文件 jsonl")
    parser.add_argument("--val-dir", default=None,
                        help="目录,递归找 jsonl(与 --stage 互斥)")
    # 模式 2: stage joint
    parser.add_argument("--stage", default=None,
                        help="curriculum 里的 stage 名,启用 joint 早停指标评估")
    # 公共
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--seg-len", type=int, default=None,
                        help="覆盖 turn_max_tokens(默认从 config / stage 读)")
    parser.add_argument("--output", default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    if args.stage and (args.val or args.val_dir):
        raise SystemExit("--stage 与 --val/--val-dir 互斥,选其一")
    if not args.stage and not args.val and not args.val_dir:
        raise SystemExit(
            "至少指定一种评估范围:\n"
            "  --stage <name>          (joint 模式,自动按 skeleton/distance/tier/substream 分桶)\n"
            "  --val <jsonl>           (单文件)\n"
            "  --val-dir <path>        (递归 jsonl 目录)"
        )

    # 加载 config(joint 模式取 curriculum,single 模式只用 base)
    config, curriculum = XinheConfig.from_yaml(args.config)
    config_explicit = "--config" in sys.argv
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not config_explicit and isinstance(checkpoint.get("config"), XinheConfig):
        config = checkpoint["config"]
        curriculum = []  # ckpt 里的 config 不带 curriculum
        print(f"  使用 checkpoint 内置配置: backbone={config.backbone_type}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("=== 心核评估 ===")
    print(f"  ckpt: {args.checkpoint}")
    print(f"  config: {args.config}")
    print(f"  device: {device} | max_episodes: {args.max_episodes}")
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint, device)

    if args.stage:
        results = run_joint_stage(model, tokenizer, config, curriculum, args, device)
    else:
        results = run_single_files(model, tokenizer, config, args, device)

    if args.output and results:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 {args.output}")

    print("\n评估完成。")


if __name__ == "__main__":
    main()
