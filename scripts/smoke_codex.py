"""codex driver：用本地 codex CLI(走用户 ChatGPT 配额)生成 stage1 5-Beat 数据。

每条 sample 流式落盘 + 断点续跑(resume):
  1. BeatPlanner 出 plan
  2. 写 SPEC.md 到系统 tempdir + 让 codex agent 读 SPEC + 写 output.json(自动 cleanup)
  3. 解析 output.json → parser → validator
  4. 通过的 append 写到 train_codex.jsonl(立即 flush 落盘)
  5. 启动时检测已有行数,从断点续跑

注意 codex Windows sandbox 有 CreateProcessAsUserW failed bug,必须 --dangerously-bypass。

用法:
    python scripts/smoke_codex.py --n 1000
    python scripts/smoke_codex.py --n 1000 --out data/v8/stage1/train_codex.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from xinhe.data.stage1.beat_planner import BeatPlanner
from xinhe.data.stage1.parser import parse_response, ParseError
from xinhe.data.stage1.prompts import build_messages
from xinhe.data.validator.api import validate


_CODEX_BIN = None
_CALL_REASONING_EFFORT = None  # 由 main 设置, 透传给 call_codex_via_files


def _resolve_codex() -> str:
    global _CODEX_BIN
    if _CODEX_BIN:
        return _CODEX_BIN
    for cand in ("codex.cmd", "codex.exe", "codex"):
        p = shutil.which(cand)
        if p:
            _CODEX_BIN = p
            return p
    raise RuntimeError("找不到 codex 可执行文件")


_OUTPUT_SCHEMA_JSON = """{
  "type": "object",
  "additionalProperties": false,
  "required": ["conversations"],
  "properties": {
    "conversations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["role", "content", "train_loss", "value", "beat"],
        "properties": {
          "role": {"type": "string", "enum": ["user", "assistant"]},
          "content": {"type": "string"},
          "train_loss": {"type": "string", "enum": ["true", "lm_only", "false"]},
          "value": {
            "anyOf": [
              {"type": "array", "items": {"type": "string"}},
              {"type": "null"}
            ]
          },
          "beat": {"type": "integer", "minimum": 1, "maximum": 5}
        }
      }
    }
  }
}
"""


def call_codex_via_files(plan, *, model: str | None = None,
                         timeout: int = 240) -> str:
    """codex stdin + --output-schema 模式:prompt 走 stdin,JSON 走 --output-last-message。
    比早期 SPEC.md + agent 写文件大法省一段 shell round trip + token(实测 ~30%)。
    Windows 上必须 --dangerously-bypass-approvals-and-sandbox(Win sandbox bug)。"""
    work_dir = Path(tempfile.mkdtemp(prefix="codex_smp_"))
    try:
        out_file = work_dir / "out.json"
        schema_file = work_dir / "schema.json"
        schema_file.write_text(_OUTPUT_SCHEMA_JSON, encoding="utf-8")

        sys_p, user_p = build_messages(plan)
        prompt_content = (
            "# 系统规范\n\n" + sys_p + "\n\n"
            "# 任务参数\n\n" + user_p + "\n"
        )

        cmd = [
            _resolve_codex(), "exec",
            "--ephemeral", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(work_dir),
            # 关掉用不到的 tool, 减少 token 浪费 + 解锁 minimal/none reasoning
            "-c", "features.image_generation=false",
            "-c", 'web_search="disabled"',
            "--output-schema", str(schema_file),
            "--output-last-message", str(out_file),
        ]
        if model:
            cmd.extend(["-m", model])
        if _CALL_REASONING_EFFORT:
            cmd.extend(["-c", f"model_reasoning_effort={_CALL_REASONING_EFFORT}"])
        cmd.append("-")  # stdin sentinel,prompt 走 stdin

        r = subprocess.run(
            cmd,
            input=prompt_content,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8",
        )
        if not out_file.exists():
            raise RuntimeError(f"codex 没写出文件; rc={r.returncode}; stdout tail: {r.stdout[-300:]!r}")
        return out_file.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_stage1_data_cfg(yaml_path: Path) -> dict:
    """从主线 yaml 读 stage1 的 data 段(单一配置源,与 generate_data.py 对齐)。"""
    from xinhe.model.config import XinheConfig
    _cfg, curriculum = XinheConfig.from_yaml(str(yaml_path))
    for stage in curriculum:
        data = stage.get("data", {})
        if data.get("stage_kind") == "stage1":
            return data
    raise RuntimeError(f"yaml {yaml_path} 没找到 stage_kind=stage1 的 stage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path("configs/persona_unified_v8_0.8b.yaml"),
                    help="主线 yaml,从中读 beat3 配置(单一配置源)")
    ap.add_argument("--n", type=int, default=1000, help="目标总条数(含已有)")
    ap.add_argument("--out", type=Path, default=Path("data/v8/stage1/train_codex.jsonl"))
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model", type=str, default=None,
                    help="覆写 codex 默认 model，不传用 codex 默认（如 gpt-5）")
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                    help="覆写 model_reasoning_effort. mini 模型支持 none/low/medium/high/xhigh, "
                         "gpt-5.4 支持 minimal/low/medium/high")
    ap.add_argument("--force", action="store_true", help="覆盖已有文件,从 0 重跑")
    args = ap.parse_args()
    global _CALL_REASONING_EFFORT
    _CALL_REASONING_EFFORT = args.reasoning_effort

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.force and args.out.exists():
        args.out.unlink()

    existing = _count_existing(args.out)
    n_remaining = max(0, args.n - existing)
    if n_remaining == 0:
        print(f"[codex] 已满 ({existing} ≥ {args.n})，跳过 {args.out}")
        return
    print(f"[codex] resume: 已有 {existing} 条，补 {n_remaining} 条到 {args.n}")

    fp = open(args.out, "a", encoding="utf-8")

    # seed 加上 existing 偏移,避免重复跑同样 plan
    rng = random.Random(args.seed + existing)
    # 从主线 yaml 读 beat3 参数, 单一配置源, 跟 generate_data.py 同源
    data_cfg = _load_stage1_data_cfg(args.config)
    beat3_min_chars = int(data_cfg.get("beat3_min_chars", 500))
    beat3_chars_tolerance = float(data_cfg.get("beat3_chars_tolerance", 0.8))
    planner = BeatPlanner(
        beat3_min_chars=beat3_min_chars,
        beat3_chars_tolerance=beat3_chars_tolerance,
    )
    print(f"=== codex driver (model={args.model or 'default-gpt5'}) "
          f"n_remaining={n_remaining} beat3={beat3_min_chars}/tol={beat3_chars_tolerance} ===")

    n_ok = existing
    consec_fail = 0
    MAX_CONSEC_FAIL = 5  # 连续 N 次失败(配额满 / API 挂)自动 break, 避免空转
    t0 = time.time()
    for i in range(n_remaining):
        ts = time.time()
        try:
            plan = planner.plan(rng)
            raw = call_codex_via_files(plan, model=args.model)
            sample = parse_response(raw, plan, generator_model="codex-cli")
            result = validate(sample.to_dict(), stage="1", plan=plan.to_validator_plan())
            dt = time.time() - ts
            if not result.ok:
                print(f"[{i+1}/{args.n}] REJECT ({dt:.1f}s) errors={result.errors[:1]}")
                consec_fail += 1
                if consec_fail >= MAX_CONSEC_FAIL:
                    print(f"!! 连续 {MAX_CONSEC_FAIL} 次失败,break(可能配额满 / API 挂)", flush=True)
                    break
                continue
            d = sample.to_dict()
            b3 = sum(len([c for c in t["content"] if "一" <= c <= "鿿"])
                     for t in d["conversations"]
                     if t.get("role") == "assistant" and t.get("train_loss") == "lm_only")
            facts = [(f["canonical_value"], f["scope"]) for f in d["meta"]["canonical_facts"]]
            print(f"[{n_ok+1}/{args.n}] OK ({dt:.1f}s) n_turns={d['meta']['n_turns']} beat3_zh={b3} recall={d['meta']['recall_form']} facts={facts}", flush=True)
            fp.write(json.dumps(d, ensure_ascii=False) + "\n")
            fp.flush()
            n_ok += 1
            consec_fail = 0  # 成功重置
        except (ParseError, RuntimeError, subprocess.TimeoutExpired) as e:
            dt = time.time() - ts
            print(f"[reject {i+1}/{n_remaining}] FAIL ({dt:.1f}s) {type(e).__name__}: {str(e)[:160]}", flush=True)
            consec_fail += 1
            if consec_fail >= MAX_CONSEC_FAIL:
                print(f"!! 连续 {MAX_CONSEC_FAIL} 次失败,break(可能配额满 / API 挂)", flush=True)
                break
        except ValueError as e:
            # BeatPlanner 偶发采样失败(canonical 池空), seed 推进即可恢复, 不计 consec_fail
            print(f"[skip {i+1}/{n_remaining}] PLAN_FAIL: {str(e)[:120]}", flush=True)

    fp.close()
    total = time.time() - t0
    new_ok = n_ok - existing
    print()
    print(f"=== 总结 ===")
    print(f"  本轮新增 {new_ok}/{n_remaining} ({new_ok*100/max(1,n_remaining):.0f}%)")
    print(f"  总数 {n_ok}/{args.n}")
    print(f"  总耗时 {total:.1f}s, 平均 {total/max(1,new_ok):.1f}s/成功条")
    print(f"  落盘 → {args.out}")


if __name__ == "__main__":
    main()
