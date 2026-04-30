#!/usr/bin/env python3
"""
本地 data/ → HuggingFace 数据集仓库的幂等同步工具。

用法:
    uv run python scripts/upload_to_hf.py             # 真推
    uv run python scripts/upload_to_hf.py --dry-run   # 只看清单
    uv run python scripts/upload_to_hf.py --repo X    # 换仓库

约定:
    data/<kind>/<split>.jsonl
    → HF config_name = "<kind>", split = <split>(val 自动映射 validation)

同步语义:
    - 本地新增 jsonl → 上传
    - 本地删除 jsonl → 远端一并删除
    - README.md 由本脚本动态生成覆盖（手改 HF 网页 README 会被覆盖！）
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    get_token,
    login,
)
from huggingface_hub.utils import RepositoryNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "flufy3d/xinhe-dataset"
DEFAULT_DATA_ROOT = "data"

# val.jsonl → HF 惯例 split 名 validation
SPLIT_NAME_REMAP = {"val": "validation"}


# ── README 正文（改文案就改这里）──────────────────────────────

README_BODY = """\
# Xinhe Persona Memory Dataset

合成中文人格记忆对话数据集，用于 [Xinhe (心核)](https://github.com/flufy3d/xinhe) 项目研究小型 Transformer 在统一状态中涌现记忆能力。

每条样本是一段多轮中文对话，包含：用户陈述/修正/查询自身画像 + 助手回应。中间穿插与画像无关的日常话题作为干扰。辅助字段（`value` / `value_span` / `value_tier` / `weight_per_span`）由 parser 在生成后定位，用于训练时构造 token 级加权损失，可计算召回准确率。

## 数据生成

由 LLM (DeepSeek / OpenRouter 多模型) 在指定骨架 (skeleton) 下合成，再由后处理 parser 在 assistant 回答中定位 value 字符跨度。

## Schema

每行 jsonl 一条样本：

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 样本 hash 短 ID |
| `stage` | str | 课程阶段编号（与 config_name 对齐） |
| `skeleton_id` | str | 对话骨架模板 ID |
| `meta` | object | n_turns / target_turns / distance_bucket (near/mid/far) / memory_snapshot |
| `conversations` | list | 对话轮次：`role` ∈ {user, assistant} + `content`；assistant 多含 `train_loss` / `value` / `value_span` / `value_tier` / `weight_per_span` |

## 加载

```python
from datasets import load_dataset

ds = load_dataset("flufy3d/xinhe-dataset", "skeleton", split="train")
print(ds[0]["conversations"])
```

config_name 直接用 kind 名(`skeleton` / `dialog` / `mix`)。每个 config 含 `train` / `validation`,部分还含 `train_codex`(codex-flavored 变体)。

## 国内镜像

国内访问可走 [hf-mirror.com](https://hf-mirror.com)：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset flufy3d/xinhe-dataset --local-dir data
```

## 许可证 & 引用

CC-BY-4.0。由 [@flufy3d](https://github.com/flufy3d) 维护，学术使用请引用 Xinhe 项目。
"""


# ── 扫描 + 生成 ───────────────────────────────────────────────

def discover_configs(data_root: Path):
    """扫 data/<kind>/<split>.jsonl, 聚合成 {kind: [(split, rel_path), ...]}。

    跳过 *.rejected.jsonl(generator 的废弃样本不上传)。
    """
    configs = defaultdict(list)
    skipped = []
    for jsonl in sorted(data_root.rglob("*.jsonl")):
        if jsonl.name.endswith(".rejected.jsonl"):
            continue
        parts = jsonl.relative_to(data_root).parts
        if len(parts) != 2:
            skipped.append(jsonl.relative_to(PROJECT_ROOT).as_posix())
            continue
        kind, fname = parts
        split_raw = Path(fname).stem
        split = SPLIT_NAME_REMAP.get(split_raw, split_raw)
        rel = jsonl.relative_to(PROJECT_ROOT).as_posix()
        configs[kind].append((split, rel))
    if skipped:
        print(f"⚠ 跳过 {len(skipped)} 个不符合 data/<kind>/<split>.jsonl 模式的文件:")
        for p in skipped:
            print(f"    {p}")
    return configs


def build_yaml(configs) -> str:
    lines = [
        "---",
        "license: cc-by-4.0",
        "language:",
        "- zh",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- memory",
        "- persona",
        "- dialogue",
        "- chinese",
        "- synthetic",
        "- xinhe",
        "pretty_name: Xinhe Persona Memory Dataset",
        "size_categories:",
        "- 10K<n<100K",
        "configs:",
    ]
    for kind, splits in sorted(configs.items()):
        lines.append(f"  - config_name: {kind}")
        lines.append(f"    data_files:")
        for split, path in sorted(splits):
            lines.append(f"      - split: {split}")
            lines.append(f"        path: {path}")
    lines.append("---")
    return "\n".join(lines)


def build_readme(configs) -> str:
    return build_yaml(configs) + "\n\n" + README_BODY


# ── 登录 ──────────────────────────────────────────────────────

def ensure_login():
    if get_token() is not None:
        return
    print("未检测到 HuggingFace token，启动交互登录...")
    print("提示: 访问 https://huggingface.co/settings/tokens 创建 Write 类型 token。")
    # huggingface_hub.login() 不带参数会交互式 prompt 输入 token，
    # 跨平台不依赖 huggingface-cli 在 PATH 中
    login()
    if get_token() is None:
        print("错误: 登录后仍无法获取 token")
        sys.exit(1)


# ── 主流程 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="同步本地 data/ → HuggingFace 数据集仓库（增删全同步）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id（默认 {DEFAULT_REPO}）")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"数据目录（项目相对路径，默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要上传/删除的清单与 README")
    args = parser.parse_args()

    data_root_rel = args.data_root.replace("\\", "/").rstrip("/")
    data_root = (PROJECT_ROOT / data_root_rel).resolve()
    if not data_root.exists():
        print(f"错误: 数据目录不存在 {data_root}")
        sys.exit(1)

    print(f"扫描 {data_root_rel}/ ...")
    configs = discover_configs(data_root)
    if not configs:
        print("错误: 未发现任何符合规范的 jsonl 文件")
        sys.exit(1)

    print(f"\n发现 {len(configs)} 个 config:")
    local_paths = set()
    for kind, splits in sorted(configs.items()):
        print(f"  {kind}:")
        for split, path in sorted(splits):
            size_mb = (PROJECT_ROOT / path).stat().st_size / 1024 / 1024
            print(f"    {split:14s}  {path}  ({size_mb:.1f} MB)")
            local_paths.add(path)
    print(f"共 {len(local_paths)} 个 jsonl 文件")

    readme_text = build_readme(configs)

    if args.dry_run:
        print("\n--- 生成的 README.md ---")
        print(readme_text)
        print("--- (end) ---")
        print(f"\n[dry-run] 不会实际上传到 {args.repo}")
        return

    ensure_login()

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)

    # 取远端文件清单, 计算需要删除的 jsonl
    try:
        remote_files = set(api.list_repo_files(repo_id=args.repo, repo_type="dataset"))
    except RepositoryNotFoundError:
        remote_files = set()

    to_delete = sorted(
        f for f in remote_files
        if f.startswith(f"{data_root_rel}/")
        and f.endswith(".jsonl")
        and f not in local_paths
    )

    if to_delete:
        print(f"\n将删除 {len(to_delete)} 个远端 jsonl（本地已不存在）:")
        for p in to_delete:
            print(f"    {p}")

    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_text.encode("utf-8")),
    ]
    for path in sorted(local_paths):
        operations.append(CommitOperationAdd(
            path_in_repo=path,
            path_or_fileobj=str(PROJECT_ROOT / path),
        ))
    for path in to_delete:
        operations.append(CommitOperationDelete(path_in_repo=path))

    print(f"\n推送到 {args.repo} ({len(local_paths)} adds, {len(to_delete)} deletes)...")
    commit_info = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Sync data/: {len(local_paths)} jsonls + README",
    )
    print(f"完成: {commit_info.commit_url}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
