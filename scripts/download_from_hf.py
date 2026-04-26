#!/usr/bin/env python3
"""
HuggingFace 数据集仓库 → 本地 data/ 的幂等下载工具。

用法:
    uv run python scripts/download_from_hf.py             # 拉取全部 data/**
    uv run python scripts/download_from_hf.py --repo X    # 换仓库

环境变量:
    HF_ENDPOINT (可选): 国内服务器请 export HF_ENDPOINT=https://hf-mirror.com
                       本地通常不需要设置, snapshot_download 自动读取此变量。

幂等性:
    snapshot_download 默认按 hash 增量下载, 已存在且未变的文件自动跳过。
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "flufy3d/xinhe-dataset"


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 同步数据集仓库到本地 data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id（默认 {DEFAULT_REPO}）")
    parser.add_argument("--data-root", default="data",
                        help="本地数据目录（项目相对路径，默认 data）")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"仓库: {args.repo}")
    print(f"端点: {endpoint}")
    print(f"目标: {PROJECT_ROOT / args.data_root}")

    # local_dir = PROJECT_ROOT, allow_patterns="data/**"
    # → 远端 data/v8/... 落到本地 data/v8/..., 跟本地原目录一一对齐
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(PROJECT_ROOT),
        allow_patterns=[f"{args.data_root}/**"],
    )

    print("同步完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
