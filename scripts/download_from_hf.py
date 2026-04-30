#!/usr/bin/env python3
"""
HuggingFace 数据集仓库 → 本地 data/ 的幂等下载工具。

用法:
    uv run python scripts/download_from_hf.py             # 拉全部 data/**
    uv run python scripts/download_from_hf.py --repo X    # 换仓库
    uv run python scripts/download_from_hf.py --clean-stale  # 顺便删本地多余 jsonl

环境变量:
    HF_ENDPOINT (可选): 国内服务器 export HF_ENDPOINT=https://hf-mirror.com
                       本地通常不设, snapshot_download 自动读取此变量。
                       注意:hf-mirror 同步主站有延迟,新上传的文件可能要等几分钟。

幂等性:
    snapshot_download 默认按 hash 增量, 已存在且未变的文件跳过。
    本脚本额外打印远端 vs 本地 jsonl 清单, stale 文件(本地有/远端无) 警告或清理。
"""
import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "flufy3d/xinhe-dataset"


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 同步数据集仓库到本地 data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id(默认 {DEFAULT_REPO})")
    parser.add_argument("--data-root", default="data",
                        help="本地数据目录(项目相对路径,默认 data)")
    parser.add_argument("--clean-stale", action="store_true",
                        help="删除本地 data_root 下、远端不存在的 jsonl(默认仅警告)")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    data_root_rel = args.data_root.replace("\\", "/").rstrip("/")
    data_root = (PROJECT_ROOT / data_root_rel).resolve()
    print(f"仓库:   {args.repo}")
    print(f"端点:   {endpoint}")
    print(f"目标:   {data_root}")

    api = HfApi(endpoint=endpoint)
    remote_all = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
    remote_data = sorted(
        f for f in remote_all
        if f.startswith(f"{data_root_rel}/") and f.endswith(".jsonl")
    )
    print(f"\n远端 jsonl ({len(remote_data)} 个):")
    for f in remote_data:
        print(f"  {f}")
    if not remote_data:
        print("  (空 — 如果用 hf-mirror,可能是镜像同步延迟,几分钟后重试或换主站 HF_ENDPOINT=https://huggingface.co)")

    # 用权威清单当 allow_patterns 白名单,而不是 data/** 通配。
    # 否则 mirror 的 git-tree 滞后老 commit 时,snapshot_download 会把已删除的
    # 历史路径(如 data/v8/*)一并镜像出来,污染本地。
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(PROJECT_ROOT),
        allow_patterns=remote_data,
    )

    # 下载后扫本地实际清单
    if data_root.exists():
        local_jsonls = sorted(
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for p in data_root.rglob("*.jsonl")
        )
    else:
        local_jsonls = []
    print(f"\n本地 jsonl ({len(local_jsonls)} 个):")
    for f in local_jsonls:
        size_mb = (PROJECT_ROOT / f).stat().st_size / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")

    # 找 stale:本地有 + 远端无
    stale = sorted(set(local_jsonls) - set(remote_data))
    if stale:
        print(f"\n⚠ 本地多余 {len(stale)} 个 jsonl(远端已不存在):")
        for f in stale:
            print(f"  {f}")
        if args.clean_stale:
            for f in stale:
                (PROJECT_ROOT / f).unlink()
                # 顺便清理空目录
                parent = (PROJECT_ROOT / f).parent
                while parent != data_root and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            print(f"已清理 {len(stale)} 个 stale jsonl + 空目录")
        else:
            print("加 --clean-stale 自动删除,或手动 rm。")

    # 找 missing:远端有 + 本地无(snapshot_download 漏拉)
    # 兜底:走裸 resolve URL 直接 HTTP 下载。绕开 hf-mirror 的 git-tree/resolve-cache
    # 同步滞后(snapshot_download 走 git-tree,hf_hub_download 走 resolve-cache,
    # 两者有 lag 时返回老 commit / 500;裸 resolve/main/<file> 通常立即反映 HEAD)。
    missing = sorted(set(remote_data) - set(local_jsonls))
    if missing:
        print(f"\n⚠ 远端有但本地缺 {len(missing)} 个 jsonl,启用裸 resolve URL 兜底:")
        recovered = _resolve_url_fallback(missing, args.repo, endpoint)
        still_missing = [f for f in missing if f not in recovered]
        if still_missing:
            print(f"\n✗ 兜底后仍缺 {len(still_missing)} 个:")
            for f in still_missing:
                print(f"  {f}")
            print("如果用 hf-mirror,试 HF_ENDPOINT=https://huggingface.co 走主站。")
            sys.exit(2)

    print("\n同步完成。")


def _resolve_url_fallback(missing: list[str], repo: str, endpoint: str) -> list[str]:
    """对 snapshot_download 漏拉的文件,走 {endpoint}/datasets/{repo}/resolve/main/<path>
    直接 HTTP 下载。返回成功下载的路径列表。"""
    recovered = []
    for path in missing:
        url = f"{endpoint}/datasets/{repo}/resolve/main/{path}"
        out = PROJECT_ROOT / path
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(out, "wb") as fp:
                downloaded = 0
                while True:
                    chunk = resp.read(1 << 20)    # 1 MB
                    if not chunk:
                        break
                    fp.write(chunk)
                    downloaded += len(chunk)
            sz_mb = downloaded / 1024 / 1024
            print(f"  ✓ {path}  ({sz_mb:.1f} MB)")
            recovered.append(path)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"  ✗ {path}  ({type(e).__name__}: {e})")
            if out.exists() and out.stat().st_size == 0:
                out.unlink()
    return recovered


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
