#!/usr/bin/env python3
"""
HuggingFace 数据集仓库 → 本地 data/ 的幂等下载工具。

用法:
    uv run python scripts/download_from_hf.py             # 拉全部 data/**.jsonl
    uv run python scripts/download_from_hf.py --repo X    # 换仓库
    uv run python scripts/download_from_hf.py --clean-stale  # 顺便删本地多余 jsonl
    uv run python scripts/download_from_hf.py --resolve-paths \\
        data/novel/train.jsonl data/novel/val.jsonl       # 跳 list, 直接拉指定文件

实现:
    完全走 resolve URL + urllib,不用 huggingface_hub.snapshot_download。
    根因(2026-05):snapshot_download 的 TCP read 无 read-timeout,hf-mirror
    上 keep-alive 被 NAT/防火墙黑洞后,Python 永久卡死(实测三个连接精确停
    在 178257920 字节,6+ 分钟零增长,State=S sleeping)。
    urllib.request.urlopen(timeout=60) 是 socket-wide timeout,read 卡 60s
    即抛 socket.timeout,可重试。

幂等性:
    - 目标文件已存在 + size 与远端一致 → 跳过
    - 目标存在但 size 不一致 → 重下
    - .partial 存在 → HTTP Range 续传(不浪费已下载部分)
    - server 返回 200 而非 206(不认 Range)→ 从头重下

环境变量:
    HF_ENDPOINT (可选): 国内服务器 export HF_ENDPOINT=https://hf-mirror.com
                       默认走主站 https://huggingface.co
                       同时影响"下载"和"list"(后者通过 HF_LIST_ENDPOINT fallback)。
    HF_LIST_ENDPOINT (可选): 列文件用此端点,默认跟随 HF_ENDPOINT。
                       想绕 mirror 的 git-tree 同步滞后(新 commit 漏列)时,
                       export HF_LIST_ENDPOINT=https://huggingface.co 走主站。
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "flufy3d/xinhe-dataset"
UA = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 60          # socket-wide,read/connect 都生效
MAX_RETRIES = 5
RETRY_SLEEP = 2
CHUNK = 1 << 20       # 1 MB


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
    parser.add_argument("--resolve-paths", nargs="+", default=None, metavar="PATH",
                        help="指定路径直接拉,跳过 list_repo_tree(应急用,无 size 校验)")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    list_endpoint = os.environ.get("HF_LIST_ENDPOINT", endpoint)

    # 应急模式:跳过 list,直接拉指定路径(size=None 表示不校验)
    if args.resolve_paths:
        paths = sorted(set(p.replace("\\", "/") for p in args.resolve_paths))
        print(f"仓库:        {args.repo}")
        print(f"端点(下载):  {endpoint}")
        print(f"模式:        --resolve-paths(跳过 list,无 size 校验)")
        print(f"\n直接拉 {len(paths)} 个文件:")
        failed = [p for p in paths if not _download_one(p, None, args.repo, endpoint)]
        if failed:
            print(f"\n✗ 失败 {len(failed)} 个:")
            for p in failed:
                print(f"  {p}")
            sys.exit(2)
        print("\n下载完成。")
        return

    data_root_rel = args.data_root.replace("\\", "/").rstrip("/")
    data_root = (PROJECT_ROOT / data_root_rel).resolve()
    print(f"仓库:        {args.repo}")
    print(f"端点(list):  {list_endpoint}")
    print(f"端点(下载):  {endpoint}")
    print(f"目标:        {data_root}")

    # 1. 列远端 jsonl(含 size,作为完整性 ground truth)
    remote = _list_remote(args.repo, list_endpoint, data_root_rel)
    remote_paths = sorted(remote.keys())
    total_mb = sum(remote.values()) / 1024 / 1024
    print(f"\n远端 jsonl ({len(remote_paths)} 个, 共 {total_mb:.0f} MB):")
    for p in remote_paths:
        print(f"  {p}  ({remote[p]/1024/1024:.1f} MB)")
    if not remote:
        print("  (空 — 如果用 hf-mirror,可能是镜像同步延迟,"
              "几分钟后重试或换主站 HF_LIST_ENDPOINT=https://huggingface.co)")
        return

    # 2. 逐文件下载(带续传 + 重试)
    print(f"\n开始同步:")
    failed = [p for p in remote_paths
              if not _download_one(p, remote[p], args.repo, endpoint)]

    # 3. 扫本地,清 stale
    if data_root.exists():
        local_jsonls = sorted(
            str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for p in data_root.rglob("*.jsonl")
        )
    else:
        local_jsonls = []

    stale = sorted(set(local_jsonls) - set(remote_paths))
    if stale:
        print(f"\n⚠ 本地多余 {len(stale)} 个 jsonl(远端已不存在):")
        for f in stale:
            print(f"  {f}")
        if args.clean_stale:
            for f in stale:
                (PROJECT_ROOT / f).unlink()
                parent = (PROJECT_ROOT / f).parent
                while parent != data_root and parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
            print(f"已清理 {len(stale)} 个 stale jsonl + 空目录")
        else:
            print("加 --clean-stale 自动删除。")

    if failed:
        print(f"\n✗ 失败 {len(failed)} 个(已重试 {MAX_RETRIES} 次):")
        for p in failed:
            print(f"  {p}")
        sys.exit(2)

    print("\n同步完成。")


def _list_remote(repo: str, endpoint: str, data_root_rel: str) -> dict[str, int]:
    """走 /api/datasets/{repo}/tree/main?recursive=true 列远端 jsonl + size。

    用裸 urllib 而非 huggingface_hub.HfApi,避开 hub 的 httpx 同样可能的卡死。
    list 是单次小 JSON 响应,60s timeout 足够。

    返回 {path: size} dict,只含 data_root_rel/ 下的 .jsonl。
    size 优先取 lfs.size(LFS 真实大小),fallback top-level size。
    """
    url = f"{endpoint}/api/datasets/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                entries = json.load(resp)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            print(f"  ! list {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_SLEEP)

    result = {}
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path", "")
        if not (path.startswith(f"{data_root_rel}/") and path.endswith(".jsonl")):
            continue
        size = (e.get("lfs") or {}).get("size") or e.get("size")
        if size is None:
            continue
        result[path] = int(size)
    return result


def _download_one(path: str, expected_size: int | None,
                  repo: str, endpoint: str) -> bool:
    """带续传 + 重试地下载单文件。返回是否成功。

    幂等:目标 size 一致即跳过;.partial 走 Range 续传;server 不认 Range 则重头。
    每次失败 sleep RETRY_SLEEP,最多 MAX_RETRIES 次。
    """
    url = f"{endpoint}/datasets/{repo}/resolve/main/{path}"
    out = PROJECT_ROOT / path
    partial = out.parent / (out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and expected_size is not None:
        if out.stat().st_size == expected_size:
            print(f"  = {path}  (本地已完整 {expected_size/1024/1024:.1f} MB)")
            return True
        print(f"  ! {path}  本地 {out.stat().st_size} != 远端 {expected_size},重下")
        out.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        start = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and start > expected_size:
            partial.unlink()
            start = 0

        headers = dict(UA)
        if start > 0:
            headers["Range"] = f"bytes={start}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                # server 不认 Range:返回 200 整文件 → 推倒重来
                if start > 0 and resp.status == 200:
                    partial.unlink()
                    start = 0
                    mode = "wb"
                else:
                    mode = "ab" if start > 0 else "wb"
                with open(partial, mode) as fp:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fp.write(chunk)

            actual = partial.stat().st_size
            if expected_size is not None and actual != expected_size:
                # 服务端提前关流;留 .partial 给下次续传
                print(f"  ! {path}  size 不符 {actual} != {expected_size},续传重试")
                continue
            if out.exists():
                out.unlink()
            partial.rename(out)
            print(f"  ✓ {path}  ({actual/1024/1024:.1f} MB)")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            sz_mb = (partial.stat().st_size if partial.exists() else 0) / 1024 / 1024
            print(f"  ✗ {path}  attempt {attempt}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e} (已下 {sz_mb:.1f} MB, {RETRY_SLEEP}s 重试)")
            time.sleep(RETRY_SLEEP)

    return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
