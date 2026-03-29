#!/usr/bin/env python3
"""
部署脚本 - 同步代码和模型权重到远端 GPU 服务器

用法:
    python deploy.py "ssh -p 33269 root@cn-north-b.ssh.damodel.com"
    python deploy.py root@cn-north-b.ssh.damodel.com 33269
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REMOTE_DIR = "/root/workspace/xinhe"
SCRIPT_DIR = Path(__file__).parent.parent


def run(cmd: list[str], check=True):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def ssh_cmd(host, port):
    return ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no", host]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="SSH 地址，如 'ssh -p 33269 root@host' 或 'root@host'")
    parser.add_argument("port", nargs="?", default="22")
    args = parser.parse_args()

    # 解析 host / port
    m = re.search(r"-p\s+(\d+)\s+(\S+)", args.target)
    if m:
        port, host = m.group(1), m.group(2)
    else:
        host = args.target
        port = args.port

    print("=" * 54)
    print(f"  目标: {host}:{port}")
    print(f"  远端: {REMOTE_DIR}")
    print("=" * 54)

    # ── 1. 同步代码 ─────────────────────────────────────────
    print("\n[1/3] 同步代码...")
    run(ssh_cmd(host, port) + [f"mkdir -p {REMOTE_DIR}"])
    tar_proc = subprocess.Popen(
        ["tar", "-czf", "-",
         "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.pyo",
         "xinhe", "scripts", "configs", "tests", "pyproject.toml", "uv.lock"],
        stdout=subprocess.PIPE,
        cwd=str(SCRIPT_DIR),
    )
    ssh_proc = subprocess.Popen(
        ssh_cmd(host, port) + [f"tar -xzf - -C {REMOTE_DIR}"],
        stdin=tar_proc.stdout,
    )
    tar_proc.stdout.close()
    ssh_proc.wait()
    tar_proc.wait()

    # ── 2. 同步模型权重 ─────────────────────────────────────
    print("\n[2/3] 同步模型权重...")
    run(ssh_cmd(host, port) + [
        f"mkdir -p {REMOTE_DIR}/models && "
        f"ln -sfn /root/public-storage/model/Qwen/Qwen3-0.6B {REMOTE_DIR}/models/qwen3-0.6b"
    ])
    print("  qwen3-0.6b -> 软链接自 public-storage")

    # ── 3. 远端：uv sync + 生成数据 ─────────────────────────
    print("\n[3/3] 远端初始化...")
    init_script = f"""
set -e
cd {REMOTE_DIR}
echo '--- 安装依赖 ---'
uv sync


"""
    run(ssh_cmd(host, port) + [init_script])

    print("\n" + "=" * 54)
    print("  部署完成！ssh 进去开始训练：")
    print(f"  ssh -p {port} {host}")
    print(f"  cd {REMOTE_DIR} && uv run python scripts/train.py")
    print("=" * 54)


if __name__ == "__main__":
    main()
