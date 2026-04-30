#!/usr/bin/env python3
"""
远程服务器管理工具

用法:
    python scripts/remote.py deploy                 # 幂等部署
    python scripts/remote.py upload  path [path..]  # 上传文件/目录
    python scripts/remote.py download path [path..] # 下载文件/目录
    python scripts/remote.py ssh                    # SSH 连入服务器
    python scripts/remote.py run "cmd"              # 远端执行命令
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 配置 ────────────────────────────────────────────────────

def load_env(env_file=".env"):
    """从指定 env 文件读取配置，返回 dict。"""
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        print(f"错误: 找不到配置文件 {env_path}")
        print(f"请复制 .env.example 为 {env_path.name} 并填写实际值")
        sys.exit(1)
    cfg = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        cfg[key.strip()] = value.strip()
    return cfg


def get_config(env_file=".env"):
    """返回 (host, port, remote_dir, models) 配置。"""
    cfg = load_env(env_file)
    host = cfg.get("REMOTE_HOST")
    port = cfg.get("REMOTE_PORT", "22")
    remote_dir = cfg.get("REMOTE_DIR", "/root/workspace/xinhe")
    models = {}
    for k, v in cfg.items():
        if k.startswith("MODEL_"):
            # 格式: MODEL_xx=链接名:目标路径
            if ":" in v:
                name, target = v.split(":", 1)
                models[name] = target
            else:
                print(f"警告: {k} 格式应为 '链接名:目标路径'，跳过")

    if not host:
        print("错误: .env 中缺少 REMOTE_HOST")
        sys.exit(1)
    return host, port, remote_dir, models


# ── SSH 工具 ────────────────────────────────────────────────

def ssh_opts():
    return ["-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR"]


def ssh_base(host, port):
    """非交互式 SSH 命令基础参数（-T 禁用伪终端）"""
    return ["ssh", "-T", "-p", port] + ssh_opts() + [host]


def run(cmd, check=True, verbose=False, **kwargs):
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(cmd, check=check, **kwargs)


# ── deploy ──────────────────────────────────────────────────

def cmd_deploy(args):
    host, port, remote_dir, models = get_config(args.env)

    print("=" * 54)
    print(f"  目标: {host}:{port}")
    print(f"  远端: {remote_dir}")
    print("=" * 54)

    # 1. 同步代码 + 词典/语料（xinhe/data/dicts/files/*.txt + *.jsonl 随 xinhe 一起带上）。
    #    训练集 data/{skeleton,dialog,mix}/ 通过步骤 [4/4] 从 HuggingFace 拉取(见 scripts/download_from_hf.py)。
    print("\n[1/4] 同步代码...")
    run(ssh_base(host, port) + [f"mkdir -p {remote_dir}"])
    tar_paths = [
        "xinhe", "scripts", "configs", "tests", "pyproject.toml", "uv.lock",
    ]
    tar_proc = subprocess.Popen(
        ["tar", "-czf", "-",
         "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.pyo",
         *tar_paths],
        stdout=subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )
    ssh_proc = subprocess.Popen(
        ssh_base(host, port) + [f"tar -xzf - -C {remote_dir}"],
        stdin=tar_proc.stdout,
    )
    tar_proc.stdout.close()
    ssh_proc.wait()
    tar_proc.wait()

    # 2. 模型软链接 + 指纹校验
    print("\n[2/4] 模型链接...")
    if models:
        fp_dir = f"{remote_dir}/models/.fingerprints"
        link_cmds = [f"mkdir -p {remote_dir}/models {fp_dir}"]
        for name, target in models.items():
            link = f"{remote_dir}/models/{name}"
            fp_file = f"{fp_dir}/{name}.sha256"
            # 创建/更新软链接
            link_cmds.append(f'ln -sfn {target} {link}')
            # 计算指纹: config.json sha256 + 权重文件总大小
            link_cmds.append(
                f'FP=$(sha256sum {target}/config.json 2>/dev/null | cut -d" " -f1)'
                f'_$(du -sb {target}/*.safetensors 2>/dev/null | awk "{{s+=\\$1}}END{{print s}}")'
                f' && if test -f {fp_file}; then'
                f'   OLD=$(cat {fp_file});'
                f'   if [ "$FP" != "$OLD" ]; then'
                f'     echo "  ⚠ {name} 指纹变化! 权重可能被更新";'
                f'     echo "    旧: $OLD";'
                f'     echo "    新: $FP";'
                f'   else'
                f'     echo "  {name} -> 指纹一致 ✓";'
                f'   fi;'
                f' else'
                f'   echo "  {name} -> 首次记录指纹";'
                f' fi'
                f' && echo "$FP" > {fp_file}'
            )
        run(ssh_base(host, port) + [" && ".join(link_cmds)])
    else:
        print("  无模型配置，跳过")

    # 3. 远端初始化（幂等）
    print("\n[3/4] 远端依赖...")
    init_script = f"""set -e
cd {remote_dir}
export PATH="/usr/local/cuda/bin:/opt/conda/bin:$HOME/.local/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export UV_CACHE_DIR="/root/shared-storage/.uv-cache"
export HF_ENDPOINT="https://hf-mirror.com"
# 持久化环境变量到 bashrc (SSH 登录也生效)
grep -q UV_CACHE_DIR ~/.bashrc 2>/dev/null || cat >> ~/.bashrc << 'ENVEOF'
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"
export UV_CACHE_DIR="/root/shared-storage/.uv-cache"
export HF_ENDPOINT="https://hf-mirror.com"
ENVEOF
# 已有 bashrc 块但缺 HF_ENDPOINT 时补一行（兼容旧机器）
grep -q HF_ENDPOINT ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
if command -v uv >/dev/null 2>&1; then
  echo '  uv 已安装，跳过'
else
  echo '  安装 uv...'
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo '  uv sync...'
uv sync
# causal-conv1d 需要 --no-build-isolation 用 venv 的 torch 编译
if ! uv pip show causal-conv1d >/dev/null 2>&1; then
  echo '  安装 causal-conv1d (首次编译，需几分钟)...'
  uv pip install causal-conv1d --no-build-isolation
fi
"""
    run(ssh_base(host, port) + [init_script])

    # 4. 同步训练数据（从 HuggingFace 经 hf-mirror 镜像拉取）
    print("\n[4/4] 同步训练数据...")
    sync_script = f"""set -e
cd {remote_dir}
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="https://hf-mirror.com"
uv run python scripts/download_from_hf.py
"""
    run(ssh_base(host, port) + [sync_script])

    print("\n" + "=" * 54)
    print("  部署完成！")
    print(f"  ssh -p {port} {host}")
    print(f"  cd {remote_dir} && uv run python scripts/train.py")
    print("=" * 54)


# ── upload ──────────────────────────────────────────────────

def _normalize(path_str):
    """规范化路径：反斜杠转正斜杠，去掉 ./ 前缀"""
    return Path(path_str).as_posix()


def cmd_upload(args):
    host, port, remote_dir, _ = get_config(args.env)

    for raw in args.paths:
        path_str = _normalize(raw)
        local_path = PROJECT_ROOT / path_str
        if not local_path.exists():
            print(f"错误: 本地路径不存在 {local_path}")
            sys.exit(1)

        remote_path = f"{remote_dir}/{path_str}"
        parent = str(Path(path_str).parent).replace("\\", "/")
        if parent != ".":
            run(ssh_base(host, port) + [f"mkdir -p {remote_dir}/{parent}"])

        if local_path.is_file():
            print(f"上传文件: {path_str}")
            run(["scp", "-P", port] + ssh_opts() +
                [str(local_path), f"{host}:{remote_path}"])
        else:
            print(f"上传目录: {path_str}")
            run(ssh_base(host, port) + [f"mkdir -p {remote_path}"])
            tar_proc = subprocess.Popen(
                ["tar", "-czf", "-", "-C", str(PROJECT_ROOT), path_str],
                stdout=subprocess.PIPE,
            )
            ssh_proc = subprocess.Popen(
                ssh_base(host, port) + [f"tar -xzf - -C {remote_dir}"],
                stdin=tar_proc.stdout,
            )
            tar_proc.stdout.close()
            ssh_proc.wait()
            tar_proc.wait()

    print("上传完成")


# ── download ────────────────────────────────────────────────

def cmd_download(args):
    host, port, remote_dir, _ = get_config(args.env)

    for raw in args.paths:
        path_str = _normalize(raw)
        local_path = PROJECT_ROOT / path_str
        parent = local_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            print(f"  创建本地目录: {parent}")

        print(f"下载: {path_str}")
        ssh_proc = subprocess.Popen(
            ssh_base(host, port) + [f"tar -czf - -C {remote_dir} {path_str}"],
            stdout=subprocess.PIPE,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-xzf", "-", "-C", str(PROJECT_ROOT)],
            stdin=ssh_proc.stdout,
        )
        ssh_proc.stdout.close()
        tar_proc.wait()
        ssh_proc.wait()

    print("下载完成")


# ── ssh ─────────────────────────────────────────────────────

def cmd_ssh(args):
    host, port, _, _ = get_config(args.env)
    result = subprocess.run(["ssh", "-p", port] + ssh_opts() + [host])
    sys.exit(result.returncode)


# ── run ─────────────────────────────────────────────────────

def cmd_run(args):
    host, port, remote_dir, _ = get_config(args.env)
    command = " ".join(args.command).replace("\\", "/")
    result = run(
        ssh_base(host, port) + [f"cd {remote_dir} && {command}"],
        check=False, verbose=False,
    )
    sys.exit(result.returncode)


# ── start ───────────────────────────────────────────────────

PID_FILE = "xinhe.pid"
LOG_FILE = "train.log"


def cmd_start(args):
    host, port, remote_dir, _ = get_config(args.env)
    command = " ".join(args.command).replace("\\", "/")

    # 检查是否已有进程在跑
    check = subprocess.run(
        ssh_base(host, port) + [
            f"test -f {remote_dir}/{PID_FILE} && "
            f"kill -0 $(cat {remote_dir}/{PID_FILE}) 2>/dev/null && echo running"
        ],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if "running" in check.stdout:
        print(f"错误: 已有任务在运行 (pid: 见 {PID_FILE})")
        print(f"  查看: python scripts/remote.py logs")
        print(f"  停止: python scripts/remote.py stop")
        sys.exit(1)

    # 自动加 uv run 前缀，-u 禁用 stdout 缓冲以便实时看日志
    if not command.startswith("uv run "):
        command = f"uv run {command}"
    command = command.replace("python ", "python -u ", 1)

    # nohup 后台运行，pid 写入文件
    # bash -c 确保 & 和 $! 正确解析，disown 让进程脱离 SSH session
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 减显存碎片（v6 双流 + torch.compile 切阶段瞬时峰值）
    start_cmd = (
        f"cd {remote_dir} && "
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        f"bash -c 'nohup {command} > {LOG_FILE} 2>&1 & "
        f"echo $! > {PID_FILE} && disown'"
    )
    run(ssh_base(host, port) + [start_cmd])
    print(f"已后台启动，日志: {LOG_FILE}")
    print(f"  实时日志: python scripts/remote.py logs -f")
    print(f"  停止训练: python scripts/remote.py stop")


# ── logs ────────────────────────────────────────────────────

def cmd_logs(args):
    host, port, remote_dir, _ = get_config(args.env)
    log_path = f"{remote_dir}/{LOG_FILE}"

    if args.follow:
        # tail -f，Ctrl+C 退出不影响训练
        try:
            subprocess.run(["ssh", "-p", port] + ssh_opts() + [host, f"tail -n {args.lines} -f {log_path}"])
        except KeyboardInterrupt:
            pass
        return
    else:
        result = run(
            ssh_base(host, port) + [f"tail -n {args.lines} {log_path}"],
            check=False,
        )
        sys.exit(result.returncode)


# ── stop ────────────────────────────────────────────────────

def cmd_stop(args):
    host, port, remote_dir, _ = get_config(args.env)
    pid_path = f"{remote_dir}/{PID_FILE}"

    # 读取 pid 并检查是否存活
    result = subprocess.run(
        ssh_base(host, port) + [
            f"test -f {pid_path} && cat {pid_path} || echo none"
        ],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    pid = result.stdout.strip()
    if pid == "none":
        print("没有运行中的任务")
        return

    run(ssh_base(host, port) + [
        f"kill {pid} 2>/dev/null && echo '已停止 (pid: {pid})' || echo '进程已结束'; "
        f"rm -f {pid_path}"
    ])


# ── 入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="远程服务器管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="配置: 项目根目录 .env 文件（参考 .env.example）",
    )
    parser.add_argument("--env", default=".env",
                        help="配置文件路径（默认 .env，可用 .env.4b 等切换多台远端）")
    sub = parser.add_subparsers(dest="cmd", title="子命令")

    sub.add_parser("deploy", help="幂等部署（同步代码、模型链接、依赖）").set_defaults(func=cmd_deploy)

    p_up = sub.add_parser("upload", help="上传文件/目录到远端（镜像路径映射）")
    p_up.add_argument("paths", nargs="+", help="本地相对路径")
    p_up.set_defaults(func=cmd_upload)

    p_down = sub.add_parser("download", help="从远端下载文件/目录到本地")
    p_down.add_argument("paths", nargs="+", help="远端相对路径")
    p_down.set_defaults(func=cmd_download)

    sub.add_parser("ssh", help="SSH 连入远端服务器").set_defaults(func=cmd_ssh)

    p_run = sub.add_parser("run", help="在远端执行命令")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令")
    p_run.set_defaults(func=cmd_run)

    p_start = sub.add_parser("start", help="后台启动命令（自动 uv run）")
    p_start.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令（自动加 uv run 前缀）")
    p_start.set_defaults(func=cmd_start)

    p_logs = sub.add_parser("logs", help="查看训练日志")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="显示最近 N 行（默认 50）")
    p_logs.add_argument("-f", "--follow", action="store_true", help="实时跟踪日志（Ctrl+C 退出）")
    p_logs.set_defaults(func=cmd_logs)

    sub.add_parser("stop", help="停止后台任务").set_defaults(func=cmd_stop)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
