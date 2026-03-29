#!/usr/bin/env bash
# 用法: ./deploy.sh "ssh -p 33269 root@cn-north-b.ssh.damodel.com"
# 或者:  ./deploy.sh root@cn-north-b.ssh.damodel.com 33269
set -e

# ── 解析参数 ─────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  echo "用法: $0 \"ssh -p PORT user@host\""
  echo "  或: $0 user@host PORT"
  exit 1
fi

if [[ "$1" == ssh* ]]; then
  # 解析 "ssh -p 33269 root@host" 格式
  SSH_ARGS=($1)
  PORT=$(echo "$1" | grep -oP '(?<=-p )\d+')
  HOST=$(echo "$1" | awk '{print $NF}')
else
  HOST="$1"
  PORT="${2:-22}"
fi

SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh -p $PORT -o StrictHostKeyChecking=no"

REPO_URL=$(git -C "$(dirname "$0")" remote get-url origin 2>/dev/null || echo "")
REMOTE_DIR="/root/xinhe"

echo "======================================================"
echo "  目标: $HOST:$PORT"
echo "  仓库: $REPO_URL"
echo "  远端路径: $REMOTE_DIR"
echo "======================================================"

# ── 1. clone / pull 代码 ──────────────────────────────────
echo ""
echo "[1/3] 同步代码..."
if [[ -n "$REPO_URL" ]]; then
  $SSH "$HOST" bash <<EOF
    if [ -d "$REMOTE_DIR/.git" ]; then
      echo "  已有仓库，pull 最新..."
      cd $REMOTE_DIR && git pull --rebase
    else
      echo "  克隆仓库..."
      git clone $REPO_URL $REMOTE_DIR
    fi
EOF
else
  echo "  [!] 未检测到 git remote，改用 rsync 同步代码"
  rsync -avz --progress \
    -e "$RSYNC_SSH" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='.venv' --exclude='models/' --exclude='checkpoints/' \
    --exclude='data/' --exclude='wandb/' \
    "$(dirname "$0")/" \
    "$HOST:$REMOTE_DIR/"
fi

# ── 2. 同步模型权重 ──────────────────────────────────────
echo ""
echo "[2/3] 同步 backbone 模型 (~1.5G，首次较慢)..."
rsync -avz --progress \
  -e "$RSYNC_SSH" \
  "$(dirname "$0")/models/" \
  "$HOST:$REMOTE_DIR/models/"

# ── 3. 远端：装依赖 → 生成数据 → 开始训练 ───────────────
echo ""
echo "[3/3] 远端初始化..."

$SSH "$HOST" bash <<EOF
  set -e
  cd $REMOTE_DIR

  echo "--- 安装依赖 ---"
  uv sync

  echo "--- 生成训练数据 ---"
  uv run python -X utf8 scripts/generate_memory_data.py --num-train 5000 --num-val 500 --max-turns 14 --min-distance 1 --max-distance 10
EOF

echo ""
echo "======================================================"
echo "  部署完成！可以开始训练："
echo "  $SSH $HOST"
echo "  cd $REMOTE_DIR && uv run python scripts/train.py"
echo "======================================================"
