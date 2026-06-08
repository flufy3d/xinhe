#!/usr/bin/env bash
# v21 = 从 v20 step_14000 resume,horizon 拉到 40000(LR 自动回升,无 --reset-step)。
# 单变量测「持续 LR 下多 visits 能否把 val read 顶过 v20 的 ~11%」。save_every=1000 中途停评。
set -e
cd /mnt/d/Projects/xinhe
RESUME_CKPT="${1:-checkpoints/xinhe_step_14000.pt}"
if [ ! -f "$RESUME_CKPT" ]; then echo "缺 $RESUME_CKPT,等 v20 跑完 14000"; exit 1; fi
mkdir -p logs
LOG_FILE=logs/train_5080_v21_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch-v21] RESUME=$RESUME_CKPT LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v21.yaml \
  --from-stage pcap_skeleton \
  --resume "$RESUME_CKPT" \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v21] PID=$!"
sleep 10
echo "=== tail ==="
tail -15 "$LOG_FILE"
