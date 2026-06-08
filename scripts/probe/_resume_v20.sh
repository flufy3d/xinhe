#!/usr/bin/env bash
# v20 从 mid-stage checkpoint 续训(恢复 optimizer+scheduler+global_step,跑到 max_steps=14000)。
# 用法: bash _resume_v20.sh checkpoints/xinhe_step_3000.pt
set -e
cd /mnt/d/Projects/xinhe
RESUME_CKPT="${1:?需要 checkpoint 路径}"
mkdir -p logs
LOG_FILE=logs/train_5080_v20_resume_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[resume-v20] RESUME=$RESUME_CKPT LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v20.yaml \
  --from-stage pcap_skeleton \
  --resume "$RESUME_CKPT" \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[resume-v20] PID=$!"
sleep 10
echo "=== tail ==="
tail -15 "$LOG_FILE"
