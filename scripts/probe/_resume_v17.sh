#!/usr/bin/env bash
# 从 step_1500 续训 v17 到 step 3000(断电恢复)。--resume 含 xinhe_step_ 会恢复 optimizer+scheduler+global_step。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG_FILE=logs/train_5080_v17_resume_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[resume] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v17.yaml \
  --resume checkpoints/xinhe_step_1500.pt \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[resume] PID=$!"
sleep 6
echo "=== tail ==="
tail -12 "$LOG_FILE"
