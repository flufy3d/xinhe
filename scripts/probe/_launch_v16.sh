#!/usr/bin/env bash
set -e
cd /mnt/d/Projects/xinhe
LOG_FILE=logs/train_5080_v16_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v16.yaml \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch] PID=$!"
sleep 5
echo "=== tail ==="
tail -10 "$LOG_FILE"
