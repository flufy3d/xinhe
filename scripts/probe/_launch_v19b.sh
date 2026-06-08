#!/usr/bin/env bash
# v19b GATE:read_scale_init=0(修死区),per_layer_delta,1000 步。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG_FILE=logs/train_5080_v19b_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch-v19b] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v19b.yaml \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v19b] PID=$!"
sleep 8
echo "=== tail ==="
tail -8 "$LOG_FILE"
