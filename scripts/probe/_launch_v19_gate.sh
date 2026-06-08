#!/usr/bin/env bash
# v19 GATE:fresh init, per_layer_delta, head_dim=128, LoRA r=16, 1000 步。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG_FILE=logs/train_5080_v19gate_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch-v19-gate] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v19.yaml \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v19-gate] PID=$!"
sleep 8
echo "=== tail ==="
tail -12 "$LOG_FILE"
