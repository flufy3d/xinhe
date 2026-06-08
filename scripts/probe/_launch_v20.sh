#!/usr/bin/env bash
# v20 真训练:per_layer_delta + shortcut + num_train=2000 + max_steps=14000(~14 visits/ep)。
# fresh 起(--from-stage,start_idx=0 → 不续 v19b)。save_every=1000 中途可停。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG_FILE=logs/train_5080_v20_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch-v20] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v20.yaml \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v20] PID=$!"
sleep 8
echo "=== tail ==="
tail -10 "$LOG_FILE"
