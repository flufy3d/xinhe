#!/usr/bin/env bash
# v18 GATE:fresh init,d_value=1024,MAC+MAL,1000 步。跑完重跑 _probe_mem_decode 看 fact_self rank。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG_FILE=logs/train_5080_v18gate_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
echo "[launch-v18-gate] LOG_FILE=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v18.yaml \
  --from-stage pcap_skeleton \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v18-gate] PID=$!"
sleep 6
echo "=== tail ==="
tail -8 "$LOG_FILE"
