#!/bin/bash
# 一次性 wrapper:5080 mini-train detached,log 写到 logs/。
# 用法(WSL 里):bash scripts/_launch_train_5080.sh
# 或 Windows 这边:wsl bash /mnt/d/Projects/xinhe/scripts/_launch_train_5080.sh
set -e
cd /mnt/d/Projects/xinhe
source .venv-linux/bin/activate
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/train_5080_${TS}.log"
echo "$LOG" > logs/latest_train.txt
echo "log -> $LOG"
CFG="${CFG:-configs/pcap_skeleton_5080_v2.yaml}"
echo "config -> $CFG"
PYTHONUNBUFFERED=1 nohup python -u scripts/train.py \
  --config "$CFG" \
  --from-stage pcap_skeleton > "$LOG" 2>&1 < /dev/null &
PID=$!
disown
echo "PID=$PID  log=$LOG"
sleep 3
ls -la "$LOG"
