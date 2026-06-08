#!/usr/bin/env bash
# 后台启动 v17 eval,detach so wsl session 关闭也不影响
set -e
STEP=${1:-1000}
N=${2:-50}
cd /mnt/d/Projects/xinhe
mkdir -p logs
LOG=logs/eval_v17_step_${STEP}.log
echo "[bg-eval] STEP=$STEP N=$N LOG=$LOG"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_${STEP}.pt \
  --config configs/pcap_skeleton_5080_v17.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes ${N} \
  --output checkpoints/validate_memory_v17_step_${STEP}_n${N}.json \
  > "$LOG" 2>&1 < /dev/null &
echo "[bg-eval] PID=$!"
sleep 2
ls -la "$LOG"
