#!/usr/bin/env bash
# v22 = v20 单变量(shortcut OFF)fresh run。判 shortcut 是否过度压制 per_layer_delta 真记忆。
# fresh init(--from-stage pcap_skeleton 无 --resume → start_idx=0 不续旧 ckpt)。save_every=1000 停评。
set -e
cd /mnt/d/Projects/xinhe
mkdir -p logs
RESUME_CKPT="${1:-}"   # 可选:传 ckpt 路径 = 续训;不传 = fresh
LOG_FILE=logs/train_5080_v22_$(date +%Y%m%d_%H%M%S).log
echo "$LOG_FILE" > logs/latest_train.txt
RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then RESUME_ARG="--resume $RESUME_CKPT"; fi
echo "[launch-v22] shortcut OFF, RESUME='${RESUME_CKPT:-fresh}' LOG=$LOG_FILE"
PYTHONUNBUFFERED=1 nohup .venv-linux/bin/python -u scripts/train.py \
  --config configs/pcap_skeleton_5080_v22.yaml \
  --from-stage pcap_skeleton \
  $RESUME_ARG \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "[launch-v22] PID=$!"
sleep 12
echo "=== tail ==="
tail -15 "$LOG_FILE"
