#!/usr/bin/env bash
# 训练健康巡检:一行总结 + 异常时展开
LATEST=/mnt/d/Projects/xinhe/logs/latest_train.txt
if [ ! -f "$LATEST" ]; then
  echo "NO_LATEST_FILE"
  exit 0
fi
LOG=$(cat "$LATEST")
if [ -z "$LOG" ]; then
  echo "EMPTY_LATEST"
  exit 0
fi
LOG_FULL="/mnt/d/Projects/xinhe/$LOG"
if [ ! -f "$LOG_FULL" ]; then
  echo "NO_LOG: $LOG_FULL"
  exit 0
fi
echo "=== tail ==="
tail -5 "$LOG_FULL"
echo "=== proc ==="
ps aux | grep "train.py" | grep -v grep | awk '{print $2, $10}' | head -1
echo "=== fatal nan (ema_loss=nan) ==="
grep -c "ema_loss=nan" "$LOG_FULL" 2>/dev/null | head -1
echo "=== benign NaN-skip count ==="
grep -c "\[NaN-skip\]" "$LOG_FULL" 2>/dev/null | head -1
