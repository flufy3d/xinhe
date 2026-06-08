#!/usr/bin/env bash
# 看当前 v11 log 从开头到 step 30 步,定位 NaN onset
LOG=$(cat /mnt/d/Projects/xinhe/logs/latest_train.txt)
FULL=/mnt/d/Projects/xinhe/$LOG
echo "=== first 35 lines ==="
head -35 "$FULL"
echo "=== Step 1-30 lines (if any) ==="
grep -E "Step [1-9][0-9]?\]" "$FULL" | head -10
