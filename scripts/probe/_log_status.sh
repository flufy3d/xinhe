#!/usr/bin/env bash
LOG=$(cat /mnt/d/Projects/xinhe/logs/latest_train.txt)
FULL=/mnt/d/Projects/xinhe/$LOG
echo "file: $FULL"
stat -c '%y  size=%s' "$FULL"
echo "---"
tail -15 "$FULL"
