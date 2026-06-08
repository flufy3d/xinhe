#!/usr/bin/env bash
# 等某个 eval log 出现 "EVAL DONE" 或 validate_memory 进程消失。
# 用法: bash _wait_eval_done.sh <EVAL_LOG>
cd /mnt/d/Projects/xinhe
LOG="$1"
for i in $(seq 1 120); do
  if grep -q 'EVAL DONE' "$LOG" 2>/dev/null; then
    echo "EVAL_DONE"; break
  fi
  if ! pgrep -f 'validate_memory.py' >/dev/null 2>&1; then
    echo "EVAL_PROC_GONE"; break
  fi
  sleep 20
done
echo "=== tail eval log ==="
tail -4 "$LOG" 2>/dev/null
