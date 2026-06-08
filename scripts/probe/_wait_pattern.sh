#!/usr/bin/env bash
# 等某 LOG 出现 PATTERN(grep -E)或某进程名消失。用法: bash _wait_pattern.sh <LOG> <PATTERN> <PROC_PATTERN>
cd /mnt/d/Projects/xinhe
LOG="$1"; PAT="$2"; PROC="$3"
for i in $(seq 1 300); do
  if grep -qE "$PAT" "$LOG" 2>/dev/null; then echo "MATCHED"; break; fi
  if ! pgrep -f "$PROC" >/dev/null 2>&1; then echo "PROC_GONE"; break; fi
  sleep 60
done
echo "=== tail $LOG ==="
tail -12 "$LOG" 2>/dev/null
