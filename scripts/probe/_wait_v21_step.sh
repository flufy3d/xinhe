#!/usr/bin/env bash
# 等 v21 训练到 [Step $TARGET] 或进程挂掉。用法: bash _wait_v21_step.sh <LOG> <TARGET>
# 注意:死亡检测用 v21 专属 pattern(v20 版 hardcoded 'v20' 会对 v21 误报 TRAIN_DIED)。
cd /mnt/d/Projects/xinhe
LOG="$1"
TARGET="$2"
for i in $(seq 1 300); do
  if grep -qE "\[Step ${TARGET}\]" "$LOG" 2>/dev/null; then
    echo "REACHED_${TARGET}"; break
  fi
  if ! pgrep -f 'scripts/train.py.*v21' >/dev/null 2>&1; then
    echo "TRAIN_DIED"; break
  fi
  sleep 120
done
echo "=== 最近 step 行 ==="
grep -E '\[Step' "$LOG" | tail -4
echo "=== 报错检测 ==="
grep -ciE 'traceback|out of memory|cuda error' "$LOG"
echo "=== checkpoints (新→旧) ==="
ls -la --time-style=+%m-%d/%H:%M checkpoints/xinhe_step_*.pt 2>/dev/null | sort -k6 | tail -4
