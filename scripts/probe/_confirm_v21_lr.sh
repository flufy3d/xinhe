#!/usr/bin/env bash
# 确认 v21 resume:global_step=14000 恢复 + LR 回升到 ~2.2e-4(证明 horizon 扩展生效)。
# 用法: bash _confirm_v21_lr.sh <LOG>
cd /mnt/d/Projects/xinhe
L="$1"
for i in $(seq 1 40); do
  if grep -qE '\[Step ' "$L" 2>/dev/null; then break; fi
  if ! pgrep -f 'python -u scripts/train.py' >/dev/null 2>&1; then echo "DIED_EARLY"; break; fi
  sleep 15
done
echo "=== resume 恢复行 ==="
grep -E '\[resume\]|global_step|加载权重' "$L" | tail -5
echo "=== 首个 step 行(LR 应 ~2.2e-4 不是 3e-6) ==="
grep -E '\[Step' "$L" | head -3
