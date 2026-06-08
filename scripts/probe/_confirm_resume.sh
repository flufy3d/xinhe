#!/usr/bin/env bash
# 确认 resume 真恢复了 global_step(首个 [Step 行应 ~30XX 而非 50)。用法: bash _confirm_resume.sh <LOG>
cd /mnt/d/Projects/xinhe
L="$1"
for i in $(seq 1 40); do
  if grep -qE '\[resume\]|\[Step ' "$L" 2>/dev/null; then break; fi
  if ! pgrep -f 'train.py.*v20' >/dev/null 2>&1; then echo "DIED_EARLY"; break; fi
  sleep 15
done
echo "=== resume 恢复行 ==="
grep -E '\[resume\]|global_step|加载权重' "$L" | tail -5
echo "=== 首个 step 行(resume 成功应 ~30XX) ==="
grep -E '\[Step' "$L" | head -2
echo "=== 进程 ==="
(pgrep -af 'train.py.*v20' | grep -v -E 'pgrep|_confirm|_wait' || echo NONE)
