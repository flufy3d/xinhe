#!/usr/bin/env bash
# 正确口径(weights>1.5 选 value token)重测 N-scaling,推翻/确认 probe 旧 21/29.5/50/63%@N。
# 同原 sweep 训练设置(fresh, shortcut OFF, flat LR 3e-4),仅 eval 修正。N=500/1000/2000,各 8000 步。
# 判:val NM-on(read,修正口径)随 N 明显上升 = entity-scaling 真杠杆;平在 ~10-16% = scaling 死。
set -e
cd /mnt/d/Projects/xinhe
CFG=configs/pcap_skeleton_5080_v19b.yaml
PY=.venv-linux/bin/python
for N in 500 1000 2000; do
  echo "########## N=${N} (fixed-eval, shortcut OFF, 8000 步) ##########"
  PYTHONUNBUFFERED=1 $PY -u scripts/probe/nm_generalize.py \
    --config "$CFG" --train data/skeleton/train.jsonl --val data/skeleton/val.jsonl \
    --train-eps "$N" --eval-eps 50 --steps 8000 --lr 3e-4 --log-every 1000 \
    > "logs/sweep_N${N}.log" 2>&1
  echo "=== N=${N} DONE ==="
done
echo "=== SWEEP ALL DONE ==="
