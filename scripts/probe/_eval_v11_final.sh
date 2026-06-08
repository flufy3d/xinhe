#!/usr/bin/env bash
# v11 step 3000 全量 200 ep eval(最终验收)
set -e
cd /mnt/d/Projects/xinhe
echo "=== ckpts ==="
ls -la checkpoints/xinhe_step_*.pt
echo "=== eval step 3000 (200 ep, val.jsonl) ==="
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_3000.pt \
  --config configs/pcap_skeleton_5080_v11.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 200 \
  --output checkpoints/validate_memory_v11_step_3000_n200.json \
  2>&1 | tee logs/eval_v11_step_3000_n200.log
