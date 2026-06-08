#!/usr/bin/env bash
# v12 step 3000 全量 200 ep eval(最终验收)
set -e
cd /mnt/d/Projects/xinhe
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_3000.pt \
  --config configs/pcap_skeleton_5080_v12.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 200 \
  --output checkpoints/validate_memory_v12_step_3000_n200.json \
  2>&1 | tee logs/eval_v12_step_3000_n200.log
