#!/usr/bin/env bash
set -e
cd /mnt/d/Projects/xinhe
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_3000.pt \
  --config configs/pcap_skeleton_5080_v15.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 200 \
  --output checkpoints/validate_memory_v15_step_3000_n200.json \
  2>&1 | tee logs/eval_v15_step_3000_n200.log
