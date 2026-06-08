#!/usr/bin/env bash
set -e
cd /mnt/d/Projects/xinhe
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_500.pt \
  --config configs/pcap_skeleton_5080_v11.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 50 \
  --output checkpoints/validate_memory_v11_step_500_n50.json \
  2>&1 | tee logs/eval_v11_step_500.log
