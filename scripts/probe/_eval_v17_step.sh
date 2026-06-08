#!/usr/bin/env bash
set -e
STEP=${1:-1000}
cd /mnt/d/Projects/xinhe
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_${STEP}.pt \
  --config configs/pcap_skeleton_5080_v17.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 50 \
  --output checkpoints/validate_memory_v17_step_${STEP}_n50.json \
  2>&1 | tee logs/eval_v17_step_${STEP}.log
