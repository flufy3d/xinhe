#!/usr/bin/env bash
# v13 step 1000 大样本 200 ep eval — 验证 1.8% read 是否真实
set -e
cd /mnt/d/Projects/xinhe
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint checkpoints/xinhe_step_1000.pt \
  --config configs/pcap_skeleton_5080_v13.yaml \
  --val data/skeleton/val.jsonl \
  --max-episodes 200 \
  --output checkpoints/validate_memory_v13_step_1000_n200.json \
  2>&1 | tee logs/eval_v13_step_1000_n200.log
