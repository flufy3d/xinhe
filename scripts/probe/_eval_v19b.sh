#!/usr/bin/env bash
set -e
cd /mnt/d/Projects/xinhe
LOG=logs/eval_v19b_run.log
echo "=== VAL n50 ===" | tee "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py --checkpoint checkpoints/xinhe_step_1000.pt --config configs/pcap_skeleton_5080_v19b.yaml --val data/skeleton/val.jsonl --max-episodes 50 --output checkpoints/validate_memory_v19b_step_1000_n50.json >> "$LOG" 2>&1
echo "=== TRAIN n30 ===" | tee -a "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py --checkpoint checkpoints/xinhe_step_1000.pt --config configs/pcap_skeleton_5080_v19b.yaml --val data/skeleton/train.jsonl --max-episodes 30 --output checkpoints/validate_memory_v19b_step_1000_train_n30.json >> "$LOG" 2>&1
echo "=== ALL DONE ===" | tee -a "$LOG"
