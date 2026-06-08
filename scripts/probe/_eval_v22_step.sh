#!/usr/bin/env bash
# v22 任意 step 三口径停评: val n50 + train n30。用法: bash _eval_v22_step.sh <STEP>
# arch 与 v20 相同,config 仅 shortcut 开关不同,eval 无影响。
set -e
cd /mnt/d/Projects/xinhe
STEP="${1:?需要 step 数}"
CKPT=checkpoints/xinhe_step_${STEP}.pt
CFG=configs/pcap_skeleton_5080_v22.yaml
LOG=logs/eval_v22_step${STEP}.log
echo "=== VAL n50 (未见实体, 真泛化) step=${STEP} ===" | tee "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint "$CKPT" --config "$CFG" --val data/skeleton/val.jsonl \
  --max-episodes 50 --output checkpoints/validate_memory_v22_step_${STEP}_n50.json >> "$LOG" 2>&1
echo "=== TRAIN n30 (见过实体, memorize sanity) ===" | tee -a "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint "$CKPT" --config "$CFG" --val data/skeleton/train.jsonl \
  --max-episodes 30 --output checkpoints/validate_memory_v22_step_${STEP}_train_n30.json >> "$LOG" 2>&1
echo "=== EVAL DONE step=${STEP} ===" | tee -a "$LOG"
