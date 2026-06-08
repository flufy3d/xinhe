#!/usr/bin/env bash
# v20 step3000 三口径停评: val n50 + train n30(看泛化 & train/val gap)。
# 训练已 kill,GPU 空闲。跑完用 _dump_eval.py 解析。
set -e
cd /mnt/d/Projects/xinhe
CKPT=checkpoints/xinhe_step_3000.pt
CFG=configs/pcap_skeleton_5080_v20.yaml
LOG=logs/eval_v20_step3000.log
echo "=== VAL n50 (未见实体, 真泛化) ===" | tee "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint "$CKPT" --config "$CFG" --val data/skeleton/val.jsonl \
  --max-episodes 50 --output checkpoints/validate_memory_v20_step_3000_n50.json >> "$LOG" 2>&1
echo "=== TRAIN n30 (见过实体, memorize sanity) ===" | tee -a "$LOG"
PYTHONUNBUFFERED=1 .venv-linux/bin/python -u scripts/validate_memory.py \
  --checkpoint "$CKPT" --config "$CFG" --val data/skeleton/train.jsonl \
  --max-episodes 30 --output checkpoints/validate_memory_v20_step_3000_train_n30.json >> "$LOG" 2>&1
echo "=== EVAL DONE ===" | tee -a "$LOG"
