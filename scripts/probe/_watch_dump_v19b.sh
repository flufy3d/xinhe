#!/usr/bin/env bash
cd /mnt/d/Projects/xinhe
LOG=logs/eval_v19b_run.log
for i in $(seq 1 150); do
  if grep -q 'ALL DONE' "$LOG" 2>/dev/null; then break; fi
  if grep -qE 'Traceback|Error:' "$LOG" 2>/dev/null; then
    echo "EVALERR"; tail -25 "$LOG"; exit 1
  fi
  sleep 20
done
echo "===== VAL n50 ====="
.venv-linux/bin/python scripts/probe/_dump_eval.py checkpoints/validate_memory_v19b_step_1000_n50.json
echo "===== TRAIN n30 ====="
.venv-linux/bin/python scripts/probe/_dump_eval.py checkpoints/validate_memory_v19b_step_1000_train_n30.json
echo "WATCH_END"
