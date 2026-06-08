#!/bin/bash
# 自动 eval loop:训练每生成新 ckpt 就跑 eval,结果汇总到 logs/auto_eval_<config>.log
# 用法:bash scripts/probe/_auto_eval_loop.sh <yaml> <eval_tag>
# 例:bash scripts/probe/_auto_eval_loop.sh configs/pcap_skeleton_5080_v5.yaml v5
set -e
CFG="${1:-configs/pcap_skeleton_5080_v5.yaml}"
TAG="${2:-auto}"
EVAL_DIR="/mnt/d/Projects/xinhe/checkpoints/auto_eval"
mkdir -p "$EVAL_DIR"
SUMMARY="$EVAL_DIR/${TAG}_summary.log"
echo "=== auto eval loop ($TAG) === $(date)" > "$SUMMARY"
echo "config: $CFG" >> "$SUMMARY"

cd /mnt/d/Projects/xinhe
source .venv-linux/bin/activate

# 等已有 ckpt 出现
last_mtime=0
seen_files=""
while true; do
  for f in checkpoints/xinhe_step_*.pt checkpoints/curriculum/pcap_skeleton.pt; do
    [ -f "$f" ] || continue
    mtime=$(stat -c %Y "$f")
    if [ "$mtime" -gt "$last_mtime" ] && ! echo "$seen_files" | grep -q "$f"; then
      seen_files="$seen_files $f"
      last_mtime=$mtime
      # 等 5s 让 ckpt 写完
      sleep 5
      OUT="$EVAL_DIR/${TAG}_$(basename ${f%.pt}).json"
      echo "[eval] $f -> $OUT" | tee -a "$SUMMARY"
      python -u scripts/validate_memory.py \
        --checkpoint "$f" \
        --config "$CFG" \
        --val data/skeleton/val.jsonl \
        --max-episodes 50 \
        --output "$OUT" 2>&1 | grep -E 'overall|read|write' >> "$SUMMARY" || echo "[eval failed]" >> "$SUMMARY"
      echo "---" >> "$SUMMARY"
      python scripts/probe/_eval_summary.py "$OUT" >> "$SUMMARY" 2>&1
      echo "===" >> "$SUMMARY"
    fi
  done

  # 检查训练是否结束
  if ! pgrep -f 'train.py' > /dev/null; then
    echo "训练已结束,auto_eval loop 退出" | tee -a "$SUMMARY"
    break
  fi
  sleep 30
done
