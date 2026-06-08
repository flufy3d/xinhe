#!/usr/bin/env bash
# 诊断 probe 63%(N=1000)是否记忆驱动:用当前(带 NM-zero 列)的 nm_generalize 重跑,
# 对比 shortcut OFF(复现 63%,看 NM-zero)vs shortcut ON(真训练纪律,NM-zero 验证后的诚实天花板)。
# 同 N=1000 隔离 shortcut 效应。fresh-init,steps=12000,flat LR 3e-4(完全照 generalize_v19b_N1000.log)。
set -e
cd /mnt/d/Projects/xinhe
CFG=configs/pcap_skeleton_5080_v19b.yaml
PY=.venv-linux/bin/python

echo "########## RUN 1: shortcut=OFF (复现 63% + 测 NM-zero) ##########"
PYTHONUNBUFFERED=1 $PY -u scripts/probe/nm_generalize.py \
  --config "$CFG" --train data/skeleton/train.jsonl --val data/skeleton/val.jsonl \
  --train-eps 1000 --eval-eps 50 --steps 12000 --lr 3e-4 --log-every 1000 \
  > logs/diag_gen_N1000_noshortcut.log 2>&1
echo "=== RUN1 DONE ==="

echo "########## RUN 2: shortcut=ON (真训练纪律,NM-zero 验证天花板) ##########"
PYTHONUNBUFFERED=1 $PY -u scripts/probe/nm_generalize.py \
  --config "$CFG" --train data/skeleton/train.jsonl --val data/skeleton/val.jsonl \
  --train-eps 1000 --eval-eps 50 --steps 12000 --lr 3e-4 --log-every 1000 \
  --shortcut --margin 0.3 --lambda-sc 2.0 \
  > logs/diag_gen_N1000_shortcut.log 2>&1
echo "=== RUN2 DONE ==="
echo "=== ALL DIAG DONE ==="
