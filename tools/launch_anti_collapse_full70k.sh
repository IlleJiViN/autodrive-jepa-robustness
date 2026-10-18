#!/bin/bash
set -e
cd /home/coder/autodrive-jepa-robustness
mkdir -p runs

export PYTHONPATH=/home/coder/autodrive-jepa-robustness/src:$PYTHONPATH
export PYTHONUNBUFFERED=1

echo "=== Launching Anti-Collapse JEPA Full-Scale 70,000-Image Benchmark at $(date) ===" >> runs/anti_collapse_full70k.log

nohup /home/coder/venv/bin/python3 -u tools/run_anti_collapse_full70k.py \
    --pretrain-epochs 5 \
    --pretrain-batch-size 64 \
    --detect-epochs 12 \
    --detect-batch-size 32 \
    --workers 6 \
    --device cuda \
    --output-dir runs/rtx5090-v2-jepa-full70k-anti-collapse \
    >> runs/anti_collapse_full70k.log 2>&1 &

PID=$!
echo $PID > runs/anti_collapse_full70k.pid
echo "Started background process PID: $PID at $(date)"
