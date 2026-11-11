#!/bin/bash
# Waits for a quiet lock, checks the GPU is idle, runs the micro-bench, unlocks. Output: data/$1 (default alloc_bench.txt)
cd ~/NewIdeas && mkdir -p data
OUT="data/${1:-alloc_bench.txt}"
peer lock quiet --queue 10m "NewIdeas: MLX fresh-allocation micro-bench (~2 min GPU, <1 GB)" >> "$OUT" 2>&1
for i in 1 2 3 4 5 6; do g=$(peer top 2>/dev/null | head -1 | sed -n 's/.*GPU \([0-9]*\)%.*/\1/p'); echo "$(date +%H:%M:%S) GPU ${g:-?}%" >> "$OUT"; [ -n "$g" ] && [ "$g" -le 30 ] && break; sleep 20; done
taskpolicy -b ~/BigMoE/.venv/bin/python -I scripts/alloc_bench.py >> "$OUT" 2>&1
echo "$(date +%H:%M:%S) done" >> "$OUT"
peer unlock >> "$OUT" 2>&1
