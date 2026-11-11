#!/bin/sh
# MoonShots day_runner14d: tierglu4 bit-trick builders: bitwise re-test + speed bench. Launch while holding a lock.
OUT=$HOME/MoonShots/data/day14; PY=$HOME/BigMoE/.venv/bin/python; E=$HOME/MoonShots/patches/planes_engine
echo "$(date +%H:%M) day_runner14d started" >> "$OUT/summary.txt"
$PY $E/test_tierglu4.py > "$OUT/test_tierglu4_bt.txt" 2>&1
echo "$(date +%H:%M) done test_tierglu4_bt: $(grep -E '^(PASS|FAIL)' "$OUT/test_tierglu4_bt.txt")" >> "$OUT/summary.txt"
$PY $E/bench_tierglu4.py > "$OUT/bench_tierglu4_bt.txt" 2>&1
echo "$(date +%H:%M) done bench_tierglu4_bt: $(grep -E 'M=(1|8):' "$OUT/bench_tierglu4_bt.txt" | tr '\n' ' ' | cut -c1-600)" >> "$OUT/summary.txt"
$PY $HOME/MoonShots/patches/sol_layer/sol_layer.py > "$OUT/sol_layer.txt" 2>&1
echo "$(date +%H:%M) done sol_layer: $(grep -E '^(M=|weights|Traceback|[A-Za-z]*Error)' "$OUT/sol_layer.txt" | tr '\n' ' ' | cut -c1-600)" >> "$OUT/summary.txt"
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) day_runner14d finished" >> "$OUT/summary.txt"
