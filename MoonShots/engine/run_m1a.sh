#!/bin/sh
# Engine M1a correctness tests vs MTPLX on the real 125B (launch while holding a quiet lock; unlocks at the end).
OUT=$HOME/MoonShots/data/day15; PY=$HOME/BigMoE/.venv/bin/python; E=$HOME/MoonShots/engine; TAG=${TAG:-faith1}
sh $HOME/MoonShots/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
echo "$(date +%H:%M) m1a $TAG started" >> "$OUT/summary.txt"
$PY $E/test_m1a_layers.py > "$OUT/m1a_layers_$TAG.txt" 2>&1
echo "$(date +%H:%M) done m1a_layers $TAG: $(grep -E '^(logits|mean single|Traceback|[A-Za-z]*Error)' "$OUT/m1a_layers_$TAG.txt" | cut -c1-200 | tr '\n' ' ')" >> "$OUT/summary.txt"
$PY $E/test_m1a_model.py 32 4 > "$OUT/m1a_model_$TAG.txt" 2>&1
echo "$(date +%H:%M) done m1a_model $TAG: $(grep -E '^(M=|PASS|CHECK|Traceback|[A-Za-z]*Error)' "$OUT/m1a_model_$TAG.txt" | cut -c1-200 | tr '\n' ' ')" >> "$OUT/summary.txt"
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) m1a $TAG finished" >> "$OUT/summary.txt"
