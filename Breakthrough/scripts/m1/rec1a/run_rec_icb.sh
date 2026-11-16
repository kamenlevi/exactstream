#!/bin/bash
# run_rec_icb.sh (Breakthrough, MODULE 1a): the ICB form vs the merged native replay on layer 0, stock and FAST routes.
# Functional (bitwise) + ROUGH CPU (share lock, other runs beside it). Usage: bash scripts/m1/rec1a/run_rec_icb.sh
cd ~/Breakthrough || exit 9
for rt in fast stock; do
  fl=""; [ "$rt" = fast ] && fl="--fast"
  o=data/m1/rec1a_icb_layer_${rt}
  REC_SCOPE=layer REC_MODES=2,9 REC_WINDOWS=16 REC_REPS=40 /usr/bin/time -l "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/rec1a/rec_layer.py $fl > "$o.out" 2> "$o.err"
  echo "== layer $rt: rc $? $(grep -E 'peak memory' "$o.err" | awk '{printf "peak %.2f GB", $1/1e9}')"
  grep -E '^\[rec\] (ICB|\(a\)|mode|RESULT|SELF)' "$o.out" | cut -c1-360
  grep -E '^[A-Za-z]*Error|error' "$o.err" | head -3
done
