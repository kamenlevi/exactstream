#!/bin/bash
# run_rec_set.sh (Breakthrough, MODULE 1a): the record/replay matrix in bash (zsh does not word-split env lists).
# Usage: bash scripts/m1/rec1a/run_rec_set.sh  -> data/m1/rec1a_<scope>_<route>.out / .err
cd ~/Breakthrough || exit 9
for cfg in "layer stock" "model fast" "layer fast"; do
  set -- $cfg
  sc=$1; rt=$2; fl=""; [ "$rt" = fast ] && fl="--fast"
  o=data/m1/rec1a_${sc}_${rt}
  REC_SCOPE=$sc REC_MODES=0,2 REC_WINDOWS=16 REC_DET=0 /usr/bin/time -l "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/rec1a/rec_layer.py $fl > "$o.out" 2> "$o.err"
  echo "== $sc $rt: rc $? $(grep -E 'peak memory' "$o.err" | awk '{printf "peak %.2f GB", $1/1e9}')"
  grep -E '^\[rec\] (scope|op types|mode|RESULT|SELF)' "$o.out" | cut -c1-330
  grep -E '^[A-Za-z]*Error' "$o.err" | head -2
done
