#!/bin/bash
# run_rec_rset.sh (Breakthrough, MODULE 1a): the ICB replay on FAST layer 0 with the per-replay CPU split, useResources vs a
# residency set. ROUGH (share lock). Usage: bash scripts/m1/rec1a/run_rec_rset.sh -> data/m1/rec1a_rset_layer_fast.out
cd ~/Breakthrough || exit 9
o=data/m1/rec1a_rset_layer_fast
REC_SCOPE=layer REC_MODES=9 REC_WINDOWS=4 REC_REPS=40 REC_RSET=1 /usr/bin/time -l "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/rec1a/rec_layer.py --fast > "$o.out" 2> "$o.err"
echo "rc $? $(grep -E 'peak memory' "$o.err" | awk '{printf "peak %.2f GB", $1/1e9}')"
grep -E '^\[rec\] (ICB|residency|mode|RESULT|SELF)' "$o.out" | cut -c1-330
grep -E '^[A-Za-z]*Error|error|failed' "$o.err" | head -3
