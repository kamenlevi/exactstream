#!/bin/bash
# run_rec_quiet.sh (Breakthrough, Tue 17 Nov 2026 01:4x; MODULE 1a milestone 1's OFFICIAL number): the ICB replay of FAST
# layer 0 with the per-replay CPU split (useResources vs a residency set), 200 repeats, run INSIDE a quiet lock that the
# caller already holds (this script takes no lock). Peak ~3.2 GB lifetime (rec1a_layer_fast: 3.18 GB); ~2 min.
# Usage (after `peer lock quiet ...` returned): bash scripts/m1/rec1a/run_rec_quiet.sh -> data/m1/rec1a_rset_layer_fast_quiet.out
cd ~/Breakthrough || exit 9
o=data/m1/rec1a_rset_layer_fast_quiet
REC_SCOPE=layer REC_MODES=9 REC_WINDOWS=4 REC_REPS=200 REC_RSET=1 "$HOME/BigMoE/.venv/bin/python" -I -B scripts/m1/rec1a/rec_layer.py --fast > "$o.out" 2> "$o.err"
echo "rc $?"
grep -E '^\[rec\] (ICB|residency|mode|RESULT|SELF)' "$o.out" | cut -c1-330
grep -E '^[A-Za-z]*Error|error|failed' "$o.err" | head -3
