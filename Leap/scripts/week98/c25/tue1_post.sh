#!/bin/bash
# tue1_post.sh (Leap, Mon 16 Nov 2026 14:1x): the post-run of Tue slot #1's c25 run (C25_POST), inside the same quiet window:
# expread (the expert one-call's bytes read with no math; + its x re-reads), exp32 (the one-call, real bytes; + kernels per
# call), gdn_fast (the GDN core vs mlx_lm's kernel, chain + independent). Each step ~10-60 s; outputs in data/week98/check25.
cd /Users/kamen/Leap || exit 9
PY=/Users/kamen/BigMoE/.venv/bin/python
echo "== expread $(date +%T)"; $PY -I scripts/week98/expread_bench.py
echo "== exp32 prof $(date +%T)"; $PY -I scripts/week98/exp32_bench.py prof
echo "== exp32 $(date +%T)"; $PY -I scripts/week98/exp32_bench.py
echo "== exp32 nolsb variants $(date +%T)"; $PY -I scripts/week98/exp32_bench.py nlv
echo "== gdn_fast $(date +%T)"; $PY -I scripts/week98/gdn_fast.py bench
echo "== done $(date +%T)"
