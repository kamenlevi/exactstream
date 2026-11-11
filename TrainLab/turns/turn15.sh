#!/bin/sh
# NewIdeas 10-22 16:55, step 1: the COMBINED quality screen Kamen set for fast modes: the per-pass expert budget ON TOP of
# v3's tiers (Eval suite variant plan4433333333: exact 4-bit prompt prefill, tiered 4-row verify passes; the suite tiers
# multi-row calls only on its full-E prefill scratch, so budget125_tf runs pool-slot verify passes row by row through the
# suite's decode tier branch), budgets 22 / 20 / 18 + the tiered 4-row floor, four tf parts, every 4th sequence (~30 min).
# Pick the largest-cut budget whose per-part KLD passes (ar4 x1.10: code 0.0025 / gsm 0.0045 / ifeval 0.0204 / multi 0.0205).
# A one-sequence smoke run goes first (19:56: the first try asserted in the suite's tier branch and lost the lock).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
echo "== smoke $(date +%H:%M:%S)"
TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 exact,b20 tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee /tmp/turn15_smoke.$$
grep -q "budget125_tf R=4 variant plan4433333333: 1 sequences" /tmp/turn15_smoke.$$ || { echo "== smoke FAILED: no full run"; rm -f /tmp/turn15_smoke.$$; exit 1; }
rm -f /tmp/turn15_smoke.$$
echo "== full $(date +%H:%M:%S)"
TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 exact,b22,b20,b18 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
