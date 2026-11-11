#!/bin/sh
# NewIdeas Sun 05:00, B2 decisive: COLD-ONLY 2-bit at C=192. Per layer an LRU replay of residency at 192 experts (seeded by
# the prompt's rows, one update per verify pass); a cold expert every row of the pass uses at rank >= R is read and
# computed as MSB-2, all else at v3's tiers. Arms c192r4 (ranks 4-9) and c192r2 (ranks 2-9). Per position as turn18 +
# __cold_frac; SSD bytes per token v3 vs cold-only in the json (_bytes) and summary. The v3 arm for the gate is turn18's
# (same sequences and positions). Kill (NewIdeas): fails the strict rules ungated AND with the gate at >= 70 % of code
# passes. Quiet-window safe; one-sequence smoke first.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
echo "== smoke $(date +%H:%M:%S)"
TL_TAG=b2c192 TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 c192r4 tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee /tmp/turn18c_smoke.$$
grep -q "SSD per token" /tmp/turn18c_smoke.$$ || { echo "== smoke FAILED"; rm -f /tmp/turn18c_smoke.$$; exit 1; }
rm -f /tmp/turn18c_smoke.$$
echo "== full $(date +%H:%M:%S)"
TL_TAG=b2c192 TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 c192r4,c192r2 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
