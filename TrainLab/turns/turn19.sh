#!/bin/sh
# NewIdeas (Sun 25 Oct): MoonShots' '3B' quality screen (B9: -39.8 % SSD bytes vs v3 at 24 GB), plus turn18c192's main arm.
# p3333333333: every rank at MSB-3 (the never-measured part: ranks 0-1 at 3 bits). 3b140t32: 3B in full, a base pool of
# 140 MSB-3 experts per layer + a 2-bit cache of 32 (LRU replays seeded by the prompt's rows); a cold expert every row of
# the pass uses at rank >= 4 is read and computed at 2 bits (MSB-2 here: the suite's tier emulation), never entering the
# base pool. c192r4: B2 cold-only at C = 192 (the 45 GB waypoint). The v3 arm is turn18's (same sequences, positions).
# Kill (NewIdeas): 3B's mean KLD > 1.5x v3's on any part, or the strict rules failing by more than cold-only r4's
# margin (c128r4: code P99.9 CI to x1.62, ifeval top-1 CI to -0.8 pt). Quiet-window safe; one-sequence smoke first.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
echo "== smoke $(date +%H:%M:%S)"
TL_TAG=b3b TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 3b140t32 tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee /tmp/turn19_smoke.$$
grep -q "SSD per token" /tmp/turn19_smoke.$$ || { echo "== smoke FAILED"; rm -f /tmp/turn19_smoke.$$; exit 1; }
rm -f /tmp/turn19_smoke.$$
echo "== full $(date +%H:%M:%S)"
TL_TAG=b3b TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 p3333333333,3b140t32,c192r4 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
