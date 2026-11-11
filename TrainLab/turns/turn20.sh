#!/bin/sh
# NewIdeas (Sun 25 Oct ~07:00), B10 v2 layer-sliced quality screen: Fit12's trellis-coded experts (base for ranks 2-9,
# base + residual for ranks 0-1; decoder ~/Fit12/scripts/fit_b10.py, read-only), one layer at a time.
# Design (TrainLab 09:35, isolated layers): the kill line is a ~1 % KLD excess per layer, so "the other layers v3" drowns
# in the cross terms with v3's 47 other layers and the R-row floor (20-35 % of v3). Each layer instead runs ALONE (the
# other layers exact 4-bit), scored as lkld = KLD(the all-4-bit arm || arm) over the full vocabulary in the same passes
# (TL_LOCAL): d<L> = v3's tiers at L only, t<L> = the trellis at L only; r_L = t / d; all 48 layers
# x((F + mean r x Q) / v3) at turn18's v3 positions (sticky/b10_extrap.py). One process per layer: a trellis layer is
# ~5.4 GB in RAM (int8), next to the engine's v3 pools.
# Smoke (one multi sequence, the other layers v3): d<L1> must reproduce v3 (its lkld under 10 % of x<L1>'s = layer L1's
# own v3 damage, KLD vs the reference within 2 %); 10-50 % adds the x<L> floor arms; above 50 % the run stops.
# Kill (NewIdeas): > 1.5x v3 per part with all 48 layers trellis-coded. Time: ~3.6 min per arm at EVERY 16 (10 sequences:
# gsm 4, the other parts 2 each; turn19: 0.096 s per position and arm) -> smoke ~2 min + 3 layers x (3 arms + decode)
# ~35 min + 3 decodes; the engine is ~20 GB (C 128 pools + scratch) + ~6.8 GB of dense sources. Quiet-window safe.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
LAYERS=${TL_B10_LAYERS:-"2 24 46"}
EVERY=${TL_B10_EVERY:-16}
L1=$(echo $LAYERS | cut -d' ' -f1)
START=$(date +%s); LIMIT=${TL_LIMIT:-2600}
trap 'pkill -P $$; exit 143' TERM                          # run_locked's stop reaches the running python too
[ -f $HOME/Fit12/scripts/fit_b10.py ] || { echo "== no decoder at ~/Fit12/scripts/fit_b10.py: not started"; exit 1; }
left() { echo $(( LIMIT - $(date +%s) + START )); }
echo "== smoke $(date +%H:%M:%S): p4433333333,d$L1,x$L1 on one multi sequence"
TL_LIMIT=$(left) TL_TAG=b10smoke TL_MTP=1 TL_LOCAL=1 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 4 p4433333333,d$L1,x$L1 tf_multi 24 2>&1 | grep --line-buffered -v "Token indices"
$PY - "$L1" <<'PYEOF'
import numpy as np, os, sys
L1 = sys.argv[1]
z = np.load(os.path.expanduser("~/TrainLab/data/budget125_tf_R4_plan4433333333_mtp_b10smoke_pos.npz"))
d, x = z[f"d{L1}__lkld"].astype(np.float64), z[f"x{L1}__lkld"].astype(np.float64)
kv, kd = z["p4433333333__kld"].astype(np.float64), z[f"d{L1}__kld"].astype(np.float64)
fl, rel = d.mean() / x.mean(), kd.mean() / kv.mean() - 1
print(f"smoke at {d.size} positions, lkld vs v3: d{L1} {d.mean():.2e} (the dense path's floor), x{L1} {x.mean():.2e} "
      f"(layer {L1}'s own v3 damage) -> floor {fl:.1%} of it; KLD vs the reference d{L1} {kd.mean():.5f}, v3 "
      f"{kv.mean():.5f} ({rel:+.2%})")
sys.exit(1 if (not x.mean() > 0 or fl > 0.5 or abs(rel) > 0.02) else 3 if fl > 0.1 else 0)
PYEOF
rc=$?
[ $rc -eq 1 ] && { echo "== smoke FAILED: the dense path does not reproduce v3"; exit 1; }
SUF=""
for L in $LAYERS; do
  ARMS="p4444444444,d${L}p4444444444,t${L}p4444444444"
  [ $rc -eq 3 ] && ARMS="$ARMS,x${L}p4444444444"
  echo "== layer $L $(date +%H:%M:%S): $ARMS, every ${EVERY}th sequence"
  TL_LIMIT=$(left) TL_TAG=b10L$L TL_MTP=1 TL_LOCAL=1 TL_VARIANT=plan4433333333 \
    $PY sticky/budget125_tf.py 4 $ARMS tf_gsm,tf_ifeval,tf_code,tf_multi $EVERY 2>&1 | grep --line-buffered -v "Token indices"
  SUF="$SUF${SUF:+,}plan4433333333_mtp_b10L$L"
  $PY sticky/b10_extrap.py $SUF
done
