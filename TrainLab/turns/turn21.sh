#!/bin/sh
# NewIdeas (Sun 25 Oct 09:45), 350 STACK S5 (HC8) + S9 (top-8): one strict tf screen at v3's tiers (TL_VARIANT
# plan4433333333; tf has no repetition guard: guard off), turn18's positions (TL_MTP=1, every 4th sequence of the four
# parts): v3 + HC8 (p4433333333h), v3 + top-8 (p44333333), both (p44333333h). v3 itself = turn18's arm at the same
# positions when the smoke reproduces it bit for bit; else v3 runs in this batch too. Read-out sticky/s5s9.py (the
# suite's strict rules vs ar4 + x of v3; pre-registered in REPORT.md 'S5 + S9').
# Smoke (one code sequence, ~2 min): p4433333333h,p4433333333,p44333333,p44333333h. v3 runs AFTER an HC8 arm and must
# equal turn18's v3 there bit for bit (HC8 switches back cleanly, the engine is unchanged); HC8 and top-8 must differ
# from it; 196 HC modules swapped (MoonShots' hc8_smoke on this model).
# Time (turn19: 0.096 s per position and arm, 7,966 positions): smoke ~2 min + 3 arms ~38 min (4 if v3 reruns: ~51).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
START=$(date +%s); LIMIT=${TL_LIMIT:-2910}
SMK=$HOME/TrainLab/logs/turn21_smoke.$$
trap 'pkill -P $$; rm -f $SMK; exit 143' TERM                 # run_locked's stop reaches the running python too
left() { echo $(( LIMIT - $(date +%s) + START )); }
echo "== smoke $(date +%H:%M:%S): HC8 / v3 / top-8 / both on one code sequence"
TL_LIMIT=$(left) TL_TAG=s5s9smoke TL_MTP=1 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 4 p4433333333h,p4433333333,p44333333,p44333333h tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee $SMK
grep -q "hc8: 196 modules" $SMK || { echo "== smoke FAILED: HC8 did not swap the 196 modules"; rm -f $SMK; exit 1; }
rm -f $SMK
$PY - <<'PYEOF'
import numpy as np, os, sys
D = os.path.expanduser("~/TrainLab/data")
z = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_s5s9smoke_pos.npz")
t = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
V3, ARMS = "p4433333333", ("p4433333333h", "p44333333", "p44333333h")
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(t[f"{V3}__seq"], t[f"{V3}__pos"]))}
j = np.array([key.get((int(s), int(p)), -1) for s, p in zip(z[f"{V3}__seq"], z[f"{V3}__pos"])])
assert len(j) and (j >= 0).all(), "the smoke's positions are not all in turn18"
v3 = z[f"{V3}__kld"].astype(np.float64)
same = bool(np.array_equal(z[f"{V3}__kld"], t[f"{V3}__kld"][j]) and np.array_equal(z[f"{V3}__agree"], t[f"{V3}__agree"][j]))
fin = all(np.isfinite(z[f"{a}__kld"]).all() for a in ARMS)
dif = {a: float(np.abs(z[f"{a}__kld"].astype(np.float64) - v3).mean()) for a in ARMS}
print(f"smoke at {len(j)} positions: v3 (after an HC8 arm) {'==' if same else '!='} turn18's v3 bit for bit; "
      + ", ".join(f"{a} x{z[f'{a}__kld'].astype(np.float64).mean() / v3.mean():.3f} of v3 (mean |diff| {dif[a]:.1e})" for a in ARMS)
      + f"; all finite: {fin}")
sys.exit(1 if not fin or min(dif.values()) == 0 else 0 if same else 3)
PYEOF
rc=$?
[ $rc -ne 0 ] && [ $rc -ne 3 ] && { echo "== smoke FAILED (rc $rc: non-finite KLD, or an arm equal to v3: its switch did nothing)"; exit 1; }
ARMS="p4433333333h,p44333333,p44333333h"
[ $rc -eq 3 ] && { ARMS="p4433333333,$ARMS"; echo "== v3 differs from turn18's: v3 runs in this batch too"; }
echo "== full $(date +%H:%M:%S): $ARMS, every 4th sequence"
TL_LIMIT=$(left) TL_TAG=s5s9 TL_MTP=1 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 4 $ARMS tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
PYTHONDONTWRITEBYTECODE=1 $PY sticky/s5s9.py plan4433333333_mtp_s5s9
