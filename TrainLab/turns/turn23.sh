#!/bin/sh
# R-26's read-out (CPU, ~1 min; back-to-back quiet windows kept it out) + R-31 UNIQ-Q (NewIdeas Tue ~20:45, QUEUE R-31;
# pre-registered in REPORT.md 'R-31 UNIQ-Q'): MoonShots' check-level UNIQ rule (an expert used only at ranks >= 4 in the
# 16-row check is dropped from it when its summed renormalized weight < theta; dynk.uniq_scores inside the suite's block)
# at 16-row verify passes of true tokens, v3's tiers: v3 at 16 rows (control), u080, u120; u000 smoke on the first
# sequence (= v3 at 16 rows bit for bit). Every 8th sequence of ifeval / code / multi. Read-out after the lock:
# sticky/s5s9.py plan4433333333_r31 (strict rules vs ar4 + x of v3 at 16 rows, from this run).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
echo "== R-26 read-out $(date +%H:%M:%S)"
PYTHONDONTWRITEBYTECODE=1 $PY sticky/s5s9.py plan4433333333_mtp_r26 | tail -9
$PY tests/test_dynk.py | tail -1 | grep -q "ALL OK" || { echo "== test_dynk FAILED: R-31 not started"; exit 1; }
echo "== R-31 $(date +%H:%M:%S): u000 (smoke, first sequence) + v3 / u080 / u120 at 16 rows, ifeval / code / multi, every 8th"
TL_ONCE=u000 TL_LIMIT=$(( ${TL_LIMIT:-570} - 30 )) TL_TAG=r31 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 16 u000,p4433333333,u080,u120 tf_ifeval,tf_code,tf_multi 8 2>&1 | grep --line-buffered -v "Token indices"
$PY - <<'PYEOF'
import numpy as np, os
z = np.load(os.path.expanduser("~/TrainLab/data/budget125_tf_R16_plan4433333333_r31_pos.npz"))
V3 = "p4433333333"
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(z[f"{V3}__seq"], z[f"{V3}__pos"]))}
j = np.array([key.get((int(s), int(p)), -1) for s, p in zip(z["u000__seq"], z["u000__pos"])])
same = len(j) > 0 and (j >= 0).all() and np.array_equal(z["u000__kld"], z[f"{V3}__kld"][j]) \
    and np.array_equal(z["u000__agree"], z[f"{V3}__agree"][j])
d = {a: float(np.abs(z[f"{a}__kld"][: len(j)].astype(np.float64) - z["u000__kld"].astype(np.float64)).mean()) for a in ("u080", "u120")}
print(f"== SMOKE {'PASS' if same and min(d.values()) > 0 else 'FAIL'}: u000 at {len(j)} positions {'==' if same else '!='} v3 at "
      f"16 rows bit for bit; u080 / u120 there differ by mean |dKLD| {d['u080']:.2e} / {d['u120']:.2e}")
PYEOF
