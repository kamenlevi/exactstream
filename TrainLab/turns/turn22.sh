#!/bin/sh
# R-26 DYNK-Q (NewIdeas Tue ~20:15, QUEUE R-26; pre-registered in REPORT.md 'R-26 DYNK-Q'): v3 + DYNK at tau 0.06
# (arm q060: each verify row drops its rank-8 / rank-9 expert when its top-10-renormalized gate weight < 0.06, inside the
# suite's block on v3's own sorted scores; dynk.py) on S9's harness: TL_MTP=1, v3's tiers (plan4433333333), R 4, S9's
# sequences (every 4th) of the three parts fixed top-8 failed (ifeval, code, multi; gsm left out to fit the 10 min).
# Smoke on the first sequence (TL_ONCE): q000 = the DYNK block with nothing dropped must equal turn18's v3 bit for bit at
# its positions (the block copy and the engine are v3's), q060 must differ there. Read-out after the lock:
# sticky/s5s9.py plan4433333333_mtp_r26 (strict rules vs ar4 + x of turn18's v3 at the same positions).
# Time (turn19: 0.096 s per position and arm): ~20 sequences x ~226 positions ~7 min + load; the time limit stops early.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
$PY tests/test_dynk.py | tail -1 | grep -q "ALL OK" || { echo "== test_dynk FAILED: R-26 not started"; exit 1; }
echo "== R-26 $(date +%H:%M:%S): q000 (smoke, first sequence) + q060 on ifeval / code / multi, every 4th sequence"
TL_ONCE=q000 TL_LIMIT=$(( ${TL_LIMIT:-510} + 60 )) TL_TAG=r26 TL_MTP=1 TL_VARIANT=plan4433333333 \
  $PY sticky/budget125_tf.py 4 q000,q060 tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
$PY - <<'PYEOF'
import numpy as np, os
D = os.path.expanduser("~/TrainLab/data")
z = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_r26_pos.npz")
t = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
V3 = "p4433333333"
key = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(t[f"{V3}__seq"], t[f"{V3}__pos"]))}
j = np.array([key.get((int(s), int(p)), -1) for s, p in zip(z["q000__seq"], z["q000__pos"])])
ok = len(j) > 0 and (j >= 0).all()
same = ok and np.array_equal(z["q000__kld"], t[f"{V3}__kld"][j]) and np.array_equal(z["q000__agree"], t[f"{V3}__agree"][j])
m = np.isin(z["q060__seq"], np.unique(z["q000__seq"]))
dif = float(np.abs(z["q060__kld"][m].astype(np.float64) - z["q000__kld"].astype(np.float64)).mean()) if m.sum() == len(j) else -1
print(f"== SMOKE {'PASS' if same and dif > 0 else 'FAIL'}: q000 at {len(j)} positions {'==' if same else '!='} turn18's v3 bit "
      f"for bit; q060 there differs by mean |dKLD| {dif:.2e}")
PYEOF
