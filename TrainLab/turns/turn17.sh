#!/bin/sh
# NewIdeas 10-22 21:10: the DRAFT-gated budget screen. As turn15 (v3 tiers via plan4433333333; exact prompt prefill,
# tiered 4-row verify passes row by row) with arms exact / b20 / b22, plus, per position, the MTP drafter's own depth-1
# top-1 log-prob for the same prediction (TL_MTP=1: the head on the committed pair (h_(p-1), token p), from each arm's
# own hidden states, as spec125's _MTPHead drafts it) -> data/budget125_tf_R4_plan4433333333_mtp_pos.npz (__mtp_lp1,
# __mtp_hit). NewIdeas judges the DRAFT gate on CPU (ugb_rule.py). Kill: the gate must pass all four parts with the
# budget on >= 80 % of code positions. A smoke run goes first: one code sequence, with drafter sanity checks.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
echo "== smoke $(date +%H:%M:%S)"
TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 exact,b20 tf_code 24 2>&1 | grep --line-buffered -v "Token indices" | tee /tmp/turn17_smoke.$$
grep -q "budget125_tf R=4 variant plan4433333333: 1 sequences" /tmp/turn17_smoke.$$ || { echo "== smoke FAILED (run)"; rm -f /tmp/turn17_smoke.$$; exit 1; }
rm -f /tmp/turn17_smoke.$$
$PY - <<'PYEOF' || { echo "== smoke FAILED (drafter sanity)"; exit 1; }
import numpy as np, os
z = np.load(os.path.expanduser("~/TrainLab/data/budget125_tf_R4_plan4433333333_mtp_pos.npz"))
for arm in ("exact", "b20"):
    lp, hit, tl = z[f"{arm}__mtp_lp1"], z[f"{arm}__mtp_hit"], z[f"{arm}__lp1"]
    ok = np.isfinite(lp)
    h = hit[hit != 255]
    c = np.corrcoef(lp[ok], tl[ok])[0, 1]
    print(f"smoke drafter {arm}: finite {ok.mean():.3f}, top-1 = next token {h.mean():.3f} (n {h.size}), mean p1 {np.exp(lp[ok]).mean():.3f}, corr with target lp1 {c:.2f}")
    assert ok.mean() > 0.9 and 0.4 < h.mean() < 0.995 and c > 0.1, "drafter numbers implausible"
ti, mi = z["exact__tgt_top_ids"], z["exact__mtp_top_ids"]
print(f"smoke top-16: target {ti.shape} drafter {mi.shape}, drafter top-1 = target top-1 {np.mean(ti[1:, 0] == mi[1:, 0]):.3f}")
assert ti.shape[1] == 16 and mi.shape == ti.shape and "b20__tgt_top_ids" not in z.files, "top-16 arrays wrong"
PYEOF
echo "== full $(date +%H:%M:%S)"
TL_MTP=1 TL_VARIANT=plan4433333333 $PY sticky/budget125_tf.py 4 exact,b20,b22 tf_gsm,tf_ifeval,tf_code,tf_multi 4 2>&1 | grep --line-buffered -v "Token indices"
