"""Paired dh_w4 - v3 control on the denser re-check (tf_dense.py npz, same sequences and positions in both arms): per set the
two means, the paired difference with 95 % bootstrap CIs by position (optimistic, as the gating table) and by sequence
(clusters: positions within a sequence are correlated), and dh_w4's own mean upper bound against the 1.10 x ar4 target.
usage: tf_dense_cmp.py CTL_NAME DH_NAME [suffix=e4o0]"""
import os, sys
import numpy as np
D = os.path.expanduser("~/MoonShots/data/defer/tfb12")
sfx = sys.argv[3] if len(sys.argv) > 3 else "e4o0"
a, b = (np.load(os.path.join(D, f"{n}_{sfx}.npz")) for n in sys.argv[1:3])
TARGET = {"tf_code": 0.00253, "tf_gsm": 0.00449, "tf_ifeval": 0.0204, "tf_multi": 0.0206}
rng = np.random.default_rng(0)
for k in sorted(x for x in a.files if x.startswith("kld_")):
    s = k[4:]
    ka, kb, seq = a[k], b[k], a[f"seq_{s}"]
    assert ka.shape == kb.shape and (seq == b[f"seq_{s}"]).all(), "arms do not pair"
    d = kb - ka
    us = np.unique(seq)
    idx = [np.nonzero(seq == u)[0] for u in us]
    bp = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    bs, bm = [], []
    for _ in range(4000):
        pick = np.concatenate([idx[j] for j in rng.integers(0, len(us), len(us))])
        bs.append(d[pick].mean()); bm.append(kb[pick].mean())
    t = TARGET.get(s, float("nan"))
    print(f"{s}: {len(us)} seqs, {len(d)} positions | control {ka.mean():.5f}, dh_w4 {kb.mean():.5f} ({100 * kb.mean() / t:.0f} % "
          f"of target {t}) | paired diff {d.mean():+.5f} [{np.percentile(bp, 2.5):+.5f}, {np.percentile(bp, 97.5):+.5f}] by "
          f"position, [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}] by sequence | dh_w4 mean 95 % upper: "
          f"control + diff (position CI) {ka.mean() + np.percentile(bp, 97.5):.5f}, by-sequence bootstrap of the mean "
          f"{np.percentile(bm, 97.5):.5f} -> {'PASS' if ka.mean() + np.percentile(bp, 97.5) <= t else 'TIGHT/OVER'}")
