"""Per-part gate screen for teacher-forced KLD probes (Breakthrough 2026-10-21): every data/tfvar/<VAR>[_Sgc]_e8.npz
(tf_frugal_var.py, NewIdeas' tf_defer125.py), mean KLD per corpus part, a one-sided 90 % upper bound (bootstrap over
positions: optimistic, positions within a sequence are correlated), top-1 agreement, and PASS/FAIL against the official
per-part targets (1.10 x ar4's teacher-forced KLD: code 0.0025, gsm 0.0045, ifeval 0.0204, multi 0.0206; the subsets are
every 8th sequence, so this is a screen, not the Eval suite's verdict). For deferral variants (d1_* / dh_*) it also
prints f = (KLD - exact) / (drop - exact) per part for W = 2, drop = noren (wait ranks 0-1, use resident ranks of the
top-10, drop the rest, no renorm: exactly what dh_w2 / d1_w2 add late); other W need their own drop row (w4_nr ...).
usage: gate_table.py [VAR ...]   (default: every variant with at least one subset)"""
import glob, os, sys
import numpy as np

D = os.path.expanduser("~/Breakthrough/data/tfvar")
TARGET = {"tf_code": 0.0025, "tf_gsm": 0.0045, "tf_ifeval": 0.0204, "tf_multi": 0.0206}
PARTS = ["tf_gsm", "tf_code", "tf_ifeval", "tf_multi"]
rng = np.random.default_rng(0)
runs = {}
for f in sorted(glob.glob(f"{D}/*_e8.npz")):
    v = os.path.basename(f)[:-len("_e8.npz")]
    base = v[:-4] if v.endswith("_Sgc") else v
    z = np.load(f)
    for k in z.files:
        if k.startswith("kld_"):
            runs.setdefault(base, {})[k[4:]] = (z[k], z["agree_" + k[4:]])
want = sys.argv[1:] or sorted(runs)
ex = runs.get("exact", {})


def ub90(a):
    return np.percentile([a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)], 90)


print(f"{'variant':22s} " + " ".join(f"{p[3:]:>24s}" for p in PARTS) + "   verdict")
for v in want:
    r = runs.get(v)
    if r is None:
        continue
    cells, ok, have = [], True, 0
    for p in PARTS:
        if p not in r:
            cells.append(f"{'-':>24s}"); continue
        k, a = r[p]
        have += 1
        u = ub90(k)
        good = u <= TARGET[p]
        ok &= good
        cells.append(f"{k.mean():.4f} ub {u:.4f} {'ok' if good else 'X'} t{a.mean():.3f}".rjust(24))
    verdict = ("PASS" if ok else "FAIL") + ("" if have == 4 else f" ({have}/4 parts)")
    print(f"{v:22s} " + " ".join(cells) + f"   {verdict}")
    if (v.startswith("d1_w2") or v.startswith("dh_w2")) and "noren" in runs:
        fs = []
        for p in PARTS:
            if p in r and p in runs["noren"] and p in ex:
                fs.append(f"{p[3:]} {(r[p][0].mean() - ex[p][0].mean()) / (runs['noren'][p][0].mean() - ex[p][0].mean()):.2f}")
        if fs:
            print(f"{'':22s} f (share of noren's excess left): " + ", ".join(fs))
