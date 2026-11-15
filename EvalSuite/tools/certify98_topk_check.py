"""CERTIFY98 amendment 5: can BigMoE's TOPK dump (scripts/dd_topk.py, 9f49e03) stand in for the cells file? CPU only.
  .venv/bin/python tools/certify98_topk_check.py <topk.npz> <cells.npz> [<topk.npz> <cells.npz> ...]
Per pair:
- fi / depth / right must be equal array for array;
- certify98.py's kept/32, kept/16 and walk-strict gate bands must be equal;
- TOPK's walk_start flags must equal certify98.py's decode walk (R0, cap 31) on every window.
Last line: `TOPK-CHECK: PASS | FAIL`; exit 0 / 1."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify98 as C                                    # noqa: E402


def starts(rt, cap=31):
    out, p = [], C.R0
    while p < rt.shape[0]:
        k = 0
        while k < cap and rt[p, k] == 1:
            k += 1
        out.append(p); p += k + 1
    return out


def pair(tp, cp):
    t, c, bad = np.load(tp), np.load(cp), []
    for k in ("fi", "depth", "right"):
        if len(t[k]) != len(c[k]) or not (t[k] == c[k]).all():
            bad.append(f"{k} differs")
    WT, WC = C.windows(tp), C.windows(cp)
    RT, RC = C.summary(WT), C.summary(WC)
    for k in ("kept_per_32", "kept_per_16"):
        if RT[k]["mean"] != RC[k]["mean"]:
            bad.append(f"{k} {RT[k]['mean']:.6f} vs {RC[k]['mean']:.6f}")
    bt, bc = RT["strict_walk"]["bands"], RC["strict_walk"]["bands"]
    if bt != bc:
        bad.append("walk-strict bands differ")
    dep, ws = t["depth"], t["walk_start"]
    st = np.flatnonzero((dep == 0) & np.append(True, dep[:-1] != 0)); en = np.append(st[1:], len(dep))
    agree = 0
    for (rt, _), a, b in zip(WT, st, en):
        d0 = np.flatnonzero(dep[a:b] == 0)                 # depth 0's cells, in rt's row order
        agree += {int(i) for i in np.flatnonzero(ws[a:b][d0])} == {i for i in starts(rt) if i < len(d0)}
    if agree != len(WT):
        bad.append(f"walk starts agree on {agree} of {len(WT)} windows")
    b0 = bt[0] if isinstance(bt, list) else next(iter(bt.values()))
    print(f"{os.path.basename(tp)} vs {os.path.basename(cp)}: {len(WT)} windows, {len(dep)} cells; kept/32 "
          f"{RT['kept_per_32']['mean']:.6f}; walk-strict band 0-2 {b0['acc']:.4f}; walk starts {agree}/{len(WT)}; "
          f"{'SAME' if not bad else 'DIFFERS: ' + '; '.join(bad)}")
    return not bad


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__); return 2
    ok = all([pair(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)])
    print(f"TOPK-CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
