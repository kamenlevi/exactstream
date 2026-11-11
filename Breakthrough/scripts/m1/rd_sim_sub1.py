#!/usr/bin/env python3
"""Sub-1-bit expert copies in the 21 Oct rate-distortion sim (Breakthrough, Wed 11 Nov 2026 18:1x; Kamen's idea 17:5x via
the orchestrator; pre-registered in docs/SUB1BIT_RD_SCREEN.md at 5e296e7). [P: a CPU sim with the energy law]
The model is scripts/rd_sim.py's, unchanged: 125B router traces (data/expA/traces125, 2,926 decode tokens), LRU of C = 51
whole 4-bit experts per layer, one token per step; ranks 0-1 at 4 bits (read if missing, cached); every missing rank
r >= 2 takes the option minimising err * w^2 + lam * MB; E = mean summed err per token-layer; KLD ~ 0.0043 + 0.0051
(E / 0.0032)^1.5. New: a dedicated 0.5-bit copy (q05) with its NMSE bracketed, the 1.0-bit tier tied to it by one
quality factor, and a variant that opens the copies to ranks 0-1. The line: KLD 0.0096 (E* = 0.00328), lam by bisection.
Speed: the option choice does not depend on the cache, so it is vectorised per lam; the loop only keeps the LRU and marks
misses (the same order of operations as rd_sim.py, so its rows are reproduced exactly: the self-check).
Output: ~/Breakthrough/data/m1/rd_sim_sub1.out. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B rd_sim_sub1.py
"""
import glob
import os
import sys
import time
from collections import OrderedDict

import numpy as np

NL, C, NEXP = 48, 51, 512 * 48
BASE = [("drop", 1.0, 0.0), ("msb2", 0.10, 1.536), ("msb3", 0.02, 2.150), ("full", 0.0, 2.765)]
TREL = [("tr2", 0.071, 1.30), ("tr1", 0.27, 0.70)]
KLD0, KA, E0 = 0.0043, 0.0051, 0.0032
LINE = 0.0096
ESTAR = E0 * ((LINE - KLD0) / KA) ** (2 / 3)
Q05_MB, Q05_MB_BEST = 0.39, 0.31
t0 = time.time()
out = []


def say(s=""):
    print(s, flush=True)
    out.append(s)


def kld(E):
    return KLD0 + KA * (E / E0) ** 1.5


files = sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz")))
sets = {f.split("/")[-1].split("_")[0]: np.load(f) for f in files}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}    # float32, as rd_sim.py
W64 = {k: v.astype(np.float64) for k, v in w.items()}                                # float(wt[t, l, r]) there
IDL = {k: [ids[k][:, l, :10].tolist() for l in range(NL)] for k in ids}
NTOK = sum(v.shape[0] for v in ids.values())
NTL = NTOK * NL
NPASS = 0


def run(opts, lam, any_rank=False):
    """One pass: returns E per token-layer, MB per token, reads per token by option."""
    global NPASS
    NPASS += 1
    err = np.array([o[1] for o in opts]); size = np.array([o[2] for o in opts])
    full = [o[0] for o in opts].index("full")
    E = mb = 0.0
    cnt = np.zeros(len(opts))
    for k in ids:
        Wk = W64[k]
        cost = np.stack([(e * Wk) * Wk + lam * s for e, s in zip(err, size)], -1)    # err * ww * ww + lam * size
        opt = cost.argmin(-1)                                                       # first minimum, as np.argmin
        if not any_rank:
            opt[:, :, :2] = full
        isfull = opt == full
        miss = np.empty(opt.shape, bool)
        T = opt.shape[0]
        for l in range(NL):
            c = OrderedDict()
            mte, pop = c.move_to_end, c.popitem
            rows, frows = IDL[k][l], isfull[:, l, :].tolist()
            ml = []
            app = ml.append
            for t in range(T):
                row, frow = rows[t], frows[t]
                for r in range(10):
                    e = row[r]
                    if e in c:
                        mte(e)
                        app(False)
                    else:
                        app(True)
                        if frow[r]:
                            c[e] = 1
                while len(c) > C:
                    pop(last=False)
            miss[:, l, :] = np.array(ml, bool).reshape(T, 10)
        o = opt[miss]
        E += float(((err[o] * Wk[miss]) * Wk[miss]).sum())
        mb += float(size[o].sum())
        cnt += np.bincount(o, minlength=len(opts))
    return E / NTL, mb / NTOK, cnt / NTOK


def at_line(opts, any_rank=False):
    """Bisection on log lam so that E = E*; MB interpolated linearly in E between the last bracket."""
    lo, hi = 1e-5, 3e-2
    rlo, rhi = run(opts, lo, any_rank), run(opts, hi, any_rank)
    if not (rlo[0] <= ESTAR <= rhi[0]):
        return None
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        r = run(opts, mid, any_rank)
        if r[0] <= ESTAR:
            lo, rlo = mid, r
        else:
            hi, rhi = mid, r
        if abs(rlo[0] / ESTAR - 1) < 0.002 or abs(rhi[0] / ESTAR - 1) < 0.002:
            break
    best = rlo if abs(rlo[0] / ESTAR - 1) <= abs(rhi[0] / ESTAR - 1) else rhi
    f = (ESTAR - rlo[0]) / (rhi[0] - rlo[0]) if rhi[0] > rlo[0] else 0.0
    mb = rlo[1] + f * (rhi[1] - rlo[1])
    return dict(lam=(lo * hi) ** 0.5, mb=mb, E=best[0], cnt=best[2])


def extra_ssd(opts, cnt):
    """GB of dedicated copies the mix uses (any copy tier read at least 0.05 times per token), all 24,576 experts."""
    return sum(s * NEXP / 1000 for (n, _, s), c in zip(opts, cnt) if n not in ("drop", "msb2", "msb3", "full") and c >= 0.05)


def fmt_cnt(opts, cnt):
    return " ".join(f"{n} {c:.1f}" for (n, _, _), c in zip(opts, cnt) if c >= 0.05)


say(f"Sub-1-bit copies in the 21 Oct RD sim: C = {C}, {NTOK} tokens x {NL} layers; line KLD {LINE} -> E* = {ESTAR:.5f}")
say("")
say("SELF-CHECK vs data/gmass/rd_sim_c51*.txt (MB/token to 0.1, E to 0.0001, reads per token to 0.1):")
REF = {("A", 1e-3): "361.8 0.0017 drop 6.8 msb2 135.2 msb3 46.2 full 19.8",
       ("A", 2e-3): "282.8 0.0041 drop 41.8 msb2 141.5 msb3 5.0 full 19.7",
       ("B", 1e-3): "282.7 0.0019 drop 1.4 msb2 0.0 msb3 3.4 full 19.8 tr2 153.6 tr1 29.8",
       ("B", 2e-3): "225.4 0.0037 drop 9.4 msb2 0.0 msb3 0.1 full 19.7 tr2 75.8 tr1 103.1"}
ok = True
for (mix, lam), ref in REF.items():
    opts = BASE if mix == "A" else BASE + TREL
    E, mb, cnt = run(opts, lam)
    got = f"{mb:.1f} {E:.4f} " + " ".join(f"{n} {c:.1f}" for (n, _, _), c in zip(opts, cnt))
    good = got == ref
    ok = ok and good
    say(f"  {mix} lam {lam:.0e}: {'PASS' if good else 'FAIL'}  got {got}" + ("" if good else f"  | ref {ref}"))
rows = []


def line_row(label, opts, nmse=None, any_rank=False, mixname=""):
    r = at_line(opts, any_rank)
    if r is None:
        say(f"| {label} | (the line is outside the lam range) |")
        return None
    ssd = extra_ssd(opts, r["cnt"])
    say(f"| {label} | {r['mb']:.1f} | {r['lam']:.2e} | {r['E']:.5f} ({kld(r['E']):.4f}) | {fmt_cnt(opts, r['cnt'])} | "
        f"{ssd:.1f} |")
    rows.append(dict(mix=mixname, nmse=nmse, mb=r["mb"], ssd=ssd, label=label))
    return r


HEAD = "| mix | MB/token at the line | lam | E (KLD) | reads per token by option | extra SSD GB |\n|---|---|---|---|---|---|"
say("")
say("BASELINES at the line:")
say(HEAD)
ra = line_row("A: drop / MSB-2 / MSB-3 / 4-bit", BASE, mixname="A")
rb = line_row("B: A + trellis tr2 / tr1 (the 245 baseline)", BASE + TREL, mixname="B")
if rb is not None:
    good = abs(rb["mb"] / 245 - 1) <= 0.05
    ok = ok and good
    say(f"B at the line {rb['mb']:.1f} MB vs 245 +- 5 %: {'PASS' if good else 'FAIL'}")
say(f"SELF-CHECK {'PASS' if ok else 'FAIL'}")
if not ok:
    open(os.path.expanduser("~/Breakthrough/data/m1/rd_sim_sub1.out"), "w").write("\n".join(out) + "\n")
    sys.exit(1)

BRACKET = [1.0, 0.8, 0.7, 0.61, 0.54, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]   # 0.2-0.0: hypothetical, to locate the 150 line
for mix, base, title in (("C", BASE, "C: A + q05 (0.5-bit copy, 0.39 MB)"),
                         ("D", BASE + TREL, "D: B + q05 (the full ladder)")):
    say("")
    say(f"{title}, by q05's NMSE:")
    say(HEAD)
    for n in BRACKET:
        line_row(f"{mix} q05 NMSE {n:.2f}", base + [("q05", n, Q05_MB)], nmse=n, mixname=mix)
say("")
say("E: A + q05 + the 1.0-bit tier (0.70 MB), one quality factor k: NMSE = min(1, k 2^(-2R)) (k 1.08 = tr1's 0.27):")
say(HEAD)
for k in (1.0, 1.08, 1.22, 1.6, 2.0):
    n05, n10 = min(1.0, k * 0.5), min(1.0, k * 0.25)
    line_row(f"E k {k:.2f} (q05 {n05:.2f}, q10 {n10:.2f})", BASE + [("q05", n05, Q05_MB), ("q10", n10, 0.70)],
             nmse=n05, mixname="E")
say("")
say("Sensitivities at the plausible edge (q05 NMSE 0.54 and 0.50):")
say(HEAD)
for n in (0.54, 0.50):
    line_row(f"D, q05 at 0.31 MB (no scales), NMSE {n:.2f}", BASE + TREL + [("q05", n, Q05_MB_BEST)], nmse=n, mixname="D-best")
    line_row(f"F: D with ranks 0-1 open too, NMSE {n:.2f}", BASE + TREL + [("q05", n, Q05_MB)], nmse=n, any_rank=True,
             mixname="F")

say("")
plaus = [r for r in rows if r["nmse"] is not None and r["nmse"] >= 0.5 and r["mix"] in ("C", "D", "E", "F")]
best = min(plaus, key=lambda r: r["mb"])
bb = rb["mb"]
say(f"Best plausible mix (q05 NMSE >= 0.50, q05 at 0.39 MB): {best['label']}: {best['mb']:.1f} MB/token "
    f"({best['mb'] / bb - 1:+.1%} vs B's {bb:.1f}; {best['mb'] / 245 - 1:+.1%} vs the 245 of 21 Oct), extra SSD {best['ssd']:.1f} GB")
bp = [r for r in rows if r["mix"] == "D-best"]
if bp:
    b2 = min(bp, key=lambda r: r["mb"])
    say(f"Best case (q05 at 0.31 MB, no scales): {b2['label']}: {b2['mb']:.1f} MB/token")


def crossing(mix, target):
    pts = sorted([(r["nmse"], r["mb"]) for r in rows if r["mix"] == mix], reverse=True)    # NMSE falling, MB falling
    for (n1, m1), (n2, m2) in zip(pts, pts[1:]):
        if m1 > target >= m2:
            return n1 + (target - m1) / (m2 - m1) * (n2 - n1)
    return None


for mix in ("C", "D"):
    for target in (196, 150):
        x = crossing(mix, target)
        say(f"{mix}: q05 NMSE needed for <= {target} MB/token: " + (f"{x:.2f}" if x is not None else "not reached in the bracket"))
v = "PASS" if best["mb"] <= 150 else ("KILL" if best["mb"] > 196 else "BETWEEN")
say(f"VERDICT (pre-registered lines: PASS <= 150 MB at q05 NMSE >= 0.50; KILL if nothing <= 196 MB): {v}")
say(f"{NPASS} passes, {time.time() - t0:.0f} s")
open(os.path.expanduser("~/Breakthrough/data/m1/rd_sim_sub1.out"), "w").write("\n".join(out) + "\n")
