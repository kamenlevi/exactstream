"""M-07 LPOM, precision on miss for the LSB plane (MoonShots Thu 16:05; NON-exact, CPU). v3 reads the LSB sidecar
(720,896 B) on every LSB miss of an expert whose best rank in the check is 0 or 1. LP1: an LSB miss at best rank 1 is
skipped (that expert's uses run at MSB-3 with the suite's group-mean bias; no read, no insert); rank 0 still reads.
LP01: rank-0 misses are skipped too (the pool serves hits only). LSB pool C_l 44 per layer, strict LFU as v3 (decay 0.99).
MSB side from M-03 / M-04 (pomvs24.py): POM R2 4 with fixed slots (R-44's build) or with M-04's VD.
Quality: f_1 / f_0 = share of the rank-1 / rank-0 gate energy on accepted rows served without the LSB; predicted KLD =
v3 + f_MSB x gap(R2 4) + f_1 x (plan4333333333 - v3) + f_0 x (plan3333333333 - plan4333333333), M-02's measured plans.
PASS (written first): LP1 on top of POM R2 4 gives <= -6 % SSD bytes / token vs POM R2 4 alone on the narrow path with
predicted KLD inside 1.10 x ar4 on all four parts. KILL: < -3 %, or over the gate.
usage: [BigMoE venv] taskpolicy -b python lpom24.py -> stdout + data/lpom24.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pomvs24 as P
import wall24 as W

lines = []


def say(s):
    print(s, flush=True); lines.append(s)


p4333 = np.array([0.00245, 0.00399, 0.01799, 0.01553])
p3333 = np.array([0.00463, 0.00670, 0.02840, 0.02545])
GL1, GL0 = p4333 - P.v3, p3333 - p4333


def lsb_run(checks, mode, A):
    """one layer -> (LSB bytes over measured checks, [rank-1 energy without LSB, rank-1 energy, the same for rank 0])"""
    Rs, score, w = set(), {}, 1.0
    byt, q = 0, np.zeros(4)
    for (meas, best, ids, g) in checks:
        w /= P.DEC
        need = {e: k for e, k in best.items() if k < 2}
        for e in need:
            score[e] = score.get(e, 0.0) + w
        for e, k in need.items():
            if e in Rs or mode == "LP01" or (mode == "LP1" and k == 1):
                continue
            if len(Rs) >= P.CL:
                vic = [v for v in Rs if v not in need]
                if vic:
                    Rs.discard(min(vic, key=lambda v: score[v]))
            Rs.add(e)
            if meas:
                byt += P.LSB
        if meas:
            for r in range(A):
                e0, e1 = int(ids[r, 0]), int(ids[r, 1])
                q += (g[r, 1] ** 2 * (e1 not in Rs), g[r, 1] ** 2, g[r, 0] ** 2 * (e0 not in Rs), g[r, 0] ** 2)
        if len(Rs) > P.CL:
            Rs = set(sorted(Rs, key=lambda v: -score[v])[:P.CL])
    return byt, q


t0 = time.time()
for R, A in ((4, 3), (16, 10)):
    per = P.build(R, A)
    msb = {}
    for name, slot2, dg in (("POM R2 4", P.SLOT3, 0), ("POM R2 4 + VD", P.SLOT2V, 2)):
        b, rm = 0.0, np.zeros(4)
        for k, (lay, ntok, lb) in per.items():
            bb = 0
            for l in range(W.NL):
                bl, r_, _, _ = P.run(lay[l], 4, slot2, dg, A)
                bb += sum(bl); rm += r_
            b += bb / ntok / len(per)
        msb[name] = (b, rm[2] / rm[3])
    lsb, chk = {}, np.mean([lb / ntok for lay, ntok, lb in per.values()])
    for mode in ("v3", "LP1", "LP01"):
        b, q = 0.0, np.zeros(4)
        for k, (lay, ntok, lb) in per.items():
            bb = 0
            for l in range(W.NL):
                b_, q_ = lsb_run(lay[l], mode, A)
                bb += b_; q += q_
            b += bb / ntok / len(per)
        lsb[mode] = (b, q[0] / q[1], q[2] / q[3])
    say(f"== {R}-row checks accepting {A}: LSB bytes / token v3 {lsb['v3'][0] / 1e6:.2f} MB (build()'s v3 count "
        f"{chk / 1e6:.2f}), LP1 {lsb['LP1'][0] / 1e6:.2f}, LP01 {lsb['LP01'][0] / 1e6:.2f}; without the LSB: LP1 rank-1 "
        f"energy {100 * lsb['LP1'][1]:.1f} %, LP01 rank-1 {100 * lsb['LP01'][1]:.1f} % + rank-0 {100 * lsb['LP01'][2]:.1f} %")
    ref = msb["POM R2 4"][0] + lsb["v3"][0]
    for mn, (mb, fe) in msb.items():
        for mode in ("v3", "LP1", "LP01"):
            lbt, f1, f0 = lsb[mode]
            tot = mb + lbt
            pred = P.v3 + fe * P.gap[4] + f1 * GL1 + f0 * GL0
            ok = " INSIDE" if (pred <= P.tgt).all() else " OVER on " + ", ".join(
                n for n, p, t in zip(("code", "gsm", "ifeval", "multi"), pred, P.tgt) if p > t)
            say(f"  {mn:14s} + LSB {mode:4s}: {tot / 1e6:6.1f} MB / token ({100 * (tot / ref - 1):+5.1f} % vs POM R2 4); "
                f"predicted KLD " + " / ".join(f"{p:.5f}" for p in pred) + f"{ok} (code at {100 * pred[0] / P.tgt[0]:.0f} % "
                f"of the gate)")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/lpom24.txt"), "w").write("\n".join(lines) + "\n")
