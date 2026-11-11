"""QOM TAIL WIDTH 4 price (NewIdeas, Wed 4 Nov 2026 16:3x; CPU only). BigMoE's engine gets a flag for the width of
QOM's kept ranks 2-9: 3 = MSB-3 (no extra reads; today's byte prices), 4 = the kept experts' LSB parts are read too
(C's measured map quality: the map kept ranks 2-9 at 4 bits). The orchestrator: "s7_ref3 tonight decides; if 3-bit
fails, re-price the bytes at 4". This prices 4 ahead of the verdict.
Frame = trellis_draft17.py's: C's rule with every last row gated (DRAFT17 'all'), all rows load their top-2 at 3 bits
through an MSB pool of 11-bit packed parts (2,064,384 B). Width 4 adds: every kept rank 2-9 expert of every row in the
block (in the block's top-2 union, or resident in the MSB pool after the union's loads) needs its LSB part
(720,896 B), served by an LSB LRU pool of Cl slots or read. RAM per layer is fixed (the line's budget):
Cm = (B - Cl x 720,896) // 2,064,384. The prefill (3-bit in FAST) warms the MSB pool only; the LSB pool starts cold.
Kept experts do not refresh the MSB LRU (as in every byte replay here). Writes ~/NewIdeas/data/ledger/w4_price.txt."""
import os, sys, time
import multiprocessing as mp
from collections import OrderedDict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ten_x_ledger as T  # noqa: E402
import ledger_lastrow as LL  # noqa: E402
import pool_quality as PQ  # noqa: E402
MR11, LR11 = 2064384, 720896
BUDGET = {"24 GB": 147 * 2162688 + 46 * 770048, "12 GB": 51 * 2162688 + 32 * 770048}
CLS = {"24 GB": [0, 32, 64, 96, 128, 160, 200], "12 GB": [0, 16, 32, 48, 64, 80]}
REF_CM = {"24 GB": 147, "12 GB": 51}        # C's map arm ran at 147 MSB slots; 51 = the old 12 GB split (no 12 GB C arm)


def replay(task):
    dom, line, cl = task
    cm = (BUDGET[line] - cl * LR11) // MR11
    bp = [OrderedDict() for _ in range(48)]; lp = [OrderedDict() for _ in range(48)]
    mb = lb = kept = kpairs = 0
    for w in T.W[dom]:
        ids = w["ids"]
        for p in range(w["a"]):
            for l in range(48):
                for e in ids[p, l].tolist(): T.touch(bp[l], e, cm)
        for rows, cfs, kp in LL.blocks_conv(w, "all"):
            for l in range(48):
                U = set()
                for r in rows: U.update(ids[r, l, :2].tolist())
                mb += sum(1 for e in U if e not in bp[l])
                for e in U: T.touch(bp[l], e, cm)
                K = set()
                for r in rows:
                    for e in ids[r, l, 2:10].tolist():
                        if e in U or e in bp[l]: K.add(e)
                kpairs += len(K)
                if cl:
                    lb += sum(1 for e in K if e not in lp[l])
                    for e in K: T.touch(lp[l], e, cl)
                else:
                    lb += len(K)
            kept += kp
    return task, (cm, mb * MR11 / kept / 1e6, lb * LR11 / kept / 1e6, kpairs / kept / 48)


def keep(task):
    dom, c = task
    n, mem, res, wn, wmem, wres, ld, ms = PQ.run((dom, "lru", str(c)))[1]
    return ("k",) + task, 100 * (wmem + wres) / wn


def job(t):
    return replay(t) if t[0] != "k" else keep(t[1:])


def cm_of(ln, c):
    return (BUDGET[ln] - c * LR11) // MR11


def main():
    t0 = time.time(); T.W = T.windows()
    tasks = [(d, ln, c) for ln in BUDGET for c in CLS[ln] for d in T.W]
    slots = sorted({cm_of(ln, c) for ln in BUDGET for c in CLS[ln]} | {147}) + ["m"]
    tasks += [("k", d, c) for d in T.W for c in slots]
    with mp.get_context("fork").Pool(int(os.environ.get("PQ_PROCS", "4")), maxtasksperchild=1) as P:
        R = dict(P.map(job, tasks, chunksize=1))
    L = ["QOM TAIL WIDTH 4: SSD MB per kept token, C's rule + DRAFT17 (every row gated), 11-bit parts; width 3 = MSB "
         "misses only (Cl 0); width 4 = + LSB parts of every kept rank 2-9 expert (LSB pool Cl, equal RAM per layer)",
         "| line | Cl | Cm | code w4 (MSB + LSB) | IFEval w4 | multi w4 | kept experts per layer per kept token (code / IFEval / multi) |",
         "|---|---|---|---|---|---|---|"]
    for ln in BUDGET:
        for c in CLS[ln]:
            r = {d: R[(d, ln, c)] for d in T.W}
            cm = r["code"][0]
            cell = {d: f"{r[d][1] + r[d][2]:.1f} ({r[d][1]:.1f} + {r[d][2]:.1f})" for d in T.W}
            L.append(f"| {ln} | {c} | {cm}{' *' if cm < REF_CM[ln] else ''} | {cell['code']} | {cell['ifeval']} | "
                     f"{cell['multi']} | {r['code'][3]:.1f} / {r['ifeval'][3]:.1f} / {r['multi'][3]:.1f} |")
    L += ["", "| line | width 3 (Cl 0) code / IFEval / multi | best width 4 code / IFEval / multi (its Cl) | factor w4 / w3 |",
          "|---|---|---|---|"]
    for ln in BUDGET:
        w3 = {d: R[(d, ln, 0)][1] for d in T.W}
        best = {d: min(CLS[ln], key=lambda c: R[(d, ln, c)][1] + R[(d, ln, c)][2]) for d in T.W}
        w4 = {d: R[(d, ln, best[d])][1] + R[(d, ln, best[d])][2] for d in T.W}
        L.append(f"| {ln} | {w3['code']:.1f} / {w3['ifeval']:.1f} / {w3['multi']:.1f} | {w4['code']:.1f} / {w4['ifeval']:.1f} / "
                 f"{w4['multi']:.1f} ({best['code']} / {best['ifeval']} / {best['multi']}) | x{w4['code'] / w3['code']:.2f} / "
                 f"x{w4['ifeval'] / w3['ifeval']:.2f} / x{w4['multi'] / w3['multi']:.2f} |")
    L += ["", "QUALITY at each split (SWEEP.md's KL model; width 4 = the map's own C form, so no ranks 2-9 term): keep = LRU "
          "kept share (MASS) of sure-row rank 2-9 pairs at Cm (pool_quality, the engine rule's proxy); g = keep gap closed "
          "from 147 toward the 45 GB pool; code KL = 0.0013 + 0.0020 + (1 - g) x 0.0024 + DRAFT17 0.0006; flips = -0.078 + "
          "15.68 sqrt(KL) (line 1.34 %). Chat is over its lines in C's form at any pool (SWEEP: the chat barrier), so code only.",
          "| line | Cl | Cm | keep code / IFEval / multi | g code | code KL [G] | code flips [G] | code verdict |", "|---|---|---|---|---|---|---|---|"]
    k147, km = R[("k", "code", 147)], R[("k", "code", "m")]
    for ln in BUDGET:
        for c in CLS[ln]:
            cm = cm_of(ln, c); k = {d: R[("k", d, cm)] for d in T.W}
            g = (k["code"] - k147) / (km - k147); kl = 0.0013 + 0.0020 + (1 - g) * 0.0024 + 0.0006
            fl = -0.078 + 15.68 * kl ** 0.5
            vd = "PASS" if fl <= 1.34 else ("BETWEEN" if fl <= 2.01 else "KILL")
            L.append(f"| {ln} | {c} | {cm} | {k['code']:.1f} / {k['ifeval']:.1f} / {k['multi']:.1f} % | {g:+.2f}"
                     f"{' (extrapolated)' if g < -1 else ''} | {kl:.4f} | {fl:.2f} | {vd} |")
    L.append(f"(keep at 147: {k147:.1f} %, 45 GB pool: {km:.1f} %, code)")
    L.append(f"({time.time() - t0:.0f} s; * = fewer MSB slots than C's map measurement ({REF_CM['24 GB']} / {REF_CM['12 GB']}), "
             f"so at 24 GB the kept share there is below C's measured; check: 24 GB width 3 must equal trellis_draft17.txt's MSB-3 "
             f"11-bit row 19.5 / 14.3 / 10.6)")
    os.makedirs(T.DST, exist_ok=True)
    open(os.path.join(T.DST, "w4_price.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
