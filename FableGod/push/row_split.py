"""row_split.py (Fable B, PUSH-B): what one verified row costs at 45 GB, by row position. CPU replay of BigMoE's 45 GB
capture (~/BigMoE/data/cap45: six prompts chained, every check's 16 rows incl. the rejected ones, rank-sorted top-10 per
row and layer from the real 8-bit router; loader = MoonShots' laysz45.load) through the adopted C 256 per-layer pools
(~/MoonShots/data/laysz45_256.json: C_m MSB-3 base slots + C_l LSB sidecar slots per layer, M-17), LRU + 30 x popularity
(leave-one-prompt-out, laysz45's live rule), strict pools, res0 forced per prompt, prompts 2-6 counted (M-17's verdict arm).
New here: every miss is charged to the FIRST row (in row order) of its check that needs the expert, with the rank it is
needed at and whether that row is kept (j < kept) or rejected (j >= kept); the union growth per row position comes from
the same pass. SSD ms per miss: 0.149 (FIXED-PART's demand slope, C 224) and today's busy 42.0 ms / 171 misses = 0.246
(C 256, the push brief). Run: nice -n 15 ~/BigMoE/.venv/bin/python row_split.py > row_split.out  (~3 min, < 3 GB)."""
import os, sys, json, heapq, time
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import laysz45 as L

t0 = time.time()
P, agree = L.load()                                   # [(name, checks[{rid: {l: int16 [rows, 10]}, kept}], res0[l])]
print("cap45 loaded in %.0f s; router self-check agreement %.3f; checks per prompt: %s" % (
    time.time() - t0, agree, ", ".join("%s %d" % (n.split("_")[0][:8] + n.split("_")[-1], len(c)) for n, c, _ in P)))
S = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))
CM, CL = S["C_m"], S["C_l"]
NL, R4, MR, LR = L.NL, L.R4, L.MR, L.LR
Q = L.needs(P, lambda c: 16)
POP = {"m": L.popularity(Q, "m"), "l": L.popularity(Q, "l")}
B = 30.0
ROWS = 16


def replay(l, key):
    """one layer, one pool: per (prompt, check): misses attributed to [row j, rank group, kept/rejected]; union growth."""
    K = CM[l] if key == "m" else CL[l]
    st, i = {}, 0
    out = []
    for p, (name, cks, res0) in enumerate(P):
        pop = POP[key][p][l]
        if key == "m":                                # res0 forced into the base pool at the prompt start
            i += 1
            for e in res0[l]:
                st[e] = float(i)
        for t, c in enumerate(cks):
            rid = c["rid"][l]
            rows, kept = rid.shape[0], int(c["kept"])
            i += 1
            seen = set()
            miss = np.zeros((ROWS, 2, 2))             # [row, rank 0-1 / 2-9, kept / rejected]
            new = np.zeros(ROWS)
            need = set()
            for j in range(rows):
                rr = rid[j][:R4] if key == "l" else rid[j]
                for r, e in enumerate(rr):
                    e = int(e)
                    if e in seen:
                        continue
                    seen.add(e); need.add(e)
                    new[j] += 1
                    if e not in st:
                        miss[j, 0 if r < R4 else 1, 0 if j < kept else 1] += 1
            out.append((p, t, miss, new, rows, kept))
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + B * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
    return out


# ---------------------------------------------------------------- replay all layers, both pools
t1 = time.time()
RES = {"m": [replay(l, "m") for l in range(NL)], "l": [replay(l, "l") for l in range(NL)]}
print("replay %.0f s" % (time.time() - t1))
# aggregate per check: sum over layers
N = len(RES["m"][0])
chk = [RES["m"][0][k][:2] + RES["m"][0][k][4:] for k in range(N)]       # (p, t, rows, kept)
missM = np.zeros((N, ROWS, 2, 2)); missL = np.zeros((N, ROWS, 2, 2)); newM = np.zeros((N, ROWS)); newL = np.zeros((N, ROWS))
for l in range(NL):
    for k in range(N):
        missM[k] += RES["m"][l][k][2]; newM[k] += RES["m"][l][k][3]
        missL[k] += RES["l"][l][k][2]; newL[k] += RES["l"][l][k][3]
names = [n for n, _, _ in P]
dom_of = lambda p: "code" if "humaneval" in names[p] else "math"
counted = [k for k in range(N) if chk[k][0] >= 1]                        # prompts 2-6 (M-17's rule); prompt 1 = cold start
print("\ncounted checks %d (prompts 2-6); prompt 1 (%s) uncounted" % (len(counted), names[0]))

MS_MISS = (0.149, 0.246)
for dom in ("code", "math", "all"):
    ks = [k for k in counted if dom == "all" or dom_of(chk[k][0]) == dom]
    kept = np.array([chk[k][3] for k in ks]); rows = np.array([chk[k][2] for k in ks])
    mM, mL, nM = missM[ks], missL[ks], newM[ks]
    totM, totL = mM.sum((1, 2, 3)), mL.sum((1, 2, 3))
    mb = (totM * MR + totL * LR) / 1e6
    print("\n==== %s: %d checks, rows %.2f, kept %.2f per check; misses per check MSB %.1f + LSB %.1f = %.1f; %.1f MB per check, "
          "%.1f MB per kept token; SSD per check %.1f / %.1f ms at %.3f / %.3f ms per miss [replay C 256 LAYSZ P30] ===="
          % (dom, len(ks), rows.mean(), kept.mean(), totM.mean(), totL.mean(), (totM + totL).mean(), mb.mean(),
             mb.sum() / kept.sum(), (totM + totL).mean() * MS_MISS[0], (totM + totL).mean() * MS_MISS[1], *MS_MISS))
    print("  by rank: MSB misses at ranks 0-1 %.1f (%.0f %%), ranks 2-9 %.1f; LSB (ranks 0-1 only) %.1f; rejected rows (j >= kept) own %.1f of %.1f misses (%.0f %%)"
          % (mM[:, :, 0, :].sum((1, 2)).mean(), 100 * mM[:, :, 0, :].sum() / max(mM.sum(), 1), mM[:, :, 1, :].sum((1, 2)).mean(),
             mL.sum((1, 2, 3)).mean(), (mM[:, :, :, 1].sum() + mL[:, :, :, 1].sum()) / len(ks), (totM + totL).mean(),
             100 * (mM[:, :, :, 1].sum() + mL[:, :, :, 1].sum()) / max((totM + totL).sum(), 1)))
    print("  %3s | %8s | %8s | %8s | %8s | %7s | %9s | %9s | %6s" % ("row", "new/lay", "MSB miss", "LSB miss", "MB", "ms@.149", "ms@.246", "cum ms", "n"))
    cum = 0.0
    for j in range(ROWS):
        has = rows > j
        if has.sum() == 0:
            continue
        nm, mm, ml = nM[has, j].mean() / NL, mM[has, j].sum((1, 2)).mean(), mL[has, j].sum((1, 2)).mean()
        mbj = (mm * MR + ml * LR) / 1e6
        cum += (mm + ml) * MS_MISS[0]
        print("  %3d | %8.2f | %8.1f | %8.1f | %8.1f | %7.2f | %9.2f | %9.1f | %6d" % (j, nm, mm, ml, mbj, (mm + ml) * MS_MISS[0], (mm + ml) * MS_MISS[1], cum, has.sum()))
    # the per-row averages over rows 1-15 (row 0 carries the check's base union)
    late = np.array([(mM[:, j].sum((1, 2)) + mL[:, j].sum((1, 2)))[rows > j].mean() for j in range(1, ROWS)])
    nl = np.array([nM[rows > j, j].mean() / NL for j in range(1, ROWS)])
    print("  rows 1-15 mean: new experts per layer %.2f, misses %.1f per row (%.2f ms at 0.149, %.2f at 0.246); row 0: new %.1f per layer, misses %.1f"
          % (nl.mean(), late.mean(), late.mean() * MS_MISS[0], late.mean() * MS_MISS[1], nM[:, 0].mean() / NL, (mM[:, 0].sum((1, 2)) + mL[:, 0].sum((1, 2))).mean()))
    # a selection-only row: its own misses skipped (resident-only) -> SSD 0; its experts: all-10 resident, or top-2
    # what the exact pass of the SAME rows would still miss if the cheap pass requested its misses one cycle early = all of them
    # (the cheap pass sees the same routes up to drift); report the rank 0-1 share as the top-2 cheap pass's request coverage
    print("  a top-2 cheap pass requests ranks 0-1 only: it names %.0f %% of this domain's misses (MSB ranks 0-1 + all LSB); an all-resident cheap pass names them all (route drift aside)"
          % (100 * (mM[:, :, 0, :].sum() + mL.sum()) / max((totM + totL).sum(), 1)))
print("\ndone %.0f s" % (time.time() - t0))
