"""M-18b LAYSZ-Q1 (MoonShots, Mon 2026-11-02; DAY9 #1 follow-up, QUEUE M-18, lines written before the run): re-fit M-17's
per-layer base / sidecar pool split for Q1's needs (top-2 on rows with DFlash2 cf >= 0.99; band45.py's rule), same C 256
pool bytes. Under Q1 the base needs shrink ~40 % while the rank 0-1 sidecar needs do not, so M-17's split (fitted on the
exact needs) may leave bytes in the wrong pool.
Cross-set held-out: fit on cap45 (Q1 0.99 with D-23 cf transplanted, seed 0; six prompts chained; band45's needs) and score
on Breakthrough's DAY9 capture (36 map windows, chained per domain, prefill chunks uncounted), and the reverse. Fit: M-17's
Lagrangian + greedy fill, FLOOR arm (C_m >= 160, C_l >= 32), on a 16-slot grid; the adopted M-17 sizes (laysz45_256.json)
and uniform 287 + 112 are scored by direct replays. Same pools and prior as band45 (LRU + 30 x popularity, fixed prior from
the EXACT needs, strict).
Lines: PASS = code SSD MB per kept token under Q1 0.99 <= -8 % vs M-17's sizes held-out in both directions; KILL = > -3 %
in both; BETWEEN otherwise. Also reported: what the Q1-fitted sizes cost the EXACT needs (one size table per mode).
usage: [BigMoE venv] taskpolicy -b python laysz_q1.py -> stdout + data/laysz_q1.txt, data/laysz_q1_256.json"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, multiprocessing as mp, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B

LZ, NL, MR, LR = B.LZ, B.NL, B.MR, B.LR
M17 = B.SIZES
UM, UL = 287, 112
BUDGET = NL * (UM * MR + UL * LR)
GM = sorted(set(range(160, 465, 16)) | {UM})
GL = sorted(set(range(32, 257, 16)) | {UL})
RULE = "Q1 0.99"
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay_grid(task):
    """one chain's misses per item for one layer's pool at every size in Ks: [len(Ks), items]"""
    key, Ks, calls, pops = task
    return np.array([B.replay((key, K, calls, pops)) for K in Ks])


def alloc(cm, cl, budget, mmin=160, lmin=32):
    """laysz45.alloc on given cost arrays (bytes per kept token per layer and grid size): Lagrangian, then greedy fill"""
    ok_m, ok_l = np.array(GM) >= mmin, np.array(GL) >= lmin
    rm, rl = np.array(GM, float) * MR, np.array(GL, float) * LR
    am, al = np.where(ok_m, 0, np.inf), np.where(ok_l, 0, np.inf)

    def pick(lam):
        i = np.argmin(cm + lam * rm[None, :] + am[None, :], 1)
        j = np.argmin(cl + lam * rl[None, :] + al[None, :], 1)
        return i, j, rm[i].sum() + rl[j].sum()
    lo, hi = 0.0, 1e-6
    while pick(hi)[2] > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[2] > budget:
            lo = mid
        else:
            hi = mid
    i, j, used = pick(hi)
    i, j = i.copy(), j.copy()
    while True:
        best = None
        for l in range(NL):
            for which, r, c, cur in (("m", rm, cm, i), ("l", rl, cl, j)):
                for n in range(cur[l] + 1, len(r)):
                    d = r[n] - r[cur[l]]
                    if used + d <= budget:
                        v = (c[l, cur[l]] - c[l, n]) / d
                        if v > 0 and (best is None or v > best[0]):
                            best = (v, which, l, n, d)
        if best is None:
            break
        _, which, l, n, d = best
        (i if which == "m" else j)[l] = n
        used += d
    return i, j, used


if __name__ == "__main__":
    if os.path.exists(LZ.PAUSE):
        sys.exit("PAUSE set: not running")
    t0 = time.time()
    P, _ = pickle.load(open(B.CACHE, "rb"))
    T = B.transplant(P, B.d23(), 0)
    W = B.load_capture(B.CAPDIR)
    SETS = {}                                             # name -> needs per rule, chains (item lists), calls(rule, key, l, chain), pops, kept, code items
    QA = {r: B.needs(P, T, B.RULES[r]) for r in ("EXACT", RULE)}
    PA = {k: LZ.popularity(QA["EXACT"], k) for k in ("m", "l")}
    SETS["cap45"] = dict(Q=QA, chains=[list(range(len(P)))], pop=PA,
                         calls=lambda r, key, l, wi: [([(-1, res0[l])] if key == "m" else []) + [(t, c[key][l]) for t, c in enumerate(cks)]
                                                      for _, cks, res0 in (QA[r][p] for p in wi)],
                         kept=np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float),
                         code=[p for p, (n, _, _) in enumerate(P) if n.startswith("humaneval")])
    QB, _, PREW = B.capture_needs(W, {r: ("conf", B.RULES[r]) for r in ("EXACT", RULE)})
    PB = {k: LZ.popularity(QB["EXACT"], k) for k in ("m", "l")}
    doms = sorted(set(w[0] for w in W))
    SETS["capture"] = dict(Q=QB, chains=[[i for i, w in enumerate(W) if w[0] == dm] for dm in doms], pop=PB,
                           calls=lambda r, key, l, wi: B.capture_calls(QB[r], PREW, wi, key, l),
                           kept=np.array([sum(c["kept"] for c in w[2]) for w in W], float),
                           code=[i for i, w in enumerate(W) if w[0] == "code"])
    say(f"M-18b LAYSZ-Q1: {RULE} needs, C 256 pools ({BUDGET / 1e9:.2f} GB), grid C_m {GM[0]}-{GM[-1]} ({len(GM)}), C_l {GL[0]}-"
        f"{GL[-1]} ({len(GL)}); cap45 {len(P)} prompts (seed 0), capture {len(W)} windows; setup {time.time() - t0:.0f} s")

    def run(spec):
        """spec: [(set, rule, key, l, Ks)] -> {(set, rule, key, l): misses [len(Ks), items of the set]}"""
        tasks, meta = [], []
        for s, r, key, l, Ks in spec:
            S = SETS[s]
            for wi in S["chains"]:
                tasks.append((key, Ks, S["calls"](r, key, l, wi), [S["pop"][key][i][l] for i in wi]))
                meta.append((s, r, key, l, wi, len(Ks)))
        with mp.get_context("fork").Pool(4) as pool:
            res = pool.map(replay_grid, tasks, chunksize=1)
        out = {}
        for (s, r, key, l, wi, nk), m in zip(meta, res):
            a = out.setdefault((s, r, key, l), np.zeros((nk, len(SETS[s]["kept"]))))
            a[:, wi] = m
        return out

    spec = [(s, RULE, key, l, (GM if key == "m" else GL) + [M17["C_m" if key == "m" else "C_l"][l]])
            for s in SETS for l in range(NL) for key in ("m", "l")]
    spec += [(s, "EXACT", key, l, [M17["C_m" if key == "m" else "C_l"][l]]) for s in SETS for l in range(NL) for key in ("m", "l")]
    R = run(spec)
    say(f"  curves done {time.time() - t0:.0f} s")

    def mb(s, r, i=None, j=None, items=None, direct=False):
        """SSD MB per kept token on a set's items: grid indices i / j per layer, or direct = M-17's sizes (last column)"""
        S = SETS[s]
        it = list(range(len(S["kept"]))) if items is None else items
        tot = 0.0
        for l in range(NL):
            mm, ll = R[(s, r, "m", l)], R[(s, r, "l", l)]
            tot += (mm[-1 if direct else i[l], it].sum() * MR + ll[-1 if direct else j[l], it].sum() * LR)
        return tot / S["kept"][it].sum() / 1e6

    def costs(s):
        S = SETS[s]
        k = S["kept"].sum()
        cm = np.array([R[(s, RULE, "m", l)][:len(GM)].sum(1) * MR / k for l in range(NL)])
        cl = np.array([R[(s, RULE, "l", l)][:len(GL)].sum(1) * LR / k for l in range(NL)])
        return cm, cl
    FIT = {}
    for s in SETS:
        i, j, used = alloc(*costs(s), BUDGET)
        FIT[s] = (i, j)
        cmv, clv = np.array(GM)[i], np.array(GL)[j]
        say(f"  fit on {s}: pools {used / 1e9:.2f} GB; C_m {cmv.min()}-{cmv.max()} (median {np.median(cmv):.0f}), C_l {clv.min()}-"
            f"{clv.max()} (median {np.median(clv):.0f}); MSB share {100 * (cmv * MR).sum() / used:.1f} % (M-17 "
            f"{100 * (np.array(M17['C_m']) * MR).sum() / ((np.array(M17['C_m']) * MR).sum() + (np.array(M17['C_l']) * LR).sum()):.1f} %)")
        say(f"    C_m by layer: {' '.join(map(str, cmv))}")
        say(f"    C_l by layer: {' '.join(map(str, clv))}")
    ui, uj = np.full(NL, GM.index(UM)), np.full(NL, GL.index(UL))
    say("")
    say(f"  {RULE} needs, SSD MB per kept token (code / all): uniform 287 + 112 | M-17 sizes | fitted (in-sample or held-out)")
    verdict = []
    for s in SETS:
        code = SETS[s]["code"]
        for fit_on in SETS:
            i, j = FIT[fit_on]
            tag = "in-sample" if fit_on == s else "HELD-OUT"
            m17c, m17a = mb(s, RULE, items=code, direct=True), mb(s, RULE, direct=True)
            fc, fa = mb(s, RULE, i, j, items=code), mb(s, RULE, i, j)
            say(f"    on {s:<7} fit on {fit_on:<7} ({tag:<9}): uniform {mb(s, RULE, ui, uj, items=code):5.1f} / {mb(s, RULE, ui, uj):5.1f} | "
                f"M-17 {m17c:5.1f} / {m17a:5.1f} | fitted {fc:5.1f} / {fa:5.1f} ({100 * (fc / m17c - 1):+.1f} % / {100 * (fa / m17a - 1):+.1f} % vs M-17)")
            if fit_on != s:
                verdict.append(fc / m17c - 1)
    call = "PASS" if all(v <= -0.08 for v in verdict) else ("KILL" if all(v > -0.03 for v in verdict) else "BETWEEN")
    say(f"  Lines: code held-out vs M-17 {', '.join(f'{100 * v:+.1f} %' for v in verdict)} -> {call} (PASS <= -8 % both ways, KILL > -3 % both)")
    # what the Q1-fitted sizes cost the EXACT needs (direct replays at the fitted sizes)
    spec2 = [(s, "EXACT", key, l, [(GM if key == "m" else GL)[FIT[f][0 if key == "m" else 1][l]] for f in SETS])
             for s in SETS for l in range(NL) for key in ("m", "l")]
    R2 = run(spec2)
    for s in SETS:
        S = SETS[s]
        for fi, f in enumerate(SETS):
            for items, lab in ((S["code"], "code"), (list(range(len(S["kept"]))), "all")):
                ex = sum(R2[(s, "EXACT", "m", l)][fi, items].sum() * MR + R2[(s, "EXACT", "l", l)][fi, items].sum() * LR
                         for l in range(NL)) / S["kept"][items].sum() / 1e6
                say(f"    EXACT needs on {s} ({lab}) with sizes fitted on {f}: {ex:.1f} MB per kept token vs M-17 "
                    f"{mb(s, 'EXACT', items=items, direct=True):.1f} ({100 * (ex / mb(s, 'EXACT', items=items, direct=True) - 1):+.1f} %)")
    i, j = FIT["capture"]
    json.dump({"C_m": [int(GM[x]) for x in i], "C_l": [int(GL[x]) for x in j], "budget_bytes": int(BUDGET),
               "fit_on": "capture (Breakthrough DAY9, 36 windows, Q1 0.99 realizable gate)", "rule": RULE,
               "floor": [160, 32], "slot_bytes": [MR, LR], "held_out_code_vs_m17": verdict,
               "cap45_fit": {"C_m": [int(GM[x]) for x in FIT["cap45"][0]], "C_l": [int(GL[x]) for x in FIT["cap45"][1]]}},
              open(os.path.join(LZ.OUT, "laysz_q1_256.json"), "w"), indent=1)
    say(f"  wrote data/laysz_q1_256.json (fit on the capture; the cap45 fit inside); total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "laysz_q1.txt"), "w").write("\n".join(lines) + "\n")
