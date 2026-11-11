"""M-25 SIDECAR SIZES (MoonShots, Mon 2026-11-02 23:2x; the orchestrator 23:2x; pre-registered in ~/NewIdeas/QUEUE.md M-25
before any run): the rank 0-1 LSB sidecar pool (Fable A, Fable C and cloud task 18: ~15.8 misses per kept token at every
C). Exact, residency only.
Replay (band45's pool rule: LRU + 30 x popularity, leave-one-window-out prior over the capture's 36 windows, strict; the
prompts as uncounted prefill) on Breakthrough's capture, chains per domain: code (12 windows), chat = IFEval (12) + multi
(12). Needs = the union per 16-row block (the uncut walk's checks): base pool top-10, sidecar top-2, per layer. Miss curves
of both pools over size grids per layer; costs in SSD MB per kept token (MSB 2,162,688 B, LSB 770,048 B per miss).
Arms at C 256 and C 320 (base pool held at the uniform split): (1) per-layer sidecar sizes at the same sidecar RAM
(Lagrangian + greedy fill, C_l >= 32 as M-17's FLOOR); (2) uniform sidecar at 1.5x / 2x slots (112 -> 168 / 224, +2.07 /
+4.14 GB); (3) per-layer sidecar sizes at 1.5x / 2x sidecar RAM; control: both pools allocated jointly (C_m >= 160,
C_l >= 32) at the uniform RAM + 0 / 2.07 / 4.14 GB. Reference: M-17's adopted sizes (laysz45_<C>.json). Allocation on one
half of a group's windows, scored on the other (2 folds by window).
Line (the orchestrator's): PASS = arm (1) cuts code SSD MB per kept token >= 5 % vs uniform at equal RAM; KILL < 2 %.
usage: [BigMoE venv] taskpolicy -b python sidecars45.py -> data/sidecars45.txt (under run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
CS = (256, 320)
UNI = {c: ((c * 2764800 - 112 * LR) // MR, 112) for c in CS}
M17 = {c: json.load(open(os.path.join(LZ.OUT, f"laysz45_{c}.json"))) for c in CS}
GM = sorted(set(range(64, 513, 8)) | {u[0] for u in UNI.values()} | {x for c in CS for x in M17[c]["C_m"]})
GL = sorted(set(range(0, 513, 8)) | {x for c in CS for x in M17[c]["C_l"]})
GROUPS = {"code": ("code",), "chat": ("ifeval", "multi")}
EXTRA = (1.0, 1.5, 2.0)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def needs16(z):
    ids = z["ids"]
    out = []
    for a, r, k in z["checks_walk"].tolist():
        rid = ids[a:a + r]
        out.append(([np.unique(rid[:, l]).astype(np.int16) for l in range(NL)],
                    [np.unique(rid[:, l, :2]).astype(np.int16) for l in range(NL)], k))
    return out


def replay_grid(task):
    """misses per window for every pool size K of one (chain, key, layer): [len(Ks), windows]"""
    key, Ks, calls, pops = task
    conv = [[(t, frozenset(n.tolist())) for t, n in cl] for cl in calls]
    return np.array([B.replay((key, K, conv, pops)) for K in Ks])


def alloc(costs, sizes, nbytes, budget, floors):
    """per-layer size indices for each pool (slot grids `sizes`, `nbytes` per slot) minimizing summed cost within budget
    bytes: Lagrangian, then greedy fill"""
    sz = [np.array(s, float) * b for s, b in zip(sizes, nbytes)]
    pen = [np.where(np.array(s) >= f, 0, np.inf) for s, f in zip(sizes, floors)]

    def pick(lam):
        ix = [np.argmin(c + lam * s[None, :] + p[None, :], 1) for c, s, p in zip(costs, sz, pen)]
        return ix, sum(s[i].sum() for s, i in zip(sz, ix))
    lo, hi = 0.0, 1e-6
    while pick(hi)[1] > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pick(mid)[1] > budget else (lo, mid)
    ix, used = pick(hi)
    ix = [i.copy() for i in ix]
    while True:
        best = None
        for q, (c, s, p) in enumerate(zip(costs, sz, pen)):
            for l in range(NL):
                for n in range(ix[q][l] + 1, len(s)):
                    d = s[n] - s[ix[q][l]]
                    if np.isfinite(p[n]) and used + d <= budget and c[l, n] < c[l, ix[q][l]]:
                        v = (c[l, ix[q][l]] - c[l, n]) / d
                        if best is None or v > best[0]:
                            best = (v, q, l, n, d)
        if best is None:
            break
        _, q, l, n, d = best
        ix[q][l] = n
        used += d
    return ix


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    NEED = [needs16(z) for z in Z]
    KEPT = np.array([sum(k for *_, k in nd) for nd in NEED], float)
    say(f"M-25 SIDECAR SIZES on Breakthrough's capture ({len(names)} windows, 16-row blocks): grids C_m {len(GM)} sizes, C_l "
        f"{len(GL)}; uniform {UNI}")
    CUR = {}                                                     # (key) -> misses [NL, grid, windows] (all chains)
    for dom in ("code", "ifeval", "multi"):
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        jobs, meta = [], []
        for key, Ks in (("m", GM), ("l", GL)):
            for l in range(NL):
                calls = [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] +
                         [(t, nd[0 if key == "m" else 1][l]) for t, nd in enumerate(NEED[i])] for i in ix]
                jobs.append((key, Ks, calls, [POP[key][i][l] for i in ix]))
                meta.append((key, l))
        for (key, l), res in zip(meta, pool.map(replay_grid, jobs, chunksize=1)):
            arr = CUR.setdefault(key, np.zeros((NL, len(GM if key == "m" else GL), len(names))))
            arr[l][:, ix] = res
        say(f"  {dom} curves done ({time.time() - t0:.0f} s)")
    pool.close(); pool.join()
    gi = {x: k for k, x in enumerate(GM)}
    gj = {x: k for k, x in enumerate(GL)}

    def score(win, im, jl):
        mb = sum(CUR["m"][l, im[l], win].sum() * MR + CUR["l"][l, jl[l], win].sum() * LR for l in range(NL)) / 1e6
        side = sum(CUR["l"][l, jl[l], win].sum() * LR for l in range(NL)) / 1e6
        return mb, side

    for cs in CS:
        um, ul = UNI[cs]
        base_gb = NL * (um * MR + ul * LR) / 1e9
        for grp, doms in GROUPS.items():
            win = [i for i, w in enumerate(names) if w.split("_")[1] in doms]
            folds = (win[0::2], win[1::2])
            arms = {}

            def fixed(im, jl):
                return [(np.array(im), np.array(jl))] * 2

            arms["uniform"] = fixed([gi[um]] * NL, [gj[ul]] * NL)
            arms["M-17 sizes"] = fixed([gi[x] for x in M17[cs]["C_m"]], [gj[x] for x in M17[cs]["C_l"]])
            for x in (1.5, 2.0):
                arms[f"sidecar x{x} uniform"] = fixed([gi[um]] * NL, [gj[int(ul * x)]] * NL)
            for x in EXTRA:
                per, joint = [], []
                for f in (0, 1):
                    tr = folds[f]
                    k = KEPT[tr].sum()
                    cm = CUR["m"][:, :, tr].sum(2) * MR / k
                    cl = CUR["l"][:, :, tr].sum(2) * LR / k
                    side_budget = NL * ul * LR * x
                    per.append((np.array([gi[um]] * NL), alloc([cl], [GL], [LR], side_budget, [32])[0]))
                    joint.append(tuple(alloc([cm, cl], [GM, GL], [MR, LR], NL * (um * MR + ul * LR) + NL * ul * LR * (x - 1), [160, 32])))
                arms[f"sidecar per-layer x{x}"] = per
                arms[f"joint +{(x - 1) * NL * ul * LR / 1e9:.2f} GB"] = joint
            say(f"  C {cs} ({grp}, {len(win)} windows; uniform pools {base_gb:.2f} GB): SSD MB per kept token (sidecar part), pool GB, vs uniform")
            ref = None
            for name, fs in arms.items():
                mb = side = 0.0
                gb = []
                for f in (0, 1):
                    te = folds[1 - f]                            # fold f's sizes score the other half
                    im, jl = fs[f]
                    a, b = score(te, im, jl)
                    mb += a; side += b
                    gb.append(sum(GM[i] * MR + GL[j] * LR for i, j in zip(im, jl)) / 1e9)
                v = mb / KEPT[win].sum()
                ref = v if name == "uniform" else ref
                say(f"    {name:24s} {v:6.2f} ({side / KEPT[win].sum():5.2f}) | {np.mean(gb):5.2f} GB | {100 * (v / ref - 1):+6.1f} %")
                if name == "sidecar per-layer x1.0" and grp == "code":
                    g = 1 - v / ref
                    say(f"    -> arm (1) at equal RAM, code: {100 * g:.1f} % cut -> " + ("PASS" if g >= 0.05 else "KILL" if g < 0.02 else "BETWEEN"))
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "sidecars45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
