"""M-25b JOINT SIZES (MoonShots, Mon 2026-11-02 23:3x; the orchestrator 23:3x after M-25; pre-registered in ~/NewIdeas/QUEUE.md
M-25b before any run): per-layer (C_m, C_l) from M-17's joint Lagrangian (+ greedy fill; FLOOR C_m >= 160, C_l >= 32; P30
pools = LRU + 30 x popularity, strict) at C 256 and C 320, at equal RAM and +2.07 / +4.14 GB, as files BigMoE loads like
M-17's (BIGMOE_TIER_SIZES). Exact, residency only.
Curves: cap45 (M-17's data: six prompts chained, res0 forced per prompt, prompts 2-6 counted, 16-row checks, popularity
leave-one-prompt-out; laysz45.load() cached in data/band45_cap45.pkl) and Breakthrough's capture (36 windows, 16-row
blocks of the uncut walk, chains per domain, prompts as uncounted prefill, leave-one-window-out popularity), per layer over
size grids of both pools. Fits: (i) cap45 only (at equal RAM this is M-17 again: an identity check; the extra-RAM sizes are
new); (ii) cap45 + the capture's training windows, pooled bytes over pooled kept tokens. Scored on the capture's held-out
windows (2 folds by window, both folds mixing the three domains). Emitted files: data/joint45_<C>_<extra MB>.json, fitted
on cap45 + all 36 windows.
Line (the orchestrator's): PASS if fit (ii) beats M-17's adopted sizes by >= 2 % code SSD MB per kept token at equal RAM
on the held-out windows; KILL < 1 %; BETWEEN otherwise.
usage: [BigMoE venv] taskpolicy -b python joint45.py -> data/joint45.txt + data/joint45_*.json (under run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, multiprocessing as mp, pickle, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import sidecars45 as SC

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
GM, GL, CS, UNI, M17 = SC.GM, SC.GL, SC.CS, SC.UNI, SC.M17
EXTRA = (0.0, 2.07e9, 4.14e9)
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def replay_grid_fs(task):
    key, Ks, calls, pops = task
    return np.array([B.replay((key, K, calls, pops)) for K in Ks])


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    # cap45 curves (M-17's protocol)
    P0 = pickle.load(open(B.CACHE, "rb"))[0]                      # band45 caches (P, router agreement)
    Q = LZ.needs(P0, lambda c: 16)
    POPc = {key: LZ.popularity(Q, key) for key in ("m", "l")}
    Kc = np.array([sum(c["kept"] for c in cks) for _, cks, _ in Q], float)
    H = list(LZ.H)
    jobs = [(key, GM if key == "m" else GL,
             [([(-1, res0[l])] if key == "m" else []) + [(t, c[key][l]) for t, c in enumerate(cks)] for _, cks, res0 in Q],
             [POPc[key][p][l] for p in range(len(Q))]) for key in ("m", "l") for l in range(NL)]
    CAP = {k: np.zeros((NL, len(GM if k == "m" else GL), len(Q))) for k in ("m", "l")}
    for (key, *_), (k_, l), res in zip(jobs, [(k, l) for k in ("m", "l") for l in range(NL)], pool.map(replay_grid_fs, jobs, chunksize=1)):
        CAP[key][l] = res
    del jobs, Q
    say(f"M-25b JOINT SIZES: cap45 curves done ({time.time() - t0:.0f} s; counted prompts {H}, kept {Kc[H].sum():.0f})")
    # the capture's curves (M-25's protocol)
    names = sorted(f[:-4] for f in os.listdir(B.CAPDIR) if f.endswith(".npz"))
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    NEED = [SC.needs16(z) for z in Z]
    KW = np.array([sum(k for *_, k in nd) for nd in NEED], float)
    CUR = {k: np.zeros((NL, len(GM if k == "m" else GL), len(names))) for k in ("m", "l")}
    for dom in ("code", "ifeval", "multi"):
        ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
        jobs = [(key, GM if key == "m" else GL,
                 [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, nd[0 if key == "m" else 1][l]) for t, nd in enumerate(NEED[i])] for i in ix],
                 [POP[key][i][l] for i in ix]) for key in ("m", "l") for l in range(NL)]
        for (key, *_), (k_, l), res in zip(jobs, [(k, l) for k in ("m", "l") for l in range(NL)], pool.map(SC.replay_grid, jobs, chunksize=1)):
            CUR[key][l][:, ix] = res
    pool.close(); pool.join()
    np.savez_compressed(os.path.join(LZ.OUT, "joint45_curves.npz"), cap_m=CAP["m"], cap_l=CAP["l"], cap_kept=Kc,
                        win_m=CUR["m"], win_l=CUR["l"], win_kept=KW, GM=np.array(GM), GL=np.array(GL), names=np.array(names))
    say(f"  capture curves done ({time.time() - t0:.0f} s); curves saved to data/joint45_curves.npz")
    gi, gj = {x: k for k, x in enumerate(GM)}, {x: k for k, x in enumerate(GL)}
    dom = np.array([w.split("_")[1] for w in names])
    folds = ([], [])
    for d in ("code", "ifeval", "multi"):
        ix = [i for i in range(len(names)) if dom[i] == d]
        folds[0].extend(ix[0::2]); folds[1].extend(ix[1::2])

    def fit(win, budget):
        k = Kc[H].sum() + KW[win].sum()
        cm = (CAP["m"][:, :, H].sum(2) + CUR["m"][:, :, win].sum(2)) * MR / k
        cl = (CAP["l"][:, :, H].sum(2) + CUR["l"][:, :, win].sum(2)) * LR / k
        return SC.alloc([cm, cl], [GM, GL], [MR, LR], budget, [160, 32])

    def score(win, im, jl):
        return sum(CUR["m"][l, im[l], win].sum() * MR + CUR["l"][l, jl[l], win].sum() * LR for l in range(NL)) / 1e6 / KW[win].sum()

    for cs in CS:
        um, ul = UNI[cs]
        base = NL * (um * MR + ul * LR)
        arms = {"uniform": [([gi[um]] * NL, [gj[ul]] * NL)] * 2,
                "M-17 sizes": [([gi[x] for x in M17[cs]["C_m"]], [gj[x] for x in M17[cs]["C_l"]])] * 2}
        for ex in EXTRA:
            arms[f"fit (i) cap45 +{ex / 1e9:.2f} GB"] = [tuple(fit([], base + ex))] * 2
            arms[f"fit (ii) +capture +{ex / 1e9:.2f} GB"] = [tuple(fit(folds[f], base + ex)) for f in (0, 1)]
        i1 = arms["fit (i) cap45 +0.00 GB"][0]
        same = sum(int(GM[a] == m and GL[b] == c) for a, b, m, c in zip(i1[0], i1[1], M17[cs]["C_m"], M17[cs]["C_l"]))
        say(f"  C {cs} (uniform pools {base / 1e9:.2f} GB): fit (i) at equal RAM matches M-17's sizes on {same} / {NL} layers; "
            f"held-out SSD MB per kept token code / IFEval / multi, vs M-17 sizes on code")
        ref, RES = None, {}
        for name, fs in arms.items():
            v = {}
            for d in ("code", "ifeval", "multi"):
                tot = kp = 0.0
                for f in (0, 1):
                    te = [i for i in folds[1 - f] if dom[i] == d]
                    im, jl = fs[f]
                    tot += score(te, im, jl) * KW[te].sum(); kp += KW[te].sum()
                v[d] = tot / kp
            ref = v["code"] if name == "M-17 sizes" else ref
            RES[name] = v
            gb = np.mean([sum(GM[a] * MR + GL[b] * LR for a, b in zip(*fs[f])) for f in (0, 1)]) / 1e9
            rel = f"{100 * (v['code'] / ref - 1):+6.1f} %" if ref else "      "
            say(f"    {name:30s} {v['code']:6.2f} / {v['ifeval']:6.2f} / {v['multi']:6.2f} | {gb:5.2f} GB | {rel}")
        gain = 1 - RES["fit (ii) +capture +0.00 GB"]["code"] / ref
        say(f"    -> fit (ii) vs M-17 at equal RAM, code: {100 * gain:+.1f} % -> " + ("PASS" if gain >= 0.02 else "KILL" if gain < 0.01 else "BETWEEN"))
        for ex in EXTRA:                                        # the files: cap45 + all 36 windows
            im, jl = fit(list(range(len(names))), base + ex)
            cm_, cl_ = [GM[a] for a in im], [GL[b] for b in jl]
            used = sum(a * MR + b * LR for a, b in zip(cm_, cl_))
            fn = os.path.join(LZ.OUT, f"joint45_{cs}_{int(ex / 1e6)}.json")
            json.dump({"C_m": cm_, "C_l": cl_, "budget_bytes": int(base + ex), "used_bytes": int(used),
                       "fit": "M-25b joint (cap45 + capture, P30, FLOOR 160 / 32)"}, open(fn, "w"))
            say(f"    file {os.path.basename(fn)}: C_m {min(cm_)}-{max(cm_)} (sum {sum(cm_)}), C_l {min(cl_)}-{max(cl_)} (sum {sum(cl_)}), "
                f"{used / 1e9:.2f} GB")
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "joint45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
