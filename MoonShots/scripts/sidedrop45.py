"""M-26 SIDECARDROP (MoonShots, Mon 2026-11-02 23:4x; the orchestrator 23:4x; pre-registered in ~/NewIdeas/QUEUE.md M-26 before
any run): the bytes side of plan 3333333333 (every rank at its 3-bit MSB part, no LSB sidecar pool, no LSB reads) on the
45 GB line. Quality is Breakthrough's t3_all (Tue); this prices the bytes only.
Arms at C 256 and C 320, the same pool RAM (uniform split bytes: 33.93 / 42.45 GB): M-17's sizes and the M-25b joint fit
(plan 4433333333 with the sidecar), and SIDECARDROP (C_l = 0, a base-only per-layer fit at the full RAM: M-17's
Lagrangian, FLOOR C_m >= 160, P30). Fits on cap45 (counted prompts) + the capture's training windows, no-Q1 needs, 16-row
blocks (M-25b's curves, data/joint45_curves.npz); scored on the held-out windows (2 folds by window).
Rules: no Q1, and Q1 (top-2 on rows whose DFlash2 cf >= 0.99; the capture's conf, NaN = exact). Harnesses: 16-row blocks of
the uncut walk (M-25 / M-25b, the verdict) and the capture's cut rows (the engine's t 0.3 checks), each scored by SSD MB per
kept token (MSB misses x 2,162,688 B + LSB misses x 770,048 B; no LSB under 3333333333).
Line (the orchestrator's): PASS = SIDECARDROP <= 33 MB per kept token on code at C 256 with Q1 (16-row harness); KILL > 40.
usage: [BigMoE venv] taskpolicy -b python sidedrop45.py -> data/sidedrop45.txt + data/sidedrop45_<C>.json (run_guarded.sh)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import json, multiprocessing as mp, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45 as B
import laysz45 as LZ
import wide45 as W4
import sidecars45 as SC

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def needs_of(z, which, q1):
    """per check: (base needs, sidecar needs) per layer as int16 arrays, and kept; which = 'blk' (checks_walk) or 'cut'"""
    ids, conf = z["ids"], z["conf"].astype(float)
    out = []
    for a, n, k in z["checks_walk" if which == "blk" else "checks"].tolist():
        rid = ids[a:a + n]
        kk = np.array([2 if (q1 and c >= 0.99) else 10 for c in conf[a:a + len(rid)]])
        use = np.arange(10)[None, :] < kk[:, None]
        out.append(([np.unique(rid[:, l][use]).astype(np.int16) for l in range(NL)],
                    [np.unique(rid[:, l, :2][use[:, :2]]).astype(np.int16) for l in range(NL)], k))
    return out


def main():
    t0 = time.time()
    pool = mp.get_context("fork").Pool(4)
    J = np.load(os.path.join(LZ.OUT, "joint45_curves.npz"))
    assert list(J["GM"]) == SC.GM and list(J["GL"]) == SC.GL
    names = [str(x) for x in J["names"]]
    CAPm, Kc, H = J["cap_m"], J["cap_kept"], list(LZ.H)
    C = {w: W4.counts_of(np.load(os.path.join(B.CAPDIR, w + ".npz"))) for w in names}
    tm, tl, tn = (sum(C[w][j] for w in names) for j in range(3))
    POP = {"m": [[(tm - C[w][0])[l] / (tn - C[w][2]) for l in range(NL)] for w in names],
           "l": [[(tl - C[w][1])[l] / (tn - C[w][2]) for l in range(NL)] for w in names]}
    Z = [{k: v for k, v in np.load(os.path.join(B.CAPDIR, w + ".npz")).items()} for w in names]
    PREW = [W4.prefill_of(z) for z in Z]
    CUR = {("blk", False): (J["win_m"], J["win_l"], J["win_kept"])}
    for which, q1 in (("blk", True), ("cut", False), ("cut", True)):
        NEED = [needs_of(z, which, q1) for z in Z]
        KW = np.array([sum(k for *_, k in nd) for nd in NEED], float)
        M = {k: np.zeros((NL, len(SC.GM if k == "m" else SC.GL), len(names))) for k in ("m", "l")}
        for dom in ("code", "ifeval", "multi"):
            ix = [i for i, w in enumerate(names) if w.split("_")[1] == dom]
            jobs = [(key, SC.GM if key == "m" else SC.GL,
                     [[(-1, pc[0 if key == "m" else 1][l]) for pc in PREW[i]] + [(t, nd[0 if key == "m" else 1][l]) for t, nd in enumerate(NEED[i])]
                      for i in ix], [POP[key][i][l] for i in ix]) for key in ("m", "l") for l in range(NL)]
            for (key, *_), (k_, l), res in zip(jobs, [(k, l) for k in ("m", "l") for l in range(NL)], pool.map(SC.replay_grid, jobs, chunksize=1)):
                M[key][l][:, ix] = res
        CUR[(which, q1)] = (M["m"], M["l"], KW)
        say(f"  curves {which} {'Q1' if q1 else 'no Q1'} done ({time.time() - t0:.0f} s)")
    pool.close(); pool.join()
    gi, gj = {x: k for k, x in enumerate(SC.GM)}, {x: k for k, x in enumerate(SC.GL)}
    dom = np.array([w.split("_")[1] for w in names])
    folds = ([], [])
    for d in ("code", "ifeval", "multi"):
        ix = [i for i in range(len(names)) if dom[i] == d]
        folds[0].extend(ix[0::2]); folds[1].extend(ix[1::2])
    Wm, Wl, Wk = CUR[("blk", False)]
    CAPl = J["cap_l"]

    def fit(win, budget, side):
        k = Kc[H].sum() + Wk[win].sum()
        cm = (CAPm[:, :, H].sum(2) + Wm[:, :, win].sum(2)) * MR / k
        if not side:
            return SC.alloc([cm], [SC.GM], [MR], budget, [160])[0], np.full(NL, gj[0])
        cl = (CAPl[:, :, H].sum(2) + Wl[:, :, win].sum(2)) * LR / k
        return tuple(SC.alloc([cm, cl], [SC.GM, SC.GL], [MR, LR], budget, [160, 32]))

    say(f"M-26 SIDECARDROP: SSD MB per kept token on the capture's held-out windows (code / IFEval / multi)")
    verdict = None
    for cs in SC.CS:
        um, ul = SC.UNI[cs]
        budget = NL * (um * MR + ul * LR)
        arms = {"M-17 sizes (4433333333)": [([gi[x] for x in SC.M17[cs]["C_m"]], [gj[x] for x in SC.M17[cs]["C_l"]])] * 2,
                "joint fit (4433333333)": [fit(folds[f], budget, True) for f in (0, 1)],
                "SIDECARDROP (3333333333)": [fit(folds[f], budget, False) for f in (0, 1)]}
        for which in ("blk", "cut"):
            for q1 in (False, True):
                cm_, cl_, kw = CUR[(which, q1)]
                say(f"  C {cs} ({budget / 1e9:.2f} GB), {'16-row blocks' if which == 'blk' else 'cut rows'}, {'Q1 0.99' if q1 else 'no Q1'}:")
                for name, fs in arms.items():
                    v = {}
                    for d in ("code", "ifeval", "multi"):
                        tot = kp = 0.0
                        for f in (0, 1):
                            te = [i for i in folds[1 - f] if dom[i] == d]
                            im, jl = fs[f]
                            side = name.startswith("SIDECARDROP")
                            tot += sum(cm_[l, im[l], te].sum() * MR + (0 if side else cl_[l, jl[l], te].sum() * LR) for l in range(NL)) / 1e6
                            kp += kw[te].sum()
                        v[d] = tot / kp
                    say(f"    {name:26s} {v['code']:6.2f} / {v['ifeval']:6.2f} / {v['multi']:6.2f}")
                    if cs == 256 and which == "blk" and q1 and name.startswith("SIDECARDROP"):
                        verdict = v["code"]
        im, _ = fit(list(range(len(names))), budget, False)
        cmf = [SC.GM[a] for a in im]
        fn = os.path.join(LZ.OUT, f"sidedrop45_{cs}.json")
        json.dump({"C_m": cmf, "C_l": [0] * NL, "budget_bytes": int(budget), "used_bytes": int(sum(cmf) * MR),
                   "fit": "M-26 SIDECARDROP: plan 3333333333, base-only (cap45 + capture, P30, FLOOR 160)"}, open(fn, "w"))
        say(f"    file {os.path.basename(fn)}: C_m {min(cmf)}-{max(cmf)} (sum {sum(cmf)}), {sum(cmf) * MR / 1e9:.2f} GB")
    say(f"  -> SIDECARDROP at C 256 with Q1, code (16-row blocks): {verdict:.2f} MB per kept token -> "
        + ("PASS" if verdict <= 33 else "KILL" if verdict > 40 else "BETWEEN"))
    say(f"  total {time.time() - t0:.0f} s")
    open(os.path.join(LZ.OUT, "sidedrop45.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
