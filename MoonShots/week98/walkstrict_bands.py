"""WEEK98 PLAN4D item 1, re-derived (MoonShots, Sun 15 Nov 2026 22:0x): walk-strict accuracy per band under the decode
walk (certify98's kept_walk logic from row 1, 32-row checks) for the head and the frozen audit (2B P = 1 / 2 / inf, 4B
P = 1; tau 4 / 2 / 1 / 0.5) on HE+ 24 and MBPP 100. An independent check of NewIdeas' research/week98/tree_audit.py
numbers (PLAN4D_REAIM978.md, 8df555d), built from checkstart.py's own layout and walk (imported read-only).
Walk-strict at depth j: among the walk's checks that reach depth j (every earlier depth right) AND have a target at j,
the share right at j (MTP Head's K98 definition). checkstart.py's d0-d2 also counted a no-target cell as reached, a
small low bias that is fixed here (its kept/32 is unaffected); both conventions are printed for depth 0.
Window-clustered 95 % lower bounds: 2,000 window resamples, seed 1115 (as tree_k98.py).
TRUE-STATE (lab) numbers: teacher-forced cells with the 125B's true state, not real decoding.
usage: ~/BigMoE/.venv/bin/python -I -B walkstrict_bands.py"""
import json, os, sys
import numpy as np

H = os.path.expanduser
sys.path.insert(0, H("~/MoonShots/week98"))
import checkstart as K                                    # noqa: E402  (per_cell, mats, certify98 / chooser read-only)

BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
NAME = {"heplus": "HE+ 24", "mbpp": "MBPP 100"}


def walk(Hw, Cw, P):
    """per window: kept list, num / den [31] (target-bearing reached cells), nt [31] (no-target cells hit by the walk)"""
    out = []
    for h, c in zip(Hw, Cw):
        kept, num, den, nt = [], np.zeros(31), np.zeros(31), np.zeros(31)
        p = K.CF.R0
        while p < h.shape[0]:
            k = used = 0
            while k < 31:
                hv, cv = h[p, k], c[p, k]
                if hv == -1:
                    nt[k] += 1
                    break
                if P is not None and used >= P:
                    ok = hv == 1
                elif hv == 1 and cv == 1:
                    ok = True
                elif hv == 0 and cv == 1:
                    ok = True; used += 1
                else:
                    ok = False
                den[k] += 1; num[k] += ok
                if not ok:
                    break
                k += 1
            kept.append(k + 1); p += k + 1
        out.append((kept, num, den, nt))
    return out


def summarize(W, rng):
    kept = np.array([sum(w[0]) for w in W], float); nck = np.array([len(w[0]) for w in W], float)
    num = np.stack([w[1] for w in W]); den = np.stack([w[2] for w in W]); nt = np.stack([w[3] for w in W])
    idx = rng.integers(0, len(W), (2000, len(W)))
    bands = {}
    for lo, hi in BANDS:
        c = num[:, lo:hi + 1].sum(1); r = den[:, lo:hi + 1].sum(1)
        bs = c[idx].sum(1) / np.maximum(r[idx].sum(1), 1)
        bands[f"{lo}-{hi}" if hi > lo else f"{lo}"] = dict(pt=round(100 * c.sum() / max(r.sum(), 1), 2),
                                                           lb=round(100 * float(np.percentile(bs, 2.5)), 2),
                                                           n=int(r.sum()))
    d0_old = 100 * num[:, 0].sum() / max(den[:, 0].sum() + nt[:, 0].sum(), 1)
    return dict(kept32=round(float(kept.sum() / nck.sum()), 3), checks=int(nck.sum()), bands=bands,
                d0_checkstart_convention=round(float(d0_old), 2), no_target_hits_d0=int(nt[:, 0].sum()))


def main():
    rng = np.random.default_rng(1115)
    res = {}
    for name in ("heplus", "mbpp"):
        for aud in ("2b", "4b"):
            c = K.per_cell(name, aud)
            Hm = K.mats(name, c["right"], "head")
            tau = np.array([K.CH.SCHED[int(d)] for d in c["depth"]], np.float32)
            comb = np.where(c["m2b"] >= tau, c["r2b"], c["right"])
            Cm = K.mats(name, comb, f"{aud}_frozen")
            if aud == "2b":
                res[f"{name} head"] = summarize(walk(Hm, Hm, 0), rng)
                for P, lab in ((1, "P=1"), (2, "P=2"), (None, "P=inf")):
                    res[f"{name} 2B {lab}"] = summarize(walk(Hm, Cm, P), rng)
            else:
                res[f"{name} 4B P=1"] = summarize(walk(Hm, Cm, 1), rng)
    for k, v in res.items():
        b = v["bands"]
        print(f"{k:18s} " + " | ".join(f"{x}: {b[x]['pt']:.1f}" for x in b) + f"  (kept/32 {v['kept32']:.3f}, "
              f"{v['checks']} checks; d0 under checkstart's convention {v['d0_checkstart_convention']:.2f}, "
              f"{v['no_target_hits_d0']} no-target d0 hits)")
        print(f"{'':18s} LB95: " + " | ".join(f"{x}: {b[x]['lb']:.1f}" for x in b) + "   n: "
              + " / ".join(str(b[x]["n"]) for x in b))
    json.dump(res, open(H("~/MoonShots/week98/walkstrict_bands.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
