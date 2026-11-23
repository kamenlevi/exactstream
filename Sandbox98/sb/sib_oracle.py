"""SIB mix oracle (NewIdeas 15:0x; results/LEVERS_A.md, SIB main result): can a linear mix of the head's own carried
stand-in and the 2B's sibling state beat the carried state alone, judged by the direction error to the true state?
One own-chain pass per window records the carried rows at the sibling slots (sibstate.make_fn's rec mode); the
sibling rows (post-norm) and the true W come from disk. For a grid w in [0, 1]: m = unit(w * unit(sib) + (1 - w) *
unit(own)), err = RMS(unit(m) - unit(true)) (sibstate.dir_err's scale). w* is fitted per band on one half of the windows
(by parity) and scored on the other half, both ways. Also: the cosine between the two error vectors (own - true,
sib - true). Decision (NewIdeas): an oracle mix within ~5 % of own's error at bands 1-2 and 3-6 -> a linear sibmix
can't reach G >= 0.25 (phase-2 KILL near-certain); >= 15 % better -> phase 2 is worth its run.
usage: sb/sib_oracle.py --out results/sib_oracle.json"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BANDS = {"1-2": (1, 2), "3-6": (3, 6), "7-8": (7, 8), "1-8": (1, 8), "3-8": (3, 8)}
WG = np.round(np.linspace(0, 1, 21), 3)


def unit(x):
    return x / np.maximum(np.sqrt((x ** 2).mean(-1, keepdims=True)), 1e-6)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="he,mbpp"); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    import mlx.core as mx
    from sb import core as C, sibstate as SB
    C.limit_memory(float(os.environ.get("SB_MEM_GB", "3")))
    C.ARGMAX_ROWS = 256
    C.load_target(); C.drop_trunk()
    D = C.load_drafter("s30")
    t0 = time.time(); res = {}
    for st in a.sets.split(","):
        fs = SB.set_files(st, a.n)
        E = np.zeros((2, SB.NSIB, len(WG))); N = np.zeros((2, SB.NSIB)); CS = np.zeros((2, SB.NSIB))
        for wi, f in enumerate(fs):
            key = os.path.basename(f)[:-4]
            o = np.load(SB.H("data", "sib", st, key + ".npz"))
            if len(o["starts"]) == 0:
                continue
            z = np.load(f); S = len(z["ids"])
            Wn = np.array(C.load_W(z).astype(mx.float32)[0])
            rep_s, msk, _ = SB.rep_rows(C, mx, o, S, "sib")
            rec = []
            C.chain_table(D, f, Dp=SB.NSIB + 1, fn=SB.make_fn(mx, None, msk, rec))
            rs = np.array(rep_s); hf = wi % 2
            for j, anc, m, h in rec:
                sel = np.flatnonzero(m)
                if len(sel) == 0:
                    continue
                slot = j - 2; ai = anc[sel]
                ou = unit(np.array(h[mx.array(sel)]).astype(np.float32))
                su = unit(rs[ai, slot]); tu = unit(Wn[ai + j - 1])
                for gi, w in enumerate(WG):
                    E[hf, slot, gi] += np.sqrt(((unit(w * su + (1 - w) * ou) - tu) ** 2).mean(-1)).sum()
                eo, es = ou - tu, su - tu
                CS[hf, slot] += ((eo * es).sum(-1) / np.maximum(np.linalg.norm(eo, axis=-1) * np.linalg.norm(es, axis=-1),
                                                                1e-9)).sum()
                N[hf, slot] += len(sel)
            mx.clear_cache()
            if (wi + 1) % 20 == 0 or wi == len(fs) - 1:
                print(f"[oracle] {st} {wi + 1}/{len(fs)}, {time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 1e9:.2f} GB",
                      flush=True)
        r = {}
        for b, (lo, hi) in BANDS.items():
            sl = slice(lo - 1, hi)
            err = [E[h, sl].sum(0) / max(N[h, sl].sum(), 1) for h in (0, 1)]     # [half][w]
            full = E[:, sl].sum((0, 1)) / max(N[:, sl].sum(), 1)
            ways = []
            for fit, ev in ((0, 1), (1, 0)):
                wi_ = int(np.argmin(err[fit]))
                ways.append({"w_star": float(WG[wi_]), "own": float(err[ev][0]), "sib": float(err[ev][-1]),
                             "mix": float(err[ev][wi_]), "gain": float(1 - err[ev][wi_] / err[ev][0])})
            r[b] = {"held_out": ways, "gain_mean": float(np.mean([w["gain"] for w in ways])),
                    "in_sample": {"w_star": float(WG[int(np.argmin(full))]), "own": float(full[0]), "sib": float(full[-1]),
                                  "mix": float(full.min()), "gain": float(1 - full.min() / full[0])},
                    "cos_err": float(CS[:, sl].sum() / max(N[:, sl].sum(), 1)), "cells": int(N[:, sl].sum()),
                    "curve": [float(x) for x in full]}
            print(f"[oracle] {st} {b}: own {full[0]:.4f}, sib {full[-1]:.4f}, best mix {full.min():.4f} at w "
                  f"{WG[int(np.argmin(full))]:.2f} (in-sample gain {100 * (1 - full.min() / full[0]):.1f} %; held-out "
                  f"{100 * ways[0]['gain']:.1f} / {100 * ways[1]['gain']:.1f} %), cos(own err, sib err) "
                  f"{r[b]['cos_err']:.3f}, {r[b]['cells']} cells", flush=True)
        res[st] = r
    json.dump({"grid": WG.tolist(), "res": res, "wall_s": time.time() - t0}, open(a.out, "w"), indent=1)
    print(f"[oracle] DONE -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
