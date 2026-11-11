"""M-32 FCASTSIZE (MoonShots, Wed 2026-11-04 09:0x; pre-registered in ~/NewIdeas/QUEUE.md M-32 before any run; the
orchestrator pulled it forward 09:0x): with a layer-ahead forecast, a miss the forecast catches is read early and hidden,
so pool slots are worth more in the layers the forecast misses. Re-fit M-25b's joint per-layer (C_m, C_l) sizes with each
layer's misses weighted by w_l = 1 - r_l, the forecast's per-row recall for layer l, at equal RAM (C 224 = laysz45_224's
budget; C 256 = uniform split bytes), against M-25b's unweighted joint fit (the current joint sizes).
Recall r_l: BigMoE's corrected pre-route (~/BigMoE/data/r57/pr2_160_preroute.txt, 'full', per row, by source-layer band
0-7 / 8-39 / 40-46: code 0.676 / 0.727 / 0.687, chat 0.745 / 0.841 / 0.857, all 0.725 / 0.808 / 0.807); layer L + 1
gets source L's band; layer 0 has no forecast (r 0). The fit uses 'all' (the train windows mix domains); each held-out
domain is scored with its own r. The forecast is taken to cover both parts of an expert it names (base and LSB).
Data: M-25b's curves (data/joint45_curves.npz: cap45 + Breakthrough's 36 capture windows, 16-row blocks, no Q1), 2 folds
by window, as M-25b / M-26. Score: forecast-missed demand MB per kept token = sum_l w_l x (base misses x MR + LSB misses
x LR) / kept; exposed ms per check = its change x kept per check (the capture's: code 9.6, chat 2.4) / 12.2 GB/s x 0.56
(SSD_WALL2's exposed share of busy).
Line: >= 1.0 ms less exposed per check at equal RAM vs the current joint sizes: PASS; < 0.3 ms: KILL; else BETWEEN
(read on code at C 256).
usage: [BigMoE venv] python fcastsize.py -> data/fcastsize.txt"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laysz45 as LZ
import sidecars45 as SC

NL, MR, LR = LZ.NL, LZ.MR, LZ.LR
BAND = {"code": (0.676, 0.727, 0.687), "chat": (0.745, 0.841, 0.857), "all": (0.725, 0.808, 0.807)}
KPC = {"code": 9.6, "ifeval": 2.4, "multi": 2.4}
GBS, EXPOSED = 12.2, 0.56
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def recall(kind):
    r = np.zeros(NL)
    for l in range(1, NL):
        src = l - 1
        r[l] = BAND[kind][0 if src <= 7 else 1 if src <= 39 else 2]
    return r


def main():
    t0 = time.time()
    J = np.load(os.path.join(LZ.OUT, "joint45_curves.npz"))
    assert list(J["GM"]) == SC.GM and list(J["GL"]) == SC.GL
    names = [str(x) for x in J["names"]]
    CAPm, CAPl, Kc, H = J["cap_m"], J["cap_l"], J["cap_kept"], list(LZ.H)
    Wm, Wl, Wk = J["win_m"], J["win_l"], J["win_kept"]
    dom = np.array([w.split("_")[1] for w in names])
    folds = ([], [])
    for d in ("code", "ifeval", "multi"):
        ix = [i for i in range(len(names)) if dom[i] == d]
        folds[0].extend(ix[0::2]); folds[1].extend(ix[1::2])
    wall = 1 - recall("all")

    def fit(win, budget, w):
        k = Kc[H].sum() + Wk[win].sum()
        cm = (CAPm[:, :, H].sum(2) + Wm[:, :, win].sum(2)) * MR / k * w[:, None]
        cl = (CAPl[:, :, H].sum(2) + Wl[:, :, win].sum(2)) * LR / k * w[:, None]
        return tuple(SC.alloc([cm, cl], [SC.GM, SC.GL], [MR, LR], budget, [160, 32]))

    budgets = {224: json.load(open(os.path.join(LZ.OUT, "laysz45_224.json")))["budget_bytes"]}
    um, ul = SC.UNI[256]
    budgets[256] = NL * (um * MR + ul * LR)
    say(f"== M-32 FCASTSIZE ({time.strftime('%a %H:%M')}): per-layer sizes re-fit to the corrected pre-route's recall (w = 1 - r); "
        f"held-out forecast-missed MB per kept token (code / IFEval / multi) and exposed ms per check")
    verdict = None
    for C, budget in budgets.items():
        arms = {"joint (M-25b)": [fit(folds[f], budget, np.ones(NL)) for f in (0, 1)],
                "forecast-aware": [fit(folds[f], budget, wall) for f in (0, 1)]}
        V, raw = {}, {}
        for name, fs in arms.items():
            v, u = {}, {}
            for d in ("code", "ifeval", "multi"):
                wd = 1 - recall("code" if d == "code" else "chat")
                tot = tr = kp = 0.0
                for f in (0, 1):
                    te = [i for i in folds[1 - f] if dom[i] == d]
                    im, jl = fs[f]
                    for l in range(NL):
                        b = Wm[l, im[l], te].sum() * MR + Wl[l, jl[l], te].sum() * LR
                        tot += wd[l] * b; tr += b
                    kp += Wk[te].sum()
                v[d], u[d] = tot / kp / 1e6, tr / kp / 1e6
            V[name], raw[name] = v, u
            say(f"  C {C} ({budget / 1e9:.2f} GB) {name:15s}: forecast-missed {v['code']:6.2f} / {v['ifeval']:6.2f} / {v['multi']:6.2f}"
                f" MB per kept token (all demand {u['code']:6.2f} / {u['ifeval']:6.2f} / {u['multi']:6.2f})")
        ms = {d: (V["joint (M-25b)"][d] - V["forecast-aware"][d]) * KPC[d] / GBS * EXPOSED for d in KPC}
        say(f"    forecast-aware vs joint: exposed -{ms['code']:.2f} / -{ms['ifeval']:.2f} / -{ms['multi']:.2f} ms per check "
            f"(code / IFEval / multi); all-demand MB {100 * (raw['forecast-aware']['code'] / raw['joint (M-25b)']['code'] - 1):+.1f} % on code")
        if C == 256:
            verdict = ms["code"]
    say(f"  -> code at C 256: {verdict:.2f} ms less exposed per check -> "
        + ("PASS" if verdict >= 1.0 else "KILL" if verdict < 0.3 else "BETWEEN") + f" (line 1.0 / 0.3 ms)  ({time.time() - t0:.0f} s)")
    open(os.path.join(LZ.OUT, "fcastsize.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
