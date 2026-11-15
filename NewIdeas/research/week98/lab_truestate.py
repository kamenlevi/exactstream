> (a quote from the project owner, kept private)
CPU only, read-only inputs (cells, MoonShots' 2B / 4B rows), the same layout and walk as tree_audit.py.
1. LAB (true state): the head with the 125B's true state at every depth. At depth d from anchor a it reads the trunk at
   row a + d, so its guess IS the depth-0 guess of anchor a + d. Hence truth_rt[a, d] = rt[a + d, 0] (-1 where the
   cell does not exist). CHECK: its kept/32 must reproduce BigMoE's measured REPLACE oracle (HE+ 20.90, MBPP 12.76),
   else the shortcut is wrong and the LAB line needs the capture's REPLACE run.
2. Head-OR-auditor upper bounds (any chooser's ceiling): right if the head OR the auditor is right, per cell (marginal)
   and under the walk.
Reported per band, marginal (all cells, information only in CERTIFY98) and walk-strict (CERTIFY98's gate, amendment 4).
usage: ~/BigMoE/.venv/bin/python -I -B lab_truestate.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/lab_truestate")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
REPLACE = {"he": 20.90, "mbpp": 12.76}
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def marginal(mats):
    num = np.zeros(31); den = np.zeros(31)
    for m in mats:
        v = m[:-1] >= 0
        num += ((m[:-1] == 1) & v).sum(0); den += v.sum(0)
    return [round(100 * num[a:b + 1].sum() / den[a:b + 1].sum(), 2) for a, b in BANDS]


def walked(mats, cap=31):
    ks, num, den = [], np.zeros(31), np.zeros(31)
    for m in mats:
        p = 1
        while p < m.shape[0]:
            k = 0
            while k < cap and m[p, k] == 1:
                k += 1
            for d in range(k):
                num[d] += 1; den[d] += 1
            if k < cap and m[p, k] >= 0:
                den[k] += 1
            ks.append(k + 1); p += k + 1
    ws = [round(100 * num[a:b + 1].sum() / den[a:b + 1].sum(), 2) if den[a:b + 1].sum() else None for a, b in BANDS]
    return round(float(np.mean(ks)), 3), ws, [int(den[a:b + 1].sum()) for a, b in BANDS]


def main():
    TA.say = lambda s="": None
    out = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        for aud in ("2b", "4b"):
            TA.attach_aud(W, f"{TA.MS}/rows{aud}_{rtag}.npz", aud)
        for X in W:
            assert (np.diff(X["an"]) == 1).all(), "anchors are not consecutive rows"
        arms = {}
        arms["t30b (head)"] = [X["rt"] for X in W]
        tr = []
        for X in W:
            rt = X["rt"]; n = rt.shape[0]; t = np.full_like(rt, -1)
            for d in range(31):
                src = np.arange(n - 1) + d
                ok = (src < n - 1) & (rt[:-1, d] >= 0)
                t[:-1, d][ok] = rt[src[ok], 0]
            tr.append(t)
        arms["LAB true state"] = tr
        for aud in ("2b", "4b"):
            arms[f"head OR {aud} (any chooser's ceiling)"] = [np.where(X["rt"] < 0, -1, np.maximum(X["rt"], X[f"A2_{aud}"])) for X in W]
        arms["head OR 2b OR 4b"] = [np.where(X["rt"] < 0, -1, np.maximum(X["rt"], np.maximum(X["A2_2b"], X["A2_4b"]))) for X in W]
        # the frozen audit, per cell (= MoonShots' bands; unlimited repairs, no cascade)
        for aud in ("2b", "4b"):
            mats = []
            for X in W:
                conf = (~np.isnan(X[f"G2_{aud}"])) & (X[f"G2_{aud}"] >= TA.TAU[None, :])
                mats.append(np.where(X["rt"] < 0, -1, np.where(conf, X[f"A2_{aud}"], X["rt"])))
            arms[f"audit {aud} per cell"] = mats
        say(f"\n=== {s} ({tag}) ===  marginal = all cells (information); walk-strict = CERTIFY98's gate")
        res = {}
        for name, mats in arms.items():
            mg = marginal(mats); k, ws, nb = walked(mats)
            res[name] = {"marginal": mg, "walk_strict": ws, "kept32": k, "band_n": nb}
            f = lambda v: " ".join(f"{x:5.1f}" if x is not None else "    -" for x in v)
            say(f"  {name:36s} kept/32 {k:6.2f} | marginal {f(mg)} | walk-strict {f(ws)}")
        k = res["LAB true state"]["kept32"]
        say(f"  CHECK LAB shortcut: kept/32 {k:.2f} vs BigMoE REPLACE {REPLACE[s]} -> {'PASS' if abs(k - REPLACE[s]) <= 0.05 else 'DIFFERS'}")
        say(f"  walk-strict band n (t30b): {res['t30b (head)']['band_n']}")
        out[s] = res
    json.dump(out, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("bands 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
