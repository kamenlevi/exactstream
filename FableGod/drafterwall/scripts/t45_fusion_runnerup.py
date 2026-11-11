#!/usr/bin/env python3
"""t45_fusion_runnerup.py (drafterwall T4 + T5, pre-registered in PREREG.md, commit 602d66d). CPU.
Official loop at 16 rows on the tree3_map tables (teacher-forced top-3 of DFlash2 L1-20 and of the MTP head b5: equal to
the live chains up to the first wrong pick, which is all that kept depends on). Windows sorted by name: even = tune,
odd = held-out.
T4: pick = argmax over the union of both top-3 of w(d) log p_dfl + (1 - w(d)) log p_mtp, floor eps; w for depth 0, 1-2, 3+.
T5: the MTP chain; where its top-1 p < theta, switch to its top-2 / top-3 candidate if exactly one of them was a rank 2-3
    token of the 125B at one of the last W verified rows and its top-1 was not."""
import numpy as np, os, glob, itertools
TD = os.path.expanduser("~/TrainLab/data/tree3_map"); out = []
def say(s=""):
    print(s); out.append(s)
def load(dom):
    W = []
    for f in sorted(glob.glob(f"{TD}/{dom}/*.npz")):
        z = np.load(f); W.append({k: z[k] for k in ("true", "m", "dfl_t3", "dfl_p3", "mtp_t3", "mtp_p3", "t125_t3")})
    return W
def run(W, picker):
    kept = []
    for w in W:
        true, mm = w["true"], w["m"]; n = true.shape[0]; p = 0
        while p < n:
            m = min(15, int(mm[p])); k = 0
            while k < m and picker(w, p, k) == true[p, k]: k += 1
            kept.append(k + 1); p += k + 1
    return float(np.mean(kept))
def pk_dfl(w, p, k): return w["dfl_t3"][p, k, 0]
def pk_mtp(w, p, k): return w["mtp_t3"][p, k, 0]
def pk_max(w, p, k): return w["dfl_t3"][p, k, 0] if w["dfl_p3"][p, k, 0] >= w["mtp_p3"][p, k, 0] else w["mtp_t3"][p, k, 0]
def pk_orc(w, p, k):
    t = w["true"][p, k]
    return t if (w["dfl_t3"][p, k, 0] == t or w["mtp_t3"][p, k, 0] == t) else -7
def mk_fuse(w0, w12, w3, eps):
    le = np.log(eps)
    def f(w, p, k):
        wd = w0 if k == 0 else (w12 if k <= 2 else w3)
        dt, dp, mt, mp = w["dfl_t3"][p, k], w["dfl_p3"][p, k], w["mtp_t3"][p, k], w["mtp_p3"][p, k]
        best, bs = -1, -1e30
        for t in list(dt) + list(mt):
            ld = le; lm = le
            for i in range(3):
                if dt[i] == t: ld = np.log(max(float(dp[i]), eps))
                if mt[i] == t: lm = np.log(max(float(mp[i]), eps))
            s = wd * ld + (1 - wd) * lm
            if s > bs: bs, best = s, t
        return best
    return f
def mk_runner(theta, Wn):
    def f(w, p, k):
        mt, mp = w["mtp_t3"][p, k], w["mtp_p3"][p, k]
        if mp[0] >= theta or p == 0: return mt[0]
        ru = set(int(x) for x in w["t125_t3"][max(0, p - Wn):p, 0, 1:3].ravel())   # ranks 2-3 at the last Wn verified rows
        if int(mt[0]) in ru: return mt[0]
        c = [int(t) for t in mt[1:3] if int(t) in ru]
        return c[0] if len(c) == 1 else mt[0]
    return f
GRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
v4 = []; v5 = []
for dom in ("code", "ifeval", "multi"):
    W = load(dom); tune, held = W[0::2], W[1::2]
    base = {nm: (run(tune, f), run(held, f)) for nm, f in (("DFlash2", pk_dfl), ("MTP", pk_mtp), ("max-confidence pick", pk_max), ("either-right oracle", pk_orc))}
    say(f"\n== {dom}: tune {len(tune)} windows / held-out {len(held)}; kept per 16-row check (tune / held-out): " + ", ".join(f"{k} {a:.2f} / {b:.2f}" for k, (a, b) in base.items()))
    best = None
    for eps in (1e-2, 1e-3, 1e-4):
        for w0, w12, w3 in itertools.product(GRID, GRID, GRID):
            s = run(tune, mk_fuse(w0, w12, w3, eps))
            if best is None or s > best[0]: best = (s, w0, w12, w3, eps)
    s, w0, w12, w3, eps = best; h = run(held, mk_fuse(w0, w12, w3, eps))
    better = max(base["DFlash2"][1], base["MTP"][1])
    say(f"   T4 fusion: tuned w (depth 0 / 1-2 / 3+) = {w0} / {w12} / {w3}, eps {eps}: tune {s:.2f}, HELD-OUT {h:.2f} = x{h / better:.3f} the better single reader ({better:.2f}); "
        f"the max-confidence pick x{base['max-confidence pick'][1] / better:.3f}, the oracle x{base['either-right oracle'][1] / better:.3f}")
    v4.append(h / better)
    bestr = None
    for theta in (0.5, 0.7, 0.9):
        for Wn in (8, 32, 128):
            s = run(tune, mk_runner(theta, Wn))
            if bestr is None or s > bestr[0]: bestr = (s, theta, Wn)
    s, theta, Wn = bestr; h5 = run(held, mk_runner(theta, Wn))
    say(f"   T5 runner-up: tuned theta {theta}, W {Wn}: tune {s:.2f} (MTP {base['MTP'][0]:.2f}), HELD-OUT {h5:.2f} = x{h5 / base['MTP'][1]:.3f} the MTP chain")
    v5.append(h5 / base["MTP"][1])
say(f"\nT4 LINE (PASS: held-out >= x1.20 on any domain; KILL: < x1.08 on all three): x{v4[0]:.3f} / x{v4[1]:.3f} / x{v4[2]:.3f} -> "
    + ("PASS" if max(v4) >= 1.20 else ("KILL" if max(v4) < 1.08 else "BETWEEN")))
say(f"T5 LINE (PASS: held-out >= x1.05 on any domain; KILL: < x1.02 on all three): x{v5[0]:.3f} / x{v5[1]:.3f} / x{v5[2]:.3f} -> "
    + ("PASS" if max(v5) >= 1.05 else ("KILL" if max(v5) < 1.02 else "BETWEEN")))
open(os.path.join(os.path.dirname(__file__), "..", "out", "t45_fusion_runnerup.txt"), "w").write("\n".join(out) + "\n")
