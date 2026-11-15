"""Rough power of the re-aimed 97.8 lines (NewIdeas, Sun 15 Nov 22:0x). CPU only. A window bootstrap from the dev sets
(HE+ 24 windows; MBPP 100), K = 140 fresh windows drawn with replacement, 2,000 draws. Each draw: the pooled ratio per band
and CERTIFY98's lower bound (ratio minus t(K-1, 0.95) x the window-clustered CR1 SE, one-sided 95 %). Rough: it treats
dev's pooled values as the truth and resamples only 24 HE+ windows; EvalSuite's certify98_power_strict.py is the
registered tool.
Lines (PLAN4D_REAIM978.md):
  LAB (marginal, the true-state head): every band point >= 97.5 AND LB >= 97.0 (also: point >= 97.8 per band, info);
  C2 (2B audit, walk-strict): 15-22 and 23-30 point >= 97.8 AND LB >= 97.0, at P = 2 (headline) and P = 1;
  C1 (tree-16, walk-strict coverage): band 0 point >= 97.8 AND LB >= 97.0;
  and the full gate for reference (every walk-strict band 0-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 LB >= 97.0).
usage: ~/BigMoE/.venv/bin/python -I -B power_reaim.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/power_reaim")
GB = [(0, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]          # CERTIFY98 amendment 4's gate bands
MB = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]   # marginal bands
K, NDRAW, T95 = 140, 2000, 1.6559
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def walk_counts(W, acc_fn, cap, bands):
    """per window: walk-strict num / den per band"""
    out = []
    for X in W:
        num = np.zeros(31); den = np.zeros(31); p = 1
        while p < X["n"]:
            a = min(int(acc_fn(X, p)), cap)
            num[:a] += 1; den[:a] += 1
            if a < cap and X["rt"][p, a] >= 0:
                den[a] += 1
            p += a + 1
        out.append([(num[a:b + 1].sum(), den[a:b + 1].sum()) for a, b in bands])
    return np.array(out)                                   # [windows, bands, 2]


def marg_counts(mats, bands):
    out = []
    for m in mats:
        v = m[:-1] >= 0; r = (m[:-1] == 1) & v
        out.append([(r[:, a:b + 1].sum(), v[:, a:b + 1].sum()) for a, b in bands])
    return np.array(out, dtype=np.float64)


def boot(C, rng):
    """C [windows, bands, 2] -> point and LB per band for NDRAW draws of K windows"""
    n = C.shape[0]; P = np.zeros((NDRAW, C.shape[1])); L = np.zeros_like(P)
    for i in range(NDRAW):
        s = C[rng.integers(0, n, K)]
        num, den = s[:, :, 0], s[:, :, 1]; R = num.sum(0) / np.maximum(den.sum(0), 1)
        e = num - R[None, :] * den
        se = np.sqrt(K / (K - 1) * (e ** 2).sum(0)) / np.maximum(den.sum(0), 1)
        P[i] = 100 * R; L[i] = 100 * (R - T95 * se)
    return P, L


def main():
    rng = np.random.default_rng(1115)
    TA.say = lambda s="": None
    res = {}
    for s, (tag, rtag, pub32, pub16) in TA.SETS.items():
        W = TA.load_windows(tag, pub32)
        TA.attach_aud(W, f"{TA.MS}/rows2b_{rtag}.npz", "2b")
        # EAGLE-2-16 accepted per anchor (tree_audit's rule)
        tz = np.load(f"{TA.CE}/tree_t30b_{tag}.npz", allow_pickle=True)
        fi, an, nd, dp, ot = (tz[k] for k in ("fi", "an", "node", "depth", "on_true"))
        hit = ot & (nd < TA.NODES)
        acc = {}
        for f, a_, d in zip(fi[hit], an[hit], dp[hit]):
            acc[(int(f), int(a_))] = max(acc.get((int(f), int(a_)), 0), int(d) + 1)
        for X in W:
            X["A_t"] = np.array([acc.get((X["fi"], int(a_)), 0) for a_ in X["an"]] + [0])
        # LAB matrices (lab_truestate.py's rule)
        lab = []
        for X in W:
            rt = X["rt"]; n = rt.shape[0]; t = np.full_like(rt, -1)
            for d in range(31):
                src = np.arange(n - 1) + d; ok = (src < n - 1) & (rt[:-1, d] >= 0)
                t[:-1, d][ok] = rt[src[ok], 0]
            lab.append(t)
        arms = {
            "t30b (walk-strict)": walk_counts(W, lambda X, p: TA.lead_right(X["rt"][p], 0, 31), 31, GB),
            "2B audit P=1 (walk-strict)": walk_counts(W, lambda X, p: TA.audit_chain(X, p, 31, "2b", P=1), 31, GB),
            "2B audit P=2 (walk-strict)": walk_counts(W, lambda X, p: TA.audit_chain(X, p, 31, "2b", P=2), 31, GB),
            "tree-16 (walk-strict, band 0 only)": walk_counts(W, lambda X, p: X["A_t"][p], TA.NODES, [(0, 0)]),
            "LAB true state (walk-strict)": walk_counts(W, lambda X, p, L=None: 0, 31, GB),   # placeholder, replaced below
            "LAB true state (marginal)": marg_counts(lab, MB),
        }
        labw = {id(X): m for X, m in zip(W, lab)}
        arms["LAB true state (walk-strict)"] = walk_counts(W, lambda X, p: TA.lead_right(labw[id(X)][p], 0, 31), 31, GB)
        say(f"\n=== {s}: {len(W)} dev windows -> K {K} fresh (bootstrap, {NDRAW} draws) ===")
        r = {}
        for name, C in arms.items():
            P, L = boot(C, rng)
            bands = MB if "marginal" in name else ([(0, 0)] if "tree" in name else GB)
            pt = 100 * C[:, :, 0].sum(0) / C[:, :, 1].sum(0)
            pass_lb = (L >= 97.0).mean(0); pass_978 = (P >= 97.8).mean(0); both = ((P >= 97.8) & (L >= 97.0)).mean(0)
            say(f"  {name}")
            say("    band        " + " ".join(f"{a}-{b}".rjust(7) for a, b in bands))
            say("    dev point   " + " ".join(f"{x:7.2f}" for x in pt))
            say("    P(LB>=97.0) " + " ".join(f"{x:7.2f}" for x in pass_lb))
            say("    P(pt>=97.8) " + " ".join(f"{x:7.2f}" for x in pass_978))
            say("    P(both)     " + " ".join(f"{x:7.2f}" for x in both))
            r[name] = {"bands": [f"{a}-{b}" for a, b in bands], "dev_point": pt.round(2).tolist(),
                       "p_lb970": pass_lb.round(3).tolist(), "p_pt978": pass_978.round(3).tolist(), "p_both": both.round(3).tolist()}
            if "marginal" in name:
                allb = ((P >= 97.5) & (L >= 97.0)).all(1).mean()
                say(f"    LAB line (every band point >= 97.5 AND LB >= 97.0): P = {allb:.2f}; every band point >= 97.8: P = {(P >= 97.8).all(1).mean():.2f}")
                r[name]["p_lab_line"] = round(float(allb), 3)
            elif "audit" in name:
                deep = ((P[:, 4:] >= 97.8) & (L[:, 4:] >= 97.0)).all(1).mean()
                say(f"    C2 line (15-22 and 23-30: point >= 97.8 AND LB >= 97.0): P = {deep:.2f}; full gate (every band LB >= 97.0): P = {(L >= 97.0).all(1).mean():.2f}")
                r[name]["p_c2_line"] = round(float(deep), 3)
        res[s] = r
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
