"""CHOOSER WITH HEAD-SIDE FEATURES (NewIdeas, Sun 15 Nov 21:5x; the orchestrator's 21:5x Q3, "if a chooser is truly
closed, say so"). MoonShots' learned chooser (week98/chooser.py, result E) used draft-time 2B-side features only; its
docstring says head-side features were not in the dumps. They are now: BigMoE's TOPK dump gives the head's own top-8
ids and probabilities for every cell (topk_t30b_*.npz, aligned with the cells). Draft-time only; never the 125B's
log-probs (cells m / dlp / rank / y_rank / p_y).
Features per cell: depth bins (0, 1-2, 3-6, 7-14, 15-30); the head's top-1 and top-2 probability (p1, p2, log p1,
log(p1 / p2)); the auditor's margin (and x depth bin); the head's probability of the auditor's token (0 when not in its
top 8) and an in-top-8 flag. L2 logistic regression on the decision cells (exactly one of head / auditor right; y = the
auditor is right); override where P > 0.5.
Protocols:
  A (MoonShots'): train on MBPP's first 50 files, test on MBPP's other 50 and all 24 HE+ windows.
  B (in-domain): HE+ 2-fold cross-fit by window (odd / even windows), MBPP likewise.
Read-outs: marginal bands (information) and the walk (CERTIFY98: kept/32, walk-strict bands) at P = 1 (the first
override in a check is a repair if right, a break if wrong; MoonShots' walk) and P = inf; the frozen tau rule and the
head-OR-auditor ceiling beside it.
LINE (written before the run): ALIVE if, out of sample on HE+, kept/32 at P = 1 >= 1.03 x the frozen rule's AND marginal
23-30 >= the frozen rule's + 1.0 pt; else CLOSED (head-side features included).
usage: ~/BigMoE/.venv/bin/python -I -B chooser_headside.py"""
import os, sys, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/NewIdeas/research/week98"))
import tree_audit as TA

OUT = os.path.expanduser("~/NewIdeas/research/week98/chooser_headside")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14), (15, 22), (23, 30)]
DBINS = [(0, 0), (1, 2), (3, 6), (7, 14), (15, 30)]
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)


def load(s):
    tag, rtag, pub32, pub16 = TA.SETS[s]
    W = TA.load_windows(tag, pub32)
    for aud in ("2b", "4b"):
        TA.attach_aud(W, f"{TA.MS}/rows{aud}_{rtag}.npz", aud)
    tz = np.load(f"{TA.CE}/topk_t30b_{tag}.npz", allow_pickle=True)
    cz = np.load(f"{TA.CE}/cells_t30b_{tag}.npz", allow_pickle=True)
    for k in ("depth", "fi", "row", "right"):
        assert (tz[k] == cz[k]).all(), f"topk / cells differ in {k}"
    ids, tp, rows, fi, dep = tz["top_ids"], tz["top_p"], tz["row"], tz["fi"], tz["depth"]
    key = {(int(f), int(r), int(d)): i for i, (f, r, d) in enumerate(zip(fi, rows, dep))}
    for X in W:
        n = X["n"]; X["IDS"] = np.full((n, 31, 8), -1, np.int64); X["TP"] = np.zeros((n, 31, 8), np.float32)
        for r in range(n - 1):
            for d in range(31):
                if X["rt"][r, d] < 0:
                    continue
                i = key[(X["fi"], int(X["RW"][r, d]), d)]
                X["IDS"][r, d] = ids[i]; X["TP"][r, d] = tp[i]
    return W


def feats(X, aud):
    n = X["n"]; F = []
    p1 = np.clip(X["TP"][:, :, 0], 1e-6, 1); p2 = np.clip(X["TP"][:, :, 1], 1e-6, 1)
    t2 = X[f"T2_{aud}"]; g = np.nan_to_num(X[f"G2_{aud}"], nan=0.0)
    inn = (X["IDS"] == t2[:, :, None]); ph = np.where(inn, X["TP"], 0).sum(-1); in8 = inn.any(-1).astype(np.float32)
    d = np.broadcast_to(np.arange(31)[None, :], (n, 31))
    F.append(np.ones((n, 31)))
    for a, b in DBINS:
        F.append(((d >= a) & (d <= b)).astype(np.float32))
    F += [p1, p2, np.log(p1), np.log(p1 / p2), g, np.minimum(g, 8.0), ph, np.log(np.clip(ph, 1e-6, 1)), in8]
    for a, b in DBINS:
        F.append(g * ((d >= a) & (d <= b)))
        F.append(np.log(p1) * ((d >= a) & (d <= b)))
    return np.stack(F, -1).astype(np.float64)


def fit(Xs, ys, lam=1e-2, iters=60):
    w = np.zeros(Xs.shape[1])
    mu = Xs.mean(0); sd = Xs.std(0); sd[sd < 1e-9] = 1; mu[0] = 0; sd[0] = 1
    Z = (Xs - mu) / sd
    for _ in range(iters):                              # Newton / IRLS
        p = 1 / (1 + np.exp(-Z @ w)); Wt = p * (1 - p)
        g = Z.T @ (p - ys) + lam * w; Hh = (Z * Wt[:, None]).T @ Z + lam * np.eye(len(w))
        w -= np.linalg.solve(Hh, g)
    return w, mu, sd


def decide(model, F):
    w, mu, sd = model
    return (1 / (1 + np.exp(-((F - mu) / sd) @ w))) > 0.5


def train(W, idx, aud):
    Xs, ys = [], []
    for i in idx:
        X = W[i]; F = feats(X, aud); h, a = X["rt"], X[f"A2_{aud}"]
        m = (h >= 0) & (a >= 0) & (h != a)
        Xs.append(F[m]); ys.append((a[m] == 1).astype(np.float64))
    return fit(np.concatenate(Xs), np.concatenate(ys))


def evaluate(W, idx, aud, over_of):
    """over_of(i) -> bool [n, 31]: override the head with the auditor's token at that cell"""
    num = np.zeros(31); den = np.zeros(31); comb_m = []
    res = {}
    for P in (1, 99):
        ks, wn, wd = [], np.zeros(31), np.zeros(31)
        for i in idx:
            X = W[i]; h, a = X["rt"], X[f"A2_{aud}"]; ov = over_of(i)
            p = 1
            while p < X["n"]:
                k = used = 0
                while k < 31:
                    hv = h[p, k]
                    if hv == -1:
                        break
                    if used >= P or not ov[p, k]:
                        ok = hv == 1
                    else:
                        ok = a[p, k] == 1
                        if ok and hv == 0:
                            used += 1
                        elif ok and hv == 1:
                            pass
                        elif not ok:
                            ok = False
                    if not ok:
                        if k < 31:
                            wd[k] += 1
                        break
                    wn[k] += 1; wd[k] += 1; k += 1
                ks.append(k + 1); p += k + 1
        res[f"P{P if P == 1 else 'inf'}"] = {"kept32": round(float(np.mean(ks)), 3),
            "walk_strict": [round(100 * wn[a_:b_ + 1].sum() / wd[a_:b_ + 1].sum(), 2) if wd[a_:b_ + 1].sum() else None for a_, b_ in BANDS]}
    for i in idx:
        X = W[i]; h, a = X["rt"][:-1], X[f"A2_{aud}"][:-1]; ov = over_of(i)[:-1]
        c = np.where(ov, a, h); v = h >= 0
        num += ((c == 1) & v).sum(0); den += v.sum(0)
    res["marginal"] = [round(100 * num[a_:b_ + 1].sum() / den[a_:b_ + 1].sum(), 2) for a_, b_ in BANDS]
    return res


def show(name, r):
    f = lambda v: " ".join(f"{x:5.1f}" if x is not None else "    -" for x in v)
    say(f"  {name:34s} kept/32 P1 {r['P1']['kept32']:6.2f} Pinf {r['Pinf']['kept32']:6.2f} | marginal {f(r['marginal'])} | "
        f"walk-strict P1 {f(r['P1']['walk_strict'])}")


def main():
    TA.say = lambda s="": None
    D = {s: load(s) for s in ("he", "mbpp")}
    out = {}
    for aud in ("2b", "4b"):
        frozen = lambda W, aud=aud: (lambda i: (~np.isnan(W[i][f"G2_{aud}"])) & (np.nan_to_num(W[i][f"G2_{aud}"], nan=-1) >= TA.TAU[None, :]))
        oracle = lambda W, aud=aud: (lambda i: (W[i]["rt"] == 0) & (W[i][f"A2_{aud}"] == 1))
        never = lambda W: (lambda i: np.zeros(W[i]["rt"].shape, bool))
        # protocol A: MBPP first 50 -> MBPP other 50 + HE+ 24
        mb, he = D["mbpp"], D["he"]
        order = np.argsort([X["file"] for X in mb]); tr = list(order[:50]); te = list(order[50:])
        mA = train(mb, tr, aud)
        say(f"\n=== auditor {aud}, protocol A (train MBPP first 50; MoonShots' split) ===")
        for setname, W, idx in (("HE+ 24", he, range(len(he))), ("MBPP test 50", mb, te)):
            say(f" {setname}:")
            rs = {"head": evaluate(W, idx, aud, never(W)), "frozen tau": evaluate(W, idx, aud, frozen(W)),
                  "chooser (head-side)": evaluate(W, idx, aud, (lambda i, W=W: decide(mA, feats(W[i], aud)))),
                  "head OR auditor (ceiling)": evaluate(W, idx, aud, oracle(W))}
            for k, v in rs.items():
                show(k, v)
            out[f"{aud} A {setname}"] = rs
        # protocol B: in-domain 2-fold cross-fit by window
        for s in ("he", "mbpp"):
            W = D[s]; n = len(W); f0 = list(range(0, n, 2)); f1 = list(range(1, n, 2))
            m0, m1 = train(W, f1, aud), train(W, f0, aud)
            over = {}
            for i in f0:
                over[i] = decide(m0, feats(W[i], aud))
            for i in f1:
                over[i] = decide(m1, feats(W[i], aud))
            say(f"=== auditor {aud}, protocol B ({s}, 2-fold cross-fit by window) ===")
            rs = {"head": evaluate(W, range(n), aud, never(W)), "frozen tau": evaluate(W, range(n), aud, frozen(W)),
                  "chooser (head-side, cross-fit)": evaluate(W, range(n), aud, lambda i: over[i]),
                  "head OR auditor (ceiling)": evaluate(W, range(n), aud, oracle(W))}
            for k, v in rs.items():
                show(k, v)
            out[f"{aud} B {s}"] = rs
    say("\nLINE (HE+, out of sample): ALIVE if kept/32 P1 >= 1.03 x frozen AND marginal 23-30 >= frozen + 1.0")
    for key in ("2b A HE+ 24", "2b B he", "4b A HE+ 24", "4b B he"):
        r = out[key]; ch = [v for k, v in r.items() if k.startswith("chooser")][0]; fr = r["frozen tau"]
        k_ok = ch["P1"]["kept32"] >= 1.03 * fr["P1"]["kept32"]; b_ok = ch["marginal"][-1] >= fr["marginal"][-1] + 1.0
        say(f"  {key}: kept P1 {ch['P1']['kept32']:.2f} vs {fr['P1']['kept32']:.2f} (x{ch['P1']['kept32'] / fr['P1']['kept32']:.3f}); "
            f"23-30 {ch['marginal'][-1]:.2f} vs {fr['marginal'][-1]:.2f} -> {'ALIVE' if (k_ok and b_ok) else 'CLOSED'}")
    json.dump(out, open(OUT + ".json", "w"), indent=1, default=float)
    open(OUT + ".txt", "w").write("bands 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
