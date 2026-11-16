"""CPU: a learned gate for the uncertainty-gated budget (UGB), CALM-style (Schuster et al. 2207.07061: a small learned
confidence measure decides per step whether the cheap computation is safe). Data: turn15's budget arms (v3 tiers, 4-row
tf passes, b22 / b20 / b18) joined with turn18's v3 arm (bit-identical to turn15's natural arm at every shared position)
for the drafter's depth-1 top-1 log-prob. Passes as the tf runner's: the prompt's last position alone (prefill, never
budgeted), then chunks of 4 from p0 + 1. Features known before a pass (realizable): the drafter's lp1 at the pass's 4
positions (min, mean, sorted), the previous pass's target lp1 (min, mean), margin (min), entropy (max), log position.
Label: the pass's largest excess KLD (budget arm - natural) > EPS. Model: L2 logistic regression (numpy), out-of-fold by
sequence (4 folds). A gated arm uses the budget arm on passes with predicted risk < thr, natural otherwise; the Eval
suite's rules (as rule125 / ugb_rule) per threshold; reported: the largest code coverage at a full PASS, against the
drafter-min gate (DRAFT_EX) on the same pass structure. usage: ugb_learned.py [EPS]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
from evalsuite.report import cluster_boot, quantile_ratio, MARGINS  # noqa: E402

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
D = os.path.expanduser("~/TrainLab/data")
t15 = np.load(f"{D}/budget125_tf_R4_plan4433333333_pos.npz")
t18 = np.load(f"{D}/budget125_tf_R4_plan4433333333_mtp_b2_pos.npz")
ar4 = np.load(os.path.expanduser("~/EvalSuite/results/ar4/tf.npz"))
akey = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(ar4["seq"], ar4["pos"]))}
k18 = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(t18["p4433333333__seq"], t18["p4433333333__pos"]))}
bt, nm = MARGINS["bootstrap"], MARGINS["noninferiority"]
parts = [str(x) for x in t15["parts"]]
seq, pos, part = t15["exact__seq"], t15["exact__pos"], t15["exact__part"]
N = len(seq)
j18 = np.array([k18[(int(s), int(p))] for s, p in zip(seq, pos)])
md = t18["p4433333333__mtp_lp1"][j18].astype(np.float64)
md = np.where(np.isfinite(md), md, -30.0)
ke, ge = t15["exact__kld"].astype(np.float64), t15["exact__agree"].astype(np.float64)
lp, mg, en = (t15[f"exact__{f}"].astype(np.float64) for f in ("lp1", "margin", "ent"))
for arm in ("b22", "b20", "b18"):                                   # every arm at the same positions in the same order
    assert (t15[f"{arm}__seq"] == seq).all() and (t15[f"{arm}__pos"] == pos).all()
aidx = np.array([akey.get((int(s), int(p)), -1) for s, p in zip(seq, pos)])

pid = np.full(N, -1, np.int64)                                       # pass id per position (-1: the prefill position)
plist, prev = [], []
for s in np.unique(seq):
    ix = np.where(seq == s)[0]
    o = ix[np.argsort(pos[ix])]
    first = len(plist)
    for c in range(1, len(o), 4):
        pid[o[c:c + 4]] = len(plist)
        plist.append(o[c:c + 4])
        prev.append(o[max(0, c - 4):c])                              # the positions just before (committed when it runs)
P = len(plist)


def feats():
    X = np.zeros((P, 12))
    for i, (g, h) in enumerate(zip(plist, prev)):
        d = np.sort(md[g])
        d4 = np.concatenate([d, np.full(4 - len(d), d[-1])])
        X[i] = [d.min(), d.mean(), *d4, lp[h].min(), lp[h].mean(), mg[h].min(), en[h].max(), np.log1p(pos[g[0]]), len(g)]
    return X


def fit_lr(X, y, lam=1.0, it=300):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(it):                                              # Newton steps, L2 on the slopes
        p_ = 1 / (1 + np.exp(-Z @ w))
        g_ = Z.T @ (p_ - y) + lam * np.r_[w[:-1], 0]
        H = (Z * (p_ * (1 - p_))[:, None]).T @ Z + lam * np.diag(np.r_[np.ones(len(w) - 1), 0])
        step = np.linalg.solve(H, g_)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w)))


def rules(kv, gv):
    ok_all, res = True, {}
    for pi, pn in enumerate(parts):
        m = (aidx >= 0) & (part == pi)
        e, lo, hi = cluster_boot(kv[m], ar4["kld"][aidx[m]].astype(np.float64), seq[m], bt["resamples"], bt["seed"])
        q, qlo, qhi = quantile_ratio(kv[m], ar4["kld"][aidx[m]].astype(np.float64), seq[m], 0.999, bt["quantile_resamples"], bt["seed"])
        d, dlo, dhi = cluster_boot(gv[m], ar4["agree"][aidx[m]].astype(np.float64), seq[m], bt["resamples"], bt["seed"], stat="diff")
        ok = hi <= nm["kld_mean_ratio_max"] and qhi <= nm["kld_p999_ratio_max"] and dlo * 100 >= -nm["top1_agree_drop_max_points"]
        ok_all &= ok
        res[pn[3:]] = f"{hi:.2f}/{qhi:.2f}/{dlo * 100:+.1f}"
    return ok_all, res


X = feats()
pseq = np.array([seq[g[0]] for g in plist])
useq = np.unique(pseq)
folds = {s: i % 4 for i, s in enumerate(np.random.default_rng(0).permutation(useq))}
fo = np.array([folds[s] for s in pseq])
pcode = np.array([part[g[0]] == parts.index("tf_code") for g in plist])
out = [f"ugb_learned (EPS {EPS}): learned risk gate vs drafter-min gate, 4-row passes; the largest code coverage at a PASS"]
for arm in ("b22", "b20", "b18"):
    kb, gb = t15[f"{arm}__kld"].astype(np.float64), t15[f"{arm}__agree"].astype(np.float64)
    exc = np.array([(kb[g] - ke[g]).max() for g in plist])
    y = (exc > EPS).astype(np.float64)
    risk = np.zeros(P)
    for f in range(4):                                               # out of fold: the gate never sees its sequences
        tr = fo != f
        risk[~tr] = fit_lr(X[tr], y[tr])(X[~tr])
    for name, score in (("learned", -risk), ("drafter-min", X[:, 0])):
        best = None
        for qq in np.linspace(0.05, 0.95, 19):
            thr = np.quantile(score, qq)
            conf_p = score >= thr
            conf = np.zeros(N, bool)
            conf[pid >= 0] = conf_p[pid[pid >= 0]]
            ok, res = rules(np.where(conf, kb, ke), np.where(conf, gb, ge))
            cov = conf_p[pcode].mean()
            if ok and (best is None or cov > best[0]):
                best = (cov, qq, res, {pn[3:]: float(conf_p[np.array([part[g[0]] == pi for g in plist])].mean()) for pi, pn in enumerate(parts)})
        out.append(f"  {arm} {name:11s}: " + (f"code coverage {best[0]:.0%} at a PASS (all parts {', '.join(f'{k} {v:.0%}' for k, v in best[3].items())}; "
                                             f"mean/P99.9/top-1 bounds {best[2]})" if best else "no PASS at any threshold")
                   + f" | label rate {y.mean():.0%}")
print("\n".join(out))
open(f"{D}/ugb_learned_eps{EPS}.txt", "w").write("\n".join(out) + "\n")
