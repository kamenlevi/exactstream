"""CALSTOP (Breakthrough Sun 1 Nov 11:5x; lines in SCOREBOARD 'CALSTOP LINES', written before the run; CPU): a calibrated
stop rule for the confidence-truncated chain (published: SpecDec++ / AdaEDL / DISCO's dynamic draft length) vs a flat
running-confidence threshold t (the 45 GB default's rule, t 0.3 live), on the TREES lattices (data/trees/lattice_*.npz).
q(c, d) = P(the chain is kept through draft d | running confidence bin c, depth group d), Laplace-smoothed, fitted on half
the answers. Per check: n drafts maximizing (1 + sum_{j<n} q_j) - lambda x cost(n + 1), q made non-increasing along the
check; lambda = the rule's own tokens per cost on the fitting half (Dinkelbach). Scored on the other half (2-fold over
answers, both ways) against the best flat t chosen on the fitting half. Per domain (q, lambda, t all fitted per domain) and
pooled (one q, lambda, t for every domain: deployable without knowing the domain).
Usage: [BigMoE venv] python calstop.py -> stdout + data/trees/calstop.txt"""
import os
import numpy as np

D = os.path.expanduser("~/Breakthrough/data/trees")
Z = [np.load(f"{D}/lattice_{t}.npz") for t in ("code", "chat", "rest")]
cat = lambda k: np.concatenate([z[k] for z in Z])
PICK, TRUE, NT, SPROB, DOM = cat("pick"), cat("true"), cat("ntrue"), cat("sel_prob").astype(np.float64), cat("dom")
SAMPLE = np.concatenate([z["sample"] + 1000 * i for i, z in enumerate(Z)])
B, DEP = PICK.shape
CAP = np.where(NT == DEP, DEP + 1, NT)
KC = np.array([(lambda h: int(np.argmin(h)) if not h.all() else len(h))(PICK[b, :NT[b]] == TRUE[b, :NT[b]]) for b in range(B)])
CP = np.cumprod(SPROB[:, :, 0], axis=1)                               # [B, 15] running confidence after draft d
CE = np.array([0, .02, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .98, 1.01])
DG = np.array([0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4])            # depth groups: 0 / 1 / 2-3 / 4-7 / 8-14
CB = np.clip(np.searchsorted(CE, CP, side="right") - 1, 0, len(CE) - 2)
NR = np.arange(DEP + 1)                                               # drafts verified: 0..15
MODELS = {"45GB": lambda r: 65.9 + 14.19 * r, "GB": lambda r: np.interp(r, [1, 2, 4, 8, 12, 16], [1.40, 2.29, 3.82, 6.29, 8.32, 10.03])}
TGRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


def kept_rows(m, n):                                                  # n drafts per check (array over m's blocks)
    return np.minimum(np.minimum(KC[m], n) + 1, CAP[m]), np.minimum(n + 1, CAP[m])


def flat_n(m, t):
    return (CP[m] >= t).sum(1)


def fit_q(m):
    """q[c, g] from the blocks in m: kept through draft d at running-confidence bin c, depth group g (only drafts with a true
    token behind them); cells with < 20 samples fall back to the confidence bin alone."""
    hit, cnt = np.zeros((len(CE) - 1, 5)), np.zeros((len(CE) - 1, 5))
    for d in range(DEP):
        v = m & (NT > d)
        np.add.at(hit, (CB[v, d], DG[d]), (KC[v] > d).astype(float)); np.add.at(cnt, (CB[v, d], DG[d]), 1.0)
    q = (hit + 1) / (cnt + 2)
    q1 = (hit.sum(1) + 1) / (cnt.sum(1) + 2)
    return np.where(cnt >= 20, q, q1[:, None])


def rule_n(m, q, lam, f):
    qq = np.minimum.accumulate(q[CB[m], DG[None, :]], axis=1)         # [nb, 15], non-increasing along the check
    gain = np.concatenate([np.zeros((m.sum(), 1)), np.cumsum(qq, 1)], 1) + 1    # E[kept] for 0..15 drafts
    return np.argmax(gain - lam * f(NR + 1.0)[None, :], axis=1)


def rate(m, n, f):
    k, r = kept_rows(m, n)
    return k.sum() / f(r.astype(float)).sum()


def fit(m, f):                                                        # -> (q, lambda, best flat t) on the fitting set m
    q = fit_q(m)
    lam = rate(m, flat_n(m, 0.3), f)
    for _ in range(30):
        new = rate(m, rule_n(m, q, lam, f), f)
        if abs(new - lam) < 1e-9 * lam:
            break
        lam = new
    t = max(TGRID, key=lambda t: rate(m, flat_n(m, t), f))
    return q, lam, t


GROUPS = [("code", ("tf_code_HumanEval",)), ("chat (ifeval + multi)", ("tf_ifeval", "tf_multi")), ("agent", ("agent",)),
          ("mgsm", ("mgsm",)), ("gsm (reference)", ("tf_gsm",))]
FOLD = SAMPLE % 2
say(f"CALSTOP: {B} blocks; 2-fold over answers; tokens per cost on the held-out halves (both folds summed); t 0.3 = live")
res = {}
for mname, f in MODELS.items():
    say(f"\n== cost model {mname}")
    for scope in ("per domain", "pooled"):
        for gname, doms in GROUPS:
            g = np.isin(DOM, doms)
            K = {k: 0.0 for k in ("rule", "flat", "live")}; C = dict(K); info = []
            for a in (0, 1):
                fitm = (FOLD == a) & (g if scope == "per domain" else True)
                ev = (FOLD != a) & g
                q, lam, t = fit(fitm, f)
                info.append(f"t {t}")
                for k, n in (("rule", rule_n(ev, q, lam, f)), ("flat", flat_n(ev, t)), ("live", flat_n(ev, 0.3))):
                    kk, rr = kept_rows(ev, n); K[k] += kk.sum(); C[k] += f(rr.astype(float)).sum()
            r = {k: K[k] / C[k] for k in K}
            res[(mname, scope, gname)] = 100 * (r["rule"] / r["flat"] - 1)
            say(f"    {scope:10s} {gname:22s} rule vs the fitted flat t ({' / '.join(info)}): {100 * (r['rule'] / r['flat'] - 1):+5.1f} %; "
                f"rule vs live t 0.3: {100 * (r['rule'] / r['live'] - 1):+5.1f} %; flat vs live: {100 * (r['flat'] / r['live'] - 1):+5.1f} %")
ok = lambda thr: all(res[(mn, "per domain", gn)] >= thr for mn in MODELS for gn in ("code", "chat (ifeval + multi)"))
low = all(res[(mn, "per domain", gn)] < 1.0 for mn in MODELS for gn in ("code", "chat (ifeval + multi)"))
say("\nVERDICT (per domain; PASS >= +3 % over the fitted flat t on chat AND code under both models, KILL < +1 % on all four): "
    + ("PASS" if ok(3.0) else "KILL" if low else "BETWEEN"))
open(f"{D}/calstop.txt", "w").write("\n".join(lines) + "\n")
