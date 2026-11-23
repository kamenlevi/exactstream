"""HPD read-out (MTP Head, Mon 11-02; Fable ~/FableGod/FINDINGS.md MISSION 8 sec 3: "measure held-out accuracy on the
uncertain code positions ... dies if it moves < +0.01"). CPU only, on CF walks (sticky/d02b_walk.py with TL_WALK_CF=1,
block 16, official 1-step loop) of the held-out sets: code (dfl2d_code, 24 HumanEval+), math (dfl2d_math, 60 GSM8K), multi
(D-02a, 24) and IFEval (D-02a, 30); none is in any training set used here.
- Base = L1-20's CF walk at its own anchors (tag HPD_BASE, default _hpd0). Its uncertain positions U(T) = the depths it
  reached at each check (1 .. accepted + 1, the first miss included, capped at the checkable drafts) whose cf < T
  (T = 0.999, Fable's; 0.99 beside it); sure positions S(T) = the reached ones with cf >= T.
- Each arm (HPD_ARMS=name:tag,...) walked at L1-20's anchors (TL_WALK_ANCHORS=_hpd0, tag <tag>_fx) is scored at exactly
  those positions with its teacher-forced picks (the true predecessor at every depth, so a pick does not depend on the
  arm's own earlier misses): paired accuracy on U and S, the paired kept per check at the same anchors, and (if the arm's
  own walk, tag <tag>, exists) its own kept per check. 95 % intervals by resampling the samples (2,000 draws, seed 0).
Line (Fable's, the orchestrator's Monday kill line): KILL = arm H's code accuracy on U(0.999) - L1-20's < +0.01; else
ALIVE. The selection's own share = H - U (the uniform arm, same data and steps) on the same positions.
Output: data/hpd_read.txt (appended)."""
import os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
BASE = os.environ.get("HPD_BASE", "_hpd0")
ARMS = [tuple(x.split(":")) for x in os.environ.get("HPD_ARMS", "H:_hpdH,U:_hpdU").split(",") if x]
SETS = os.environ.get("HPD_SETS", "code,math,multi,ifeval").split(",")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def load(tag, st):
    f = os.path.join(TD, f"d02b_walk{tag}_{st}.npz")
    return np.load(f) if os.path.exists(f) else None

def positions(Z):
    """per sample: list of (check index, depth 1-based, true token, cf) over the reached depths"""
    out = []
    for s in range(len(Z["names"])):
        ids, off, ch, cf = Z[f"{s}_ids"], int(Z[f"{s}_off"]), Z[f"{s}_checks"], Z[f"{s}_dcf"]
        rows = []
        for c, (k, m, a) in enumerate(ch):
            for dd in range(1, min(int(k) + 1, int(m)) + 1):
                rows.append((c, dd, int(ids[a - off + dd]), float(cf[c, dd - 1])))
        out.append(rows)
    return out

def score(Z, pos, key):
    """per sample: right (1 / 0) of the picks in Z[key] at the given positions"""
    return [np.array([int(Z[f"{s}_{key}"][c, dd - 1] == t) for c, dd, t, _ in rows], np.int64) for s, rows in enumerate(pos)]

def ci(per_s_diff, per_s_n, rng):
    """95 % interval of a pooled mean difference by resampling samples"""
    S = len(per_s_n); draws = []
    for _ in range(2000):
        ix = rng.integers(0, S, S)
        n = sum(per_s_n[i] for i in ix)
        draws.append(sum(per_s_diff[i] for i in ix) / max(n, 1))
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)

rng = np.random.default_rng(0)
say(f"HPD read-out: base L1-20 ({BASE}) vs " + ", ".join(f"{n} ({t})" for n, t in ARMS) + "; block 16, held-out")
res = {}
for st in SETS:
    B = load(BASE, st)
    if B is None:
        say(f"  {st}: no base walk"); continue
    P = positions(B)
    tf_ok = sum(int((B[f"{s}_ttok"][c, dd - 1] == B[f"{s}_dtok"][c, dd - 1])) for s, rows in enumerate(P) for c, dd, _, _ in rows)
    n_all = sum(len(r) for r in P)
    kb = np.mean([int(k) + 1 for s in range(len(B["names"])) for k, _, _ in B[f"{s}_checks"]])
    say(f"  {st}: {len(P)} samples, {sum(len(B[f'{s}_checks']) for s in range(len(P)))} checks, {n_all} reached positions; "
        f"L1-20 kept per check {kb:.2f}; self-check teacher-forced pick == own pick at reached depths {tf_ok}/{n_all}")
    base_r = score(B, P, "dtok")
    for T in (0.999, 0.99):
        uq = [np.array([cf < T for _, _, _, cf in rows], bool) for rows in P]
        nU = sum(int(u.sum()) for u in uq); nS = n_all - nU
        b_u = sum(int(r[u].sum()) for r, u in zip(base_r, uq)) / max(nU, 1)
        b_s = sum(int(r[~u].sum()) for r, u in zip(base_r, uq)) / max(nS, 1)
        say(f"    T {T}: uncertain share {nU / max(n_all, 1):.3f} (n {nU}); L1-20 accuracy uncertain {b_u:.4f} / sure {b_s:.4f}")
        for name, tag in ARMS:
            X = load(tag + "_fx", st)
            if X is None:
                continue
            assert all(np.array_equal(X[f"{s}_checks"][:, 2], B[f"{s}_checks"][:, 2]) for s in range(len(P))), "anchors differ"
            xr = score(X, P, "ttok")
            a_u = sum(int(r[u].sum()) for r, u in zip(xr, uq)) / max(nU, 1)
            a_s = sum(int(r[~u].sum()) for r, u in zip(xr, uq)) / max(nS, 1)
            lo, hi = ci([int(x[u].sum()) - int(b[u].sum()) for x, b, u in zip(xr, base_r, uq)], [int(u.sum()) for u in uq], rng)
            res[(st, T, name)] = a_u - b_u
            say(f"      {name}: uncertain {a_u:.4f} ({a_u - b_u:+.4f}, 95 % {lo:+.4f} .. {hi:+.4f}) / sure {a_s:.4f} ({a_s - b_s:+.4f})")
    for name, tag in ARMS:
        X = load(tag + "_fx", st)
        if X is None:
            continue
        kx = np.mean([int(k) + 1 for s in range(len(X["names"])) for k, _, _ in X[f"{s}_checks"]])
        O = load(tag, st)
        ko = np.mean([int(k) + 1 for s in range(len(O["names"])) for k, _, _ in O[f"{s}_checks"]]) if O is not None else float("nan")
        say(f"    {name}: kept per check at L1-20's anchors {kx:.2f} ({kx - kb:+.2f}); own walk {ko:.2f} ({ko - kb:+.2f})")
names = [n for n, _ in ARMS]
if ("code", 0.999, "H") in res:
    d = res[("code", 0.999, "H")]
    v = "KILL" if d < 0.01 else "ALIVE"
    sh = f"; selection's own share H - U {d - res[('code', 0.999, 'U')]:+.4f}" if ("code", 0.999, "U") in res else ""
    say(f"  HPD line (code, L1-20-uncertain positions at cf < 0.999, H vs L1-20 >= +0.01): {d:+.4f} -> {v}{sh}")
open(os.path.join(TD, "hpd_read.txt"), "a").write("\n".join(lines) + "\n\n")
