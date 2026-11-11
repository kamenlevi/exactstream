"""D-24 read-out (MTP Head, Sun 11-01, CPU): tree / cut policies for the MTP chain scored on D-24's teacher-forced ranks.
Per anchor s and depth j (data/d24/<set>_<run>.npz): the head's top-1 p and the true token's rank r_j (0 = its argmax hits).
Up to the first miss k* (the first r_j != 0) the teacher-forced chain is the real chain; a branch holding the true token
at k* continues as the forced chain does (its inputs are the true tokens), so a rescued branch's continuation is exact too.
Policies (per check: kept = accepted drafts + the bonus, rows = 1 + drafted rows):
  chain K      drafts 1..K                                   (kept = min(k*, K + 1))
  cut t        drafts while the running product of top-1 p >= t (the 45 GB default's rule), K max
  cut + leaf   the cut chain plus leaf rows holding the rank-1 (and rank-2) candidates at each kept depth whose top-1 p
               < tau: a first miss at a depth with leaves holding the true token gains that token + the bonus (+1)
  oracle w     a tree of width w at every depth (accept while r_j < w): the upper bound, no row budget
Walked over each answer as D-17 (the pass at output position p uses anchor gen + p - 2), tokens per pass. Rows are averaged
over the same passes. Cost per check (ms) from D-23 on the 45 GB default at C 224 (chat): F + 7.82 x rows + 2.2 x drafted
steps (early exit), F fitted from D-23's mtp7 chat checks; leaves need no extra draft steps (same logits) but a tree-aware
verify (GDN state per leaf parent) whose cost is assumed = a chain row. tok/s ratios are vs the cut chain.
Output: data/d24_read.txt"""
import os, json, glob
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
RUN = os.environ.get("D24_RUN", "b5")
D = 15
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

# ---- cost model from D-23's mtp7 chat checks: time per check = F + SLOPE x rows + STEP x drafted steps
SLOPE, STEP = 7.82, 2.2
R = [json.loads(l) for l in open(os.path.join(TD, "d23_runs.jsonl"))]
Fs = {}
for typ in ("chat", "math", "code"):
    v = []
    for r in R:
        if r["arm"] == "mtp7" and r["type"] == typ and not (r["cut"] or r["paused"] or r["invalid"]):
            pc = np.array(r["pc"], float)
            dec = r["n"] / r["tok_s"]
            ov = (dec - 1e-3 * (pc[:, 2] + pc[:, 3]).sum()) / len(pc) * 1e3
            v.append(np.mean(pc[:, 2] + pc[:, 3] + ov - SLOPE * pc[:, 0] - STEP * (pc[:, 0] - 1)))
    Fs[typ] = float(np.mean(v)) if v else float("nan")
say(f"D-24 read-out, head {RUN}: MTP chain policies on teacher-forced ranks (held-out answers); cost model from D-23 (45 GB "
    f"default, C 224): F + {SLOPE} ms x rows + {STEP} ms x drafted steps; F = " + ", ".join(f"{k} {v:.1f}" for k, v in Fs.items()) + " ms")


def per_anchor(p1, rk, policy):
    """kept, rows, steps per anchor for one policy"""
    n = len(rk)
    kept = np.zeros(n); rows = np.zeros(n); steps = np.zeros(n)
    for s in range(n):
        r, p = rk[s], p1[s]
        valid = r >= 0
        L = int(valid.sum())                                     # depths available (the answer's end)
        if L == 0:
            kept[s], rows[s] = 1, 1
            continue
        miss = np.where(r[:L] != 0)[0]
        kstar = int(miss[0]) + 1 if len(miss) else L + 1         # first miss depth (1-based); L + 1 = none within reach
        kind, a = policy[0], policy[1:]
        if kind == "chain":
            m = min(a[0], L)
            kept[s] = min(kstar, m + 1); rows[s] = m + 1; steps[s] = m
        elif kind == "oracle":
            w = a[0]
            bad = np.where(r[:L] >= w)[0]
            acc = int(bad[0]) if len(bad) else L                 # accepted with a width-w tree at every depth
            kept[s] = acc + 1; rows[s] = np.nan; steps[s] = np.nan
        else:                                                    # cut / leaf
            K, t = a[0], a[1]
            cp = np.cumprod(np.nan_to_num(p[:min(K, L)], nan=0.0))
            m = int((cp >= t).sum())
            k0 = min(kstar, m + 1)
            extra = 0
            if kind == "leaf":
                tau, w = a[2], a[3]
                leafd = [j for j in range(1, m + 1) if p[j - 1] < tau]
                extra = len(leafd) * w
                if kstar <= m and kstar in leafd and 1 <= r[kstar - 1] <= w:
                    k0 = kstar + 1
            kept[s] = k0; rows[s] = m + 1 + extra
            steps[s] = min(m + 1, min(K, L))                     # early exit: the failing step is evaluated too
    return kept, rows, steps


def walk(kept, rows, steps, gen, ids, a0):
    """D-17's walk: the pass at output position p uses anchor s = gen + p - 2 (row s - a0); returns tokens, passes, rows, steps"""
    L = len(ids) - gen
    p, tok, passes, rw, st = 1, 0, 0, 0.0, 0.0
    while p < L:
        i = gen + p - 2 - a0
        if not (0 <= i < len(kept)):
            k, r_, s_ = 1, 1, 0
        else:
            k, r_, s_ = int(min(kept[i], L - p)), rows[i], steps[i]
        k = max(k, 1)
        tok += k; passes += 1; rw += r_; st += s_; p += k
    return tok, passes, rw, st


POL = [("chain", 7), ("chain", 15), ("cut", 7, 0.3), ("cut", 15, 0.3),
       ("leaf", 7, 0.3, 0.5, 1), ("leaf", 7, 0.3, 0.5, 2), ("leaf", 7, 0.3, 0.8, 1), ("leaf", 7, 0.3, 0.8, 2),
       ("leaf", 15, 0.3, 0.5, 1), ("leaf", 15, 0.3, 0.8, 2), ("oracle", 2), ("oracle", 3)]
name = lambda q: {"chain": lambda: f"chain K{q[1]}", "cut": lambda: f"cut K{q[1]} t {q[2]}",
                  "leaf": lambda: f"cut K{q[1]} + {q[4]} leaf if p < {q[3]}", "oracle": lambda: f"oracle width {q[1]}"}[q[0]]()
TYPE = {"ifeval": "chat", "multi": "chat", "mgsm": "math", "gsm": "math", "code": "code"}
for st in ("ifeval", "multi", "mgsm", "gsm", "code"):
    f = os.path.join(TD, "d24", f"{st}_{RUN}.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f)
    keys, gens = z["keys"], z["gen"]
    nA = len(keys)
    d1 = np.concatenate([z[f"rk_{i}"][:, 0] for i in range(nA)]); d1 = d1[d1 >= 0]
    fm = []                                                     # the true token's rank at each anchor's first miss
    for i in range(nA):
        rk = z[f"rk_{i}"]
        for s in range(len(rk)):
            v = rk[s][rk[s] >= 0]
            mm = np.where(v != 0)[0]
            if len(mm):
                fm.append(v[mm[0]])
    fm = np.array(fm)
    say(f"\n== {st} ({nA} answers): depth 1 rank 0 / <=1 / <=2 = {np.mean(d1 == 0):.3f} / {np.mean(d1 <= 1):.3f} / "
        f"{np.mean(d1 <= 2):.3f}; at first misses (any depth) the true token is the head's #2 {np.mean(fm == 1):.2f}, "
        f"#3 {np.mean(fm == 2):.2f} (top-3 rescues {np.mean(fm <= 2):.2f}; DFlash2 at chat first misses 0.34, R-61N)")
    F = Fs[TYPE[st]]
    base = None
    for q in POL:
        T_, P_, RW, SP = 0, 0, 0.0, 0.0
        for i in range(nA):
            ids = z[f"ids_{i}"]; gen = int(gens[i]); a0 = max(0, gen - 2)
            kept, rows, steps = per_anchor(z[f"p1_{i}"], z[f"rk_{i}"], q)
            t, p, rw, sp = walk(kept, rows, steps, gen, ids, a0)
            T_ += t; P_ += p; RW += rw; SP += sp
        tpp, rpp, spp = T_ / P_, RW / P_, SP / P_
        if q[0] == "oracle":
            say(f"  {name(q):28s} tokens per pass {tpp:5.2f} (upper bound, no row budget)")
            continue
        ms = F + SLOPE * rpp + STEP * spp
        tps = tpp / ms * 1e3
        if q == ("cut", 7, 0.3):
            base = tps
        rel = f" ({tps / base - 1:+.1%} vs cut K7)" if base else ""
        say(f"  {name(q):28s} tokens per pass {tpp:5.2f}, rows {rpp:5.2f}, drafted steps {spp:5.2f}: est. {ms:5.1f} ms per check, "
            f"{tps:5.1f} tok/s{rel}")
open(os.path.join(TD, "d24_read.txt"), "w").write("\n".join(lines) + "\n")
