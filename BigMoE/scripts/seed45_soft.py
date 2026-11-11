"""SEED45-SOFT (BigMoE, Sun 2026-11-01; follow-up of SEED45: LFU leaves 60 % to Belady at K 266): do the router's NEAR
MISSES predict reuse? Experts that almost made a row's top-10 in this check are likely picked soon; keep them longer.
Same chained replay as scripts/seed45_replay.py (cap45, prompts 2-6 counted, res0 forced in per prompt). Router logits are
rebuilt from the captured MoE inputs x and routers.npz (layers 1-47; layer 0 has no saved router: plain LFU there);
first a check that the rebuilt top-10 matches the captured ids.
near[e] per check and layer = max over the 16 rows of p_row[e] / (the row's 10th-largest p), capped at 1 (1 = picked).
Policies: LFU (default, d 0.99); SOFT g,a: key = decayed hits + g * decayed sum of near^a over checks (unpicked experts
included, resident or not); +PRIOR 30 (SEED45's best) on top.
Lines (written before the run): PASS = the best SOFT cuts misses per kept token >= 10 % vs LFU at K 266 (C 240), pooled over
prompts 2-6, with no prompt worse by > 5 %; KILL = < 5 %; BETWEEN otherwise.
usage: VECLIB_MAXIMUM_THREADS=4 taskpolicy -b ~/BigMoE/.venv/bin/python scripts/seed45_soft.py -> data/seed45/seed45_soft.txt"""
import heapq, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
OUT = os.path.expanduser("~/BigMoE/data/seed45")
NL, E = 48, 512
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


if os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
    sys.exit("PAUSE set: not running")
t0 = time.time()
RW = np.load(os.path.join(CAP, "routers.npz"))
W = {l: RW[f"w{l}"].astype(np.float32) for l in range(1, NL)}
P, agree, tot = [], 0, 0
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, xoff, ids, X = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"need": {}, "near": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["need"]:
            c["kept"] = r                                            # the commit replay: kept rows
            continue
        sel = ids[roff[n]:roff[n] + r]
        c["need"][l] = frozenset(np.unique(sel).tolist())
        if l == 0:
            continue
        x = (X[xoff[n]:xoff[n] + r].astype(np.uint32) << 16).view(np.float32)   # bf16 bits -> fp32
        lg = x @ W[l].T
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        top = np.argsort(-lg, 1)[:, :10]
        agree += sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top, sel)); tot += sel.size
        thr = np.take_along_axis(p, top[:, 9:10], 1)
        c["near"][l] = np.minimum((p / thr).max(0), 1.0).astype(np.float32)
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    res0 = [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]
    P.append((name, cks, res0))
    del X, z
say(f"SEED45-SOFT: {sum(len(c) for _, c, _ in P)} checks; rebuilt router top-10 vs captured ids: {100 * agree / tot:.2f} % "
    f"of picks agree ({time.time() - t0:.0f} s)")
POP = []
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, list(c["need"][l])] += 1
            n += len(cks)
    POP.append(pop / n)


def replay(K, g=0.0, a=1.0, b=0.0):
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st, soft, w = {}, np.zeros(E), 1.0
        for p, (_, cks, res0) in enumerate(P):
            items = [(-1, res0[l], None)] + [(t, c["need"][l], c["near"].get(l)) for t, c in enumerate(cks)]
            pop = POP[p][l]
            for t, need, near in items:
                w /= 0.99
                if t >= 0:
                    miss[p][t] += sum(1 for e in need if e not in st)
                for e in need:
                    st[e] = st.get(e, 0.0) + w
                if g and near is not None:
                    soft += w * (near ** a)
                if len(st) <= K:
                    continue
                key = lambda e: st[e] + g * soft[e] + b * w * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=key):
                    del st[e]
    return miss


def pooled(miss):
    return sum(miss[p].sum() for p in range(1, len(P))) / sum(sum(c["kept"] for c in P[p][1]) for p in range(1, len(P)))


ARMS = [("lfu", 0, 1, 0)] + [(f"soft{g}^{a}", g, a, 0) for g in (0.3, 1, 3) for a in (1, 4)] + \
       [(f"soft{g}^4+prior30", g, 4, 30) for g in (1, 3)] + [("prior30", 0, 1, 30)]
res = {}
for K in (246, 266, 287):
    for nm, g, a, b in ARMS:
        res[(K, nm)] = replay(K, g, a, b)
    base = pooled(res[(K, "lfu")])
    say(f"K {K}: " + ", ".join(f"{nm} {pooled(res[(K, nm)]):.1f} ({100 * (pooled(res[(K, nm)]) / base - 1):+.1f} %)"
                              for nm, *_ in ARMS))
soft = [nm for nm, *_ in ARMS if nm.startswith("soft")]
best = min(soft, key=lambda nm: pooled(res[(266, nm)]))
gain = 1 - pooled(res[(266, best)]) / pooled(res[(266, "lfu")])
worst = max(res[(266, best)][p].mean() / res[(266, "lfu")][p].mean() - 1 for p in range(1, len(P)))
for p in range(1, len(P)):
    say(f"  {P[p][0][:30]:30s} K 266 misses per check: LFU {res[(266, 'lfu')][p].mean():.0f}, {best} {res[(266, best)][p].mean():.0f}")
call = "PASS" if gain >= 0.10 and worst <= 0.05 else ("KILL" if gain < 0.05 else "BETWEEN")
say(f"Lines: best {best} cuts misses per kept token {100 * gain:.1f} % vs LFU at K 266 (worst prompt {100 * worst:+.1f} %) -> {call}")
say(f"({time.time() - t0:.0f} s)")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "seed45_soft.txt"), "w").write("\n".join(lines) + "\n")
