"""SEED45-LRU (BigMoE, Sun 2026-11-01; after SEED45: the live 45 GB loop runs LRU, which beats decayed LFU d 0.99 by ~9 % in
the cap45 replay, and LFU + PRIOR 30 only matches LRU): does the popularity prior help on top of LRU, or does a faster LFU
decay? Same chained replay and data as scripts/seed45_replay.py (prompts 2-6 counted).
  LRU+P b   victim key = last-use stamp (checks) + b * popularity (leave-one-prompt-out share of checks)
  LFU d     decayed LFU with decay d per call; LFU d+P b: + b * w * popularity
Lines (written before the run): PASS = the best variant cuts misses per kept token >= 10 % vs LRU at K 266 (C 240) with no
prompt worse by > 5 %; KILL = < 5 %; BETWEEN otherwise.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python scripts/seed45_lru.py -> data/seed45/seed45_lru.txt"""
import heapq, os, sys, time
import numpy as np
sys.argv = sys.argv[:1]
CAP = os.path.expanduser("~/BigMoE/data/cap45")
NL, E = 48, 512
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
if os.path.exists(os.path.expanduser("~/.local/share/quiet/PAUSE")):
    sys.exit("PAUSE set: not running")
t0 = time.time()
P = []
for name in ORDER:
    z = np.load(os.path.join(CAP, name + "_gate_0.npz"))
    seq, L, rows, roff, ids = z["seq"], z["L"], z["rows"], z["row_off"], z["ids"]
    checks, order = {}, []
    for n, (s, l, r) in enumerate(zip(seq, L, rows)):
        s, l, r = int(s), int(l), int(r)
        c = checks.setdefault(s, {"need": {}, "kept": None})
        if s not in order:
            order.append(s)
        if l in c["need"]:
            c["kept"] = r
        else:
            c["need"][l] = frozenset(np.unique(ids[roff[n]:roff[n] + r]).tolist())
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    P.append((name, cks, [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]))
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

def replay(K, lru, d=0.99, b=0.0):
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st, w, i = {}, 1.0, 0
        for p, (_, cks, res0) in enumerate(P):
            pop = POP[p][l]
            for t, need in [(-1, res0[l])] + [(t, c["need"][l]) for t, c in enumerate(cks)]:
                w /= d; i += 1
                if t >= 0:
                    miss[p][t] += sum(1 for e in need if e not in st)
                for e in need:
                    st[e] = float(i) if lru else st.get(e, 0.0) + w
                if len(st) > K:
                    key = (lambda e: st[e] + b * pop[e]) if lru else (lambda e: st[e] + b * w * pop[e])
                    for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=key):
                        del st[e]
    return miss

def pooled(m):
    return sum(m[p].sum() for p in range(1, len(P))) / sum(sum(c["kept"] for c in P[p][1]) for p in range(1, len(P)))
ARMS = [("LRU", True, 0.99, 0)] + [(f"LRU+P{b}", True, 0.99, b) for b in (3, 10, 30)] + \
       [(f"LFU{d}", False, d, 0) for d in (0.8, 0.9, 0.95)] + [("LFU0.9+P3", False, 0.9, 3), ("LFU0.95+P5", False, 0.95, 5)]
res = {}
for K in (246, 266, 287):
    for nm, lru, d, b in ARMS:
        res[(K, nm)] = replay(K, lru, d, b)
    base = pooled(res[(K, "LRU")])
    say(f"K {K}: " + ", ".join(f"{nm} {pooled(res[(K, nm)]):.1f} ({100 * (pooled(res[(K, nm)]) / base - 1):+.1f} %)" for nm, *_ in ARMS))
best = min([nm for nm, *_ in ARMS if nm != "LRU"], key=lambda nm: pooled(res[(266, nm)]))
gain = 1 - pooled(res[(266, best)]) / pooled(res[(266, "LRU")])
worst = max(res[(266, best)][p].mean() / res[(266, "LRU")][p].mean() - 1 for p in range(1, len(P)))
call = "PASS" if gain >= 0.10 and worst <= 0.05 else ("KILL" if gain < 0.05 else "BETWEEN")
say(f"Lines: best {best} cuts misses per kept token {100 * gain:.1f} % vs LRU at K 266 (worst prompt {100 * worst:+.1f} %) -> {call}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/BigMoE/data/seed45/seed45_lru.txt"), "w").write("\n".join(lines) + "\n")
