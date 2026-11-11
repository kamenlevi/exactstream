"""SEED45 (BigMoE, Sun 2026-11-01; MoonShots M-13's pointer "seed the cache by global popularity"): does a popularity
prior on the expert cache cut SSD misses per check at the 45 GB default's budgets? Residency only, so exact if it pays.
Budgets: C 224 / 240 / 256 (4-bit slots' bytes, C_l 112) = 246 / 266 / 287 resident MSB-3 slots per layer (the K below);
MoonShots' K 224 / 288 / 384 for reference.
Data (read only): data/cap45 (CAP45, the exact REF2 loop on the gate at C 160, real DFlash2 16-row checks, S45's six prompts
in one process). Per layer a check needs the union of its 16 rows' experts; the commit replay is all hits (ignored).
The prompts are CHAINED as one process serves them: the cache state carries over, and each prompt's prefill is its
captured res0 forced in as one uncounted call (its loads are prefill cost). Misses counted on prompts 2-6 (prompt 1,
HumanEval/68 with 5 captured checks, is the warm-up).
Policies per layer (victims = lowest key outside the call's needs, BigMoE's ext rule):
  LFU      decayed LFU d 0.99 (each call scales the increment by 1/0.99; BIGMOE_EVICT_LFU=0.99, the default)
  PRIOR b  key = decayed count + b * popularity (popularity = share of the OTHER five prompts' checks that needed the
           expert at that layer, leave-one-prompt-out; a decayed count saturates at 100 for an expert in every check)
  PIN p    the top-p popular experts per layer leave last; LFU for the rest
  MIN      Belady's optimum (future known): the headroom bound for any policy
  LRU      reference row (added after the first run: the live 45 GB loop never set BIGMOE_EVICT_LFU, so it runs LRU)
Lines (written before the run): PASS = the best PRIOR / PIN cuts misses per kept token >= 10 % vs LFU at K 266 (C 240),
pooled over prompts 2-6, with no prompt worse by > 5 %; KILL = < 5 %; BETWEEN otherwise.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python scripts/seed45_replay.py (CPU, one core) -> data/seed45/seed45_replay.txt"""
import bisect, heapq, os, sys, time
import numpy as np

CAP = os.path.expanduser("~/BigMoE/data/cap45")
OUT = os.path.expanduser("~/BigMoE/data/seed45")
NL, E = 48, 512
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")                  # the capture run's order (cap45_run.txt)
MSB_MB = 2.16
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
            c["kept"] = r                                            # the commit replay: kept rows
        else:
            c["need"][l] = frozenset(np.unique(ids[roff[n]:roff[n] + r]).tolist())
    cks = [checks[s] for s in order if len(checks[s]["need"]) == NL and checks[s]["kept"] is not None]
    res0 = [frozenset(np.nonzero(z["res0"][l] >= 0)[0].tolist()) for l in range(NL)]
    P.append((name, cks, res0))
POP = []                                                             # leave-one-prompt-out popularity [48, 512]
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, list(c["need"][l])] += 1
            n += len(cks)
    POP.append(pop / n)
say(f"SEED45: cap45, {sum(len(c) for _, c, _ in P)} live checks over {len(P)} prompts, chained; res0 "
    f"{np.mean([len(x) for _, _, r in P for x in r]):.0f} experts per layer")


def replay(K, pol, arg=0.0):
    """misses per prompt (list of per-check sums over layers)"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        items = []                                                   # (prompt, check index or -1 for res0, need)
        for p, (_, cks, res0) in enumerate(P):
            items.append((p, -1, res0[l]))
            items += [(p, t, c["need"][l]) for t, c in enumerate(cks)]
        if pol == "min":
            uses = {}
            for i, (_, _, need) in enumerate(items):
                for e in need:
                    uses.setdefault(e, []).append(i)
        st, w = {}, 1.0
        for i, (p, t, need) in enumerate(items):
            w /= 0.99
            if t >= 0:
                miss[p][t] += sum(1 for e in need if e not in st)
            for e in need:
                st[e] = i + 1.0 if pol == "lru" else st.get(e, 0.0) + w
            if len(st) <= K:
                continue
            pop = POP[p][l]
            if pol in ("lfu", "lru"):                                 # LRU: the last-use stamp
                key = st.__getitem__
            elif pol == "prior":
                key = lambda e: st[e] + arg * w * pop[e]
            elif pol == "pin":
                top = set(np.argsort(-pop)[:int(arg)].tolist())
                key = lambda e: st[e] + (1e30 if e in top else 0.0)
            else:                                                    # Belady: the farthest next use leaves first
                def key(e):
                    u = uses[e]; k = bisect.bisect_right(u, i)
                    return -(u[k] if k < len(u) else 1e9)
            for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=key):
                del st[e]
    return miss


def pooled(miss):
    m = sum(miss[p].sum() for p in range(1, len(P)))
    k = sum(sum(c["kept"] for c in P[p][1]) for p in range(1, len(P)))
    return m / k


KS = (246, 266, 287, 224, 288, 384)
POLS = [("lru", 0), ("lfu", 0)] + [("prior", b) for b in (3, 10, 30, 100)] + [("pin", q) for q in (64, 128, 192)] + [("min", 0)]
res = {}
for K in KS:
    for pol, a in POLS:
        if pol == "pin" and a >= K - 64:
            continue
        res[(K, pol, a)] = replay(K, pol, a)
say("misses per kept token, prompts 2-6 pooled (MB per kept token at 2.16 MB; vs LFU):")
say("  K    C~   " + " ".join(f"{(pol + str(a) if a else pol):>9s}" for pol, a in POLS))
for K in KS:
    base = pooled(res[(K, "lfu", 0)])
    c = {246: "224", 266: "240", 287: "256"}.get(K, "  -")
    say(f"  {K:3d}  {c}   " + " ".join(
        (f"{pooled(res[(K, pol, a)]):9.1f}" if (K, pol, a) in res else f"{'-':>9s}") for pol, a in POLS))
    say(f"            " + " ".join(
        (f"{100 * (pooled(res[(K, pol, a)]) / base - 1):+8.1f}%" if (K, pol, a) in res else f"{'':>9s}") for pol, a in POLS))
say(f"  LFU MB per kept token: " + ", ".join(f"K {K} {pooled(res[(K, 'lfu', 0)]) * MSB_MB:.0f}" for K in KS))
say("per prompt at K 266 (C 240), mean misses per check: LFU / best PRIOR / best PIN / MIN")
cand = [(pol, a) for pol, a in POLS if pol in ("prior", "pin") and (266, pol, a) in res]
best = min(cand, key=lambda x: pooled(res[(266,) + x]))
bp = min([x for x in cand if x[0] == "prior"], key=lambda x: pooled(res[(266,) + x]))
bq = min([x for x in cand if x[0] == "pin"], key=lambda x: pooled(res[(266,) + x]))
worst = 0.0
for p in range(1, len(P)):
    lf, pr, pn, mn = (res[(266,) + x][p].mean() for x in (("lfu", 0), bp, bq, ("min", 0)))
    b = res[(266,) + best][p].mean()
    worst = max(worst, b / lf - 1)
    say(f"  {P[p][0][:30]:30s} {lf:6.0f} {pr:6.0f} ({bp[0]}{bp[1]}) {pn:6.0f} ({bq[0]}{bq[1]}) {mn:6.0f}  kept {np.mean([c['kept'] for c in P[p][1]]):.2f}")
gain = 1 - pooled(res[(266,) + best]) / pooled(res[(266, "lfu", 0)])
head = 1 - pooled(res[(266, "min", 0)]) / pooled(res[(266, "lfu", 0)])
call = "PASS" if gain >= 0.10 and worst <= 0.05 else ("KILL" if gain < 0.05 else "BETWEEN")
say(f"Lines: best {best[0]} {best[1]} cuts misses per kept token {100 * gain:.1f} % vs LFU at K 266 (worst prompt "
    f"{100 * worst:+.1f} %; Belady headroom {100 * head:.1f} %) -> {call}")
say(f"({time.time() - t0:.0f} s)")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "seed45_replay.txt"), "w").write("\n".join(lines) + "\n")
