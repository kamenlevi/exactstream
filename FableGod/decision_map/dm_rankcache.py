"""The rank-aware cache, replayed (Breakthrough, Mon 2026-11-02 20:4x; cloud Fable's 12 GB v4 via the orchestrator; CPU,
exact: bytes and timing only). Pass 1 of the two-pass check only reads ranks 0-1, so the cache admits only experts used at
ranks 0-1 and streams ranks 2-9 (pass 2's reads, which hide under its compute); against today's policy, the same per-layer
LRU admitting every read. Pre-registered in ~/Breakthrough/SCOREBOARD.md (7b42f30).
Replay (cache_replay.py's method, FableGod mission 20): DAY9's routing capture, the HPD-0 walks' checks, per-layer caches
of C slots carried across a domain's windows, 2.2 MB per expert record, prefill row by row (rank-aware: its ranks 0-1
admitted, the rest streamed). Per check and layer: pass 1 = the union of the block rows' top-2 (rows A .. A + m); then,
when the block needs pass 2, ranks 2-9 of the uncertified rows up to the first exact miss (top-2 certificate p1 >= theta
from the topk2 run; prompt rows are prefill's). Rows after the first miss: "proxy" = row p + 97's routing (a wrong
draft's stand-in, as the engine sees them; the pre-registered variant), "true" = their own (cache_replay.py's way).
Usage (BigMoE venv): python dm_rankcache.py -> out/RANKCACHE_read.txt"""
import json, os
from collections import OrderedDict
import numpy as np
import dm_gate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P2 = json.load(open(os.path.join(OUT, "topk2.json")))
MB, SSD = 2.2, 13.0                                      # MB per expert record; GB/s
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def replay(dom, C, theta, policy, after):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    cache = [OrderedDict() for _ in range(48)]; T = dict(chk=0, kept=0, acc1=0, hit1=0, miss1=0, acc2=0, miss2=0, p2=0)
    def use(l, e, admit):
        c = cache[l]
        if e in c:
            if admit: c.move_to_end(e)                   # a rank 2-9 use does not refresh a rank-aware cache
            return True
        if admit:
            c[e] = 1
            if len(c) > C: c.popitem(last=False)
        return False
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
        p1 = np.exp(np.array(P2[nm]["lp"])[:, 0]); N = len(ids)
        ok = np.r_[np.full(a, True), p1 >= theta, np.full(N - a - len(p1) + 1, False)]   # trusted rows (prefill, certified)
        for p in range(a):
            for l in range(48):
                for r, e in enumerate(ids[p][l]): use(l, e, policy == "lru" or r < 2)
        for k, m, A in Z[f"{i}_checks"].tolist():
            last, fm = min(A + m, n1 - 1), A + k; T["chk"] += 1; T["kept"] += k + 1
            rows = [p if (after == "true" or p <= fm) else (p + 97) % N for p in range(A, last + 1)]
            unc = [p for p in range(A, min(fm, last) + 1) if not ok[p]]
            T["p2"] += bool(unc)
            for l in range(48):
                s1 = set()
                for p in rows: s1.update(ids[p][l][:2])
                for e in s1:
                    T["acc1"] += 1
                    if use(l, e, True): T["hit1"] += 1
                    else: T["miss1"] += 1
                s2 = set()
                for p in unc: s2.update(ids[p][l][2:])
                for e in s2 - s1:
                    T["acc2"] += 1
                    if not use(l, e, policy == "lru"): T["miss2"] += 1
    c = T["chk"]
    return dict(hit1=100 * T["hit1"] / T["acc1"], b1=T["miss1"] * MB / 1e3 / c, b2=T["miss2"] * MB / 1e3 / c,
                kept=T["kept"] / c, p2=100 * T["p2"] / c)

say("Rank-aware cache vs today's LRU (two-pass top-2; per check; GB = 2.2 MB x misses; pass-1 read time at 13 GB/s)")
say("| domain | C | theta | after-miss rows | policy | pass-1 hit rate | pass-1 GB | pass-2 GB | total GB | pass-1 read ms | blocks with pass 2 | kept |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
R = {}
for dom in ("code", "ifeval", "multi"):
    for C, theta in ((51, 0.8), (160, 0.8), (51, 0.99)):
        for after in ("proxy", "true"):
            for pol in ("lru", "rank"):
                r = replay(dom, C, theta, pol, after); R[(dom, C, theta, after, pol)] = r
                say(f"| {dom} | {C} | {theta} | {after} | {pol} | {r['hit1']:.1f} % | {r['b1']:.3f} | {r['b2']:.3f} | {r['b1'] + r['b2']:.3f} | "
                    f"{1e3 * r['b1'] / SSD:.0f} | {r['p2']:.0f} % | {r['kept']:.1f} |")
say("\n## Verdict (pre-registered, SCOREBOARD 7b42f30): code, C 51, theta 0.8, wrong-draft rows after the miss")
o, n = R[("code", 51, 0.8, "proxy", "lru")], R[("code", 51, 0.8, "proxy", "rank")]
up = 100 * ((n["b1"] + n["b2"]) / (o["b1"] + o["b2"]) - 1)
v = "KILL" if n["b1"] > 0.40 else "PASS" if n["b1"] <= 0.30 and up <= 10 else "BETWEEN"
say(f"- pass-1 bytes {o['b1']:.3f} -> {n['b1']:.3f} GB per check (hit rate {o['hit1']:.1f} -> {n['hit1']:.1f} %), total "
    f"{o['b1'] + o['b2']:.3f} -> {n['b1'] + n['b2']:.3f} GB ({up:+.1f} %) -> {v} (PASS <= 0.30 with total not up > 10 %, KILL > 0.40)")
open(os.path.join(OUT, "RANKCACHE_read.txt"), "w").write("\n".join(lines) + "\n")
