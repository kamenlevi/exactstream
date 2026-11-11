"""Pass-1 bytes of the 12 GB two-pass check when pass 1 reads only each row's top-k1 (NewIdeas, Tue 3 Nov 2026).
A copy of Breakthrough's dm_rankcache.py replay (read-only use of ~/FableGod/decision_map; LRU policy, wrong-draft
stand-ins after the miss, C 51, cache carried across a domain's windows, 2.2 MB per record, 13 GB/s), with pass 1 = the
union of the block rows' top-k1 (k1 = 2 is today's design; k1 = 1 is the OEA-k1 pass 1, whose union members add no
reads). Pass 2 here keeps topk2's certificate at theta 0.8 for every k1 (the OEA-k1 certificate needs its own map run),
so only the pass-1 column is the k1 = 1 design's own. Writes nothing. Usage (BigMoE venv): python pass1_bytes.py"""
import json, os, sys
from collections import OrderedDict
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P2 = json.load(open(os.path.join(OUT, "topk2.json")))
MB, SSD = 2.2, 13.0


def replay(dom, C, theta, k1):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    cache = [OrderedDict() for _ in range(48)]; T = dict(chk=0, kept=0, acc1=0, hit1=0, miss1=0, miss2=0, p2=0, u1=0)
    def use(l, e):
        c = cache[l]
        if e in c: c.move_to_end(e); return True
        c[e] = 1
        if len(c) > C: c.popitem(last=False)
        return False
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
        p1 = np.exp(np.array(P2[nm]["lp"])[:, 0]); N = len(ids)
        ok = np.r_[np.full(a, True), p1 >= theta, np.full(N - a - len(p1) + 1, False)]
        for p in range(a):
            for l in range(48):
                for e in ids[p][l]: use(l, e)
        for k, m, A in Z[f"{i}_checks"].tolist():
            last, fm = min(A + m, n1 - 1), A + k; T["chk"] += 1; T["kept"] += k + 1
            rows = [p if p <= fm else (p + 97) % N for p in range(A, last + 1)]
            unc = [p for p in range(A, min(fm, last) + 1) if not ok[p]]
            T["p2"] += bool(unc)
            for l in range(48):
                s1 = set()
                for p in rows: s1.update(ids[p][l][:k1])
                T["u1"] += len(s1)
                for e in s1:
                    T["acc1"] += 1
                    if use(l, e): T["hit1"] += 1
                    else: T["miss1"] += 1
                s2 = set()
                for p in unc: s2.update(ids[p][l][k1:])
                for e in s2 - s1:
                    if not use(l, e): T["miss2"] += 1
    c = T["chk"]
    return dict(hit1=100 * T["hit1"] / T["acc1"], b1=T["miss1"] * MB / 1e3 / c, b2=T["miss2"] * MB / 1e3 / c,
                u1=T["u1"] / c / 48, kept=T["kept"] / c, p2=100 * T["p2"] / c)


print("| domain | k1 | pass-1 union per layer | pass-1 hit rate | pass-1 GB per check | pass-1 read ms (13 GB/s) | pass-2 GB (topk2 cert 0.8) | kept |")
print("|---|---|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi"):
    for k1 in (2, 1):
        r = replay(dom, 51, 0.8, k1)
        print(f"| {dom} | {k1} | {r['u1']:.1f} | {r['hit1']:.1f} % | {r['b1']:.3f} | {1e3 * r['b1'] / SSD:.0f} | {r['b2']:.3f} | {r['kept']:.1f} |",
              flush=True)
