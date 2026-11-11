"""The ceiling for pass 1's reads at C slots (Breakthrough, Mon 2026-11-02 20:5x; CPU): Belady's MIN (evict the resident
expert used farthest in the future) on the rank-aware cache's stream, i.e. prefill rows' ranks 0-1 and each check's
union of its block rows' top-2 (rows after the first miss: row p + 97's routing, or their own). No realizable residency
policy reads fewer pass-1 bytes. Same method as dm_rankcache.py. Usage (BigMoE venv): python dm_belady1.py"""
import json, os, bisect
import numpy as np
import dm_gate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json")))
for dom in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    for after in ("proxy", "true"):
        steps = [[] for _ in range(48)]; is_chk = []          # per layer: the access set per step
        for i, nm in enumerate(names):
            if nm not in base or base[nm]["dom"] != dom: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a; N = len(ids)
            for p in range(a):
                is_chk.append(False)
                for l in range(48): steps[l].append(set(ids[p][l][:2]))
            for k, m, A in Z[f"{i}_checks"].tolist():
                last, fm = min(A + m, n1 - 1), A + k; is_chk.append(True)
                rows = [p if (after == "true" or p <= fm) else (p + 97) % N for p in range(A, last + 1)]
                for l in range(48):
                    s = set()
                    for p in rows: s.update(ids[p][l][:2])
                    steps[l].append(s)
        nchk = sum(is_chk)
        for C in (51, 160):
            miss = acc = 0
            for l in range(48):
                occ = {}
                for t, s in enumerate(steps[l]):
                    for e in s: occ.setdefault(e, []).append(t)
                def nxt(e, t):
                    o = occ[e]; j = bisect.bisect_left(o, t); return o[j] if j < len(o) else 10 ** 9
                cache = set()
                for t, s in enumerate(steps[l]):
                    for e in s:
                        if is_chk[t]: acc += 1
                        if e in cache: continue
                        if is_chk[t]: miss += 1
                        if len(cache) >= C: cache.remove(max(cache, key=lambda r: nxt(r, t)))
                        cache.add(e)
            print(f"| {dom} | {C} | {after} | Belady MIN | {100 * (1 - miss / acc):.1f} % | {miss * 2.2 / 1e3 / nchk:.3f} |", flush=True)
