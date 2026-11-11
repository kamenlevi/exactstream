"""P1-REACH screen (NewIdeas, Tue 3 Nov 2026; CPU only, writes nothing). On top of P1-K1 (the 12 GB two-pass check's pass 1 =
each row's top-1 + its ranks already in the block's union), a row whose chance of being reached (the product of the drafts'
cf before it, from the HPD-0 walk) is < tau adds NOTHING to pass 1's union: it computes with the union members only. If
such a row is reached and fails the certificate, pass 2 recomputes it exactly, so quality stays guarded by the certificate;
the cost is more pass-2 blocks. Fit12's T3 runs pass 1 on all 16 rows with no running-confidence cut.
Replay = pass1_bytes.py's (Breakthrough's dm_rankcache: LRU per layer, C 51, wrong-draft stand-ins p + 97, cache carried
across a domain's windows, 2.2 MB per record, 13 GB/s). Certificate: p1_oea_k1's own p1 >= 0.8 (the map run), with the
low-reach realized rows forced UNCERTIFIED (pessimistic: they keep fewer experts than in that run). Pass 2 = all ranks of
the uncertified realized rows not already loaded. Time model: pass-1 reads at 13 GB/s; a block needing pass 2 costs ~120 ms
(Fit12's T3: <= 4-row exact calls). Usage (BigMoE venv): python p1_reach_screen.py"""
import json, os, sys
from collections import OrderedDict
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P1 = json.load(open(os.path.join(OUT, "p1_oea_k1.json")))
MB, SSD, P2MS, THETA = 2.2, 13.0, 120.0, 0.8


def replay(dom, C, tau):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    cache = [OrderedDict() for _ in range(48)]; T = dict(chk=0, miss1=0, miss2=0, p2=0, low_real=0, real=0)
    def use(l, e):
        c = cache[l]
        if e in c: c.move_to_end(e); return True
        c[e] = 1
        if len(c) > C: c.popitem(last=False)
        return False
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
        p1 = np.exp(np.array(P1[nm]["lp"])[:, 0]); N = len(ids)
        ok = np.r_[np.full(a, True), p1 >= THETA, np.full(N - a - len(p1) + 1, False)]
        for p in range(a):
            for l in range(48):
                for e in ids[p][l]: use(l, e)
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            last, fm = min(A + m, n1 - 1), A + k; T["chk"] += 1
            reach = np.r_[1.0, np.cumprod(np.array(dc, float))]
            rows = [(p if p <= fm else (p + 97) % N, reach[p - A] >= tau) for p in range(A, last + 1)]
            unc = []
            for p in range(A, min(fm, last) + 1):
                T["real"] += 1; low = reach[p - A] < tau; T["low_real"] += low
                if low or not ok[p]: unc.append(p)
            T["p2"] += bool(unc)
            for l in range(48):
                s1 = {ids[r][l][0] for r, hi in rows if hi}
                for e in s1:
                    if not use(l, e): T["miss1"] += 1
                s2 = set()
                for p in unc: s2.update(ids[p][l])
                for e in s2 - s1:
                    if not use(l, e): T["miss2"] += 1
    c = T["chk"]
    return dict(b1=T["miss1"] * MB / 1e3 / c, b2=T["miss2"] * MB / 1e3 / c, p2=100 * T["p2"] / c, low=100 * T["low_real"] / T["real"])


print("| domain | tau | pass-1 GB per check | pass-1 read ms | pass-2 GB per check | blocks needing pass 2 | low-reach realized rows | model ms per check vs P1-K1 |")
print("|---|---|---|---|---|---|---|---|")
for dom in ("code",):
    ref = None
    for tau in (0.0, 0.2, 0.4, 0.6):
        r = replay(dom, 51, tau)
        t = 1e3 * r["b1"] / SSD + P2MS * r["p2"] / 100
        if ref is None: ref = t
        print(f"| {dom} | {tau} | {r['b1']:.3f} | {1e3 * r['b1'] / SSD:.1f} | {r['b2']:.3f} | {r['p2']:.1f} % | {r['low']:.1f} % | "
              f"{t - ref:+.1f} |", flush=True)
