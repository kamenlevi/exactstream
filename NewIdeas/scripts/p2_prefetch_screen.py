"""P2-PREFETCH screen (NewIdeas, Tue 3 Nov 2026; CPU only, writes nothing). Fit12's T3P trace: pass 1 of the 12 GB two-pass
check is 57 % SSD wait + 43 % GPU gap (~1.3 ms per layer at 16 rows), and pass 2's <= 4-row exact calls are bandwidth-bound
(78 ms of wait per pass at ~10.6 GB/s). Idea: during pass 1's GPU gap at layer l the SSD idles, so read layer l's pass-2
experts ahead, for rows predicted to fail the certificate (drafter cf < c; known before the check), up to B experts per
layer (B ~ 1.3 ms x 11 GB/s / 2.2 MB per expert ~ 6). Pass 2 then finds them resident. Adapted from draft-phase
prefetch of the target's experts (MoE-SpeQ / SP-MoE); here the window is pass 1's GPU gap and the target is pass 2.
Replay = pass1_bytes.py's (Breakthrough's dm_rankcache: LRU per layer at C 51, wrong-draft stand-ins p + 97, cache carried
across a domain's windows, prompt rows warm the cache). Pass 1 = P1-K1 (each row's rank 0; members add no reads).
Certificate = p1_oea_k1's p1 >= 0.8. Pass 2 = all ranks of the uncertified realized rows not already resident.
Prefetch order: rows by position, then rank 1..9. A prefetched expert enters the LRU like a use, so evictions hit later.
Reported per check: pass-1 misses, pass-2 misses, prefetch reads (and how many pass 2 used), and the model ms saved
(pass-2 misses avoided minus extra pass-1 misses, at 2.2 MB per expert and 11 GB/s).
Usage (BigMoE venv): python p2_prefetch_screen.py"""
import json, os, sys
from collections import OrderedDict
import numpy as np
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(0, DM)
import dm_gate  # noqa: E402
OUT = os.path.join(DM, "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P1 = json.load(open(os.path.join(OUT, "p1_oea_k1.json")))
G = dm_gate.load()
MB, GBS, THETA, C = 2.2, 11.0, 0.8, 51


def replay(dom, pred, B):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    cache = [OrderedDict() for _ in range(48)]
    T = dict(chk=0, m1=0, m2=0, pf=0, pf_used=0, p2blk=0)
    def touch(l, e):
        c = cache[l]
        if e in c: c.move_to_end(e); return True
        c[e] = 1
        if len(c) > C: c.popitem(last=False)
        return False
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
        p1 = np.exp(np.array(P1[nm]["lp"])[:, 0]); N = len(ids); g = np.nan_to_num(G[nm], nan=-1.0)
        ok = np.r_[np.full(a, True), p1 >= THETA, np.full(N - a - len(p1) + 1, False)]
        for p in range(a):
            for l in range(48):
                for e in ids[p][l]: touch(l, e)
        for (k, m, A), dc in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
            last, fm = min(A + m, n1 - 1), A + k; T["chk"] += 1
            rows = [p if p <= fm else (p + 97) % N for p in range(A, last + 1)]
            cfs = [dc[j] if j < m else -1.0 for j in range(len(rows))]
            unc = [p for p in range(A, min(fm, last) + 1) if not ok[p]]
            T["p2blk"] += bool(unc)
            if pred == "oracle": hot = [p for p in unc]
            elif pred == "none": hot = []
            elif isinstance(pred, tuple):                   # (cf limit, reach floor): unsure rows likely to be reached
                rch = np.r_[1.0, np.cumprod(np.array(dc, float))][:len(rows)]
                hot = [r for r, c, q in zip(rows, cfs, rch) if c < pred[0] and q >= pred[1]]
            else: hot = [r for r, c in zip(rows, cfs) if c < pred]
            pf_sets = []
            for l in range(48):                                  # pass 1, layer by layer
                s1 = {ids[r][l][0] for r in rows}
                for e in s1:
                    if not touch(l, e): T["m1"] += 1
                pf = []
                for r in hot:                                    # prefetch in pass 1's GPU gap at this layer
                    for e in ids[r][l][1:]:
                        if len(pf) >= B: break
                        if e not in cache[l] and e not in pf: pf.append(e)
                    if len(pf) >= B: break
                for e in pf: touch(l, e)
                T["pf"] += len(pf); pf_sets.append(set(pf))
            for l in range(48):                                  # pass 2 (exact) on the uncertified realized rows
                s2 = set()
                for p in unc: s2.update(ids[p][l])
                for e in s2:
                    if e in pf_sets[l]: T["pf_used"] += 1
                    if not touch(l, e): T["m2"] += 1
    c = T["chk"]
    return {k: v / c for k, v in T.items() if k != "chk"}


print("B = experts per layer read in pass 1's GPU gap (1.3 ms x 11 GB/s / 2.2 MB ~ 6)")
print("| domain | predictor | B | pass-1 misses | pass-2 misses | prefetch reads (used by pass 2) | model ms per check vs none |")
print("|---|---|---|---|---|---|---|")
for dom in ("code",):
    r0 = replay(dom, "none", 0)
    print(f"| {dom} | none | 0 | {r0['m1']:.1f} | {r0['m2']:.1f} | 0 | +0.0 |  (blocks needing pass 2 {100 * r0['p2blk']:.1f} %)", flush=True)
    for pred, B in (("oracle", 6), (0.99, 6), (0.9, 6), ((0.99, 0.5), 6), ((0.99, 0.3), 6), ((0.99, 0.5), 3), ("oracle", 3)):
        r = replay(dom, pred, B)
        ms = ((r0["m2"] - r["m2"]) - (r["m1"] - r0["m1"])) * MB / GBS
        print(f"| {dom} | {pred} | {B} | {r['m1']:.1f} | {r['m2']:.1f} | {r['pf']:.1f} ({r['pf_used']:.1f}) | {-ms:+.1f} |", flush=True)
