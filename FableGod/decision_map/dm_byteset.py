"""The exact byte set, replayed (Breakthrough, Mon 2026-11-02 21:1x; cloud Fable's 12 GB v5 via the orchestrator; CPU,
exact: bytes only). Pre-registered in ~/Breakthrough/SCOREBOARD.md (9098370). Today's policy (per-layer LRU admitting every
read, dm_rankcache.py's replay: DAY9's routing capture, prefill row by row at top-10, cache carried across a domain's
windows, 2.2 MB per expert record, rows after a check's first miss holding wrong drafts = row p + 97's routing), with:
(a) C 64 slots instead of 51;
(b) block trimming: the check drafts up to and including its first draft with cf < 0.5 ("cf0.5"; "run0.3" = the
    running-confidence cut at 0.3, the live engine's default), and pass 1 runs only those rows + 1;
(c) early-layer prefetch: after a check that ran pass 2, the next check's pass-1 misses at layers 0-2 leave the
    critical path (an ORACLE bound: during pass 2 the next check's tokens are not known yet).
Re-walk (needed once checks end early): from an anchor t inside exact-walk check c, the first miss stays c's (A + k) and
the drafts' cf are c's at the same target positions (unknown beyond c's drafts = 0, so trimming ends there).
Per check: pass-1 GB on the critical path, prefetched GB, pass-2 GB, kept per check, pass-1 GB per kept token.
Usage (BigMoE venv): python dm_byteset.py -> out/BYTESET_read.txt"""
import json, os
from collections import OrderedDict
import numpy as np
import dm_gate, dm_acert
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P2 = json.load(open(os.path.join(OUT, "topk2.json")))
MB, EARLY = 2.2, 3
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def replay(dom, C, cert, trim, prefetch):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    cache = [OrderedDict() for _ in range(48)]; T = dict(chk=0, kept=0, m1=0, mpf=0, m2=0, p2=0)
    def use(l, e):
        c = cache[l]
        if e in c: c.move_to_end(e); return True
        c[e] = 1
        if len(c) > C: c.popitem(last=False)
        return False
    for i, nm in enumerate(names):
        if nm not in base or base[nm]["dom"] != dom: continue
        ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; N = len(ids)
        p1 = np.exp(np.array(P2[nm]["lp"])[:, 0]); a1 = np.array(P2[nm]["tk"])[:, 0]
        cm = p1 >= 0.8 if cert == "0.8" else dm_acert.acert(nm, a, p1, a1, 0.5, 0.95, 0.99)
        ok = np.r_[np.full(a, True), cm, np.full(N + 1 - a - len(cm), False)]
        chk = Z[f"{i}_checks"].tolist(); dcf = Z[f"{i}_dcf"].tolist()
        cidx = np.full(N + 32, -1)
        for c, (k, m, A) in enumerate(chk): cidx[A:A + k + 1] = c
        for p in range(a):
            for l in range(48):
                for e in ids[p][l]: use(l, e)
        t, end, prev_p2 = chk[0][2], min(chk[-1][2] + chk[-1][0], N - 1), False
        while t <= end:
            c = int(cidx[t]); k, m, A = chk[c]; o = t - A; acc = min(A + k - t, 15); d = min(15, N - 1 - t)
            if trim:
                run = 1.0
                for j in range(1, d + 1):
                    cf = dcf[c][o + j - 1] if o + j - 1 < m else 0.0; run *= cf
                    if (trim == "cf0.5" and cf < 0.5) or (trim == "run0.3" and run < 0.3): d = j; break
            kept = acc + 1 if acc < d else d + 1; fm = t + min(acc, d)
            rows = [p if p <= fm else (p + 97) % N for p in range(t, t + d + 1)]
            unc = [p for p in range(t, fm + 1) if not ok[p]]
            T["chk"] += 1; T["kept"] += kept; T["p2"] += bool(unc)
            for l in range(48):
                s1 = set()
                for p in rows: s1.update(ids[p][l][:2])
                for e in s1:
                    if not use(l, e):
                        if prefetch and prev_p2 and l < EARLY: T["mpf"] += 1
                        else: T["m1"] += 1
                s2 = set()
                for p in unc: s2.update(ids[p][l][2:])
                for e in s2 - s1:
                    if not use(l, e): T["m2"] += 1
            prev_p2 = bool(unc); t += kept
    c = T["chk"]
    return dict(b1=T["m1"] * MB / 1e3 / c, bpf=T["mpf"] * MB / 1e3 / c, b2=T["m2"] * MB / 1e3 / c, kept=T["kept"] / c,
                p2=100 * T["p2"] / c, b1k=T["m1"] * MB / T["kept"])

CONF = [("today (C 51)", 51, None, False), ("(a) C 64", 64, None, False), ("(b) trim cf < 0.5", 51, "cf0.5", False),
        ("(b') trim run < 0.3 (live default)", 51, "run0.3", False), ("(c) prefetch layers 0-2 (oracle)", 51, None, True),
        ("SET: C 64 + trim cf0.5 + prefetch", 64, "cf0.5", True), ("set with run0.3", 64, "run0.3", True)]
say("The exact byte set at 12 GB (LRU; per check; raw replay, FLOOR's 1.3x calibrated pass-1 in brackets)")
say("| domain | certificate | config | pass-1 GB (critical) | prefetched GB | pass-2 GB | kept per check | pass-1 MB per kept token | blocks with pass 2 |")
say("|---|---|---|---|---|---|---|---|---|")
R = {}
for dom in ("code", "ifeval", "multi"):
    for cert in ("0.8", "adapt"):
        for lab, C, trim, pf in CONF:
            r = replay(dom, C, cert, trim, pf); R[(dom, cert, lab)] = r
            say(f"| {dom} | {cert} | {lab} | {r['b1']:.3f} ({r['b1'] / 1.3:.3f}) | {r['bpf']:.3f} | {r['b2']:.3f} | {r['kept']:.2f} | "
                f"{r['b1k']:.1f} | {r['p2']:.0f} % |")
say("\n## Verdict (pre-registered, SCOREBOARD 9098370): code, the SET, raw replay")
for cert in ("0.8", "adapt"):
    b, s = R[("code", cert, "today (C 51)")], R[("code", cert, "SET: C 64 + trim cf0.5 + prefetch")]
    drop = 100 * (1 - s["kept"] / b["kept"]); v1 = s["b1"] if drop <= 5 else s["b1k"] * b["kept"] / 1e3
    v = "PASS" if v1 <= 0.50 else "KILL" if v1 > 0.60 else "BETWEEN"
    say(f"- {cert}: pass-1 {b['b1']:.3f} -> {s['b1']:.3f} GB per check (kept {b['kept']:.2f} -> {s['kept']:.2f}, {drop:+.1f} % down; "
        f"judged value {v1:.3f}; calibrated {v1 / 1.3:.3f}) -> {v} (PASS <= 0.50, KILL > 0.60)")
open(os.path.join(OUT, "BYTESET_read.txt"), "w").write("\n".join(lines) + "\n")
