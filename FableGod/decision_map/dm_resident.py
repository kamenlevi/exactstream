"""Resident-only pass 1? One replay count (Breakthrough, Mon 2026-11-02 19:3x; cloud Fable's 12 GB v2 pushback: "a
resident-only pass 1 is a dead end ... one replay count settles it"). CPU. The LRU replay of cache_replay.py (FableGod,
mission 20: per-layer LRU at C slots, the prompt's top-10 loads, then each check's union under the two-pass top-2 scheme
at theta 0.99, rows after the first exact miss at top-2; the cache carries over between a domain's windows), and at the
START of each check (before its loads) the count: for every row of the block and every layer, are BOTH of the row's
top-2 experts resident? A resident-only pass 1 needs that at all 48 layers for a row's pass to be the true top-2 pass.
LOADS = "kept" (default): a check loads its rows A .. first miss only. The replay's rows after the miss carry the TRUE
tokens' routing, which the live engine never sees there (those rows hold wrong drafts), so loading them would pre-load
the next check's rows and inflate the count; "all" = cache_replay.py's loads, for comparison.
Usage (BigMoE venv): python dm_resident.py -> out/RESIDENT_read.txt"""
import json, os
from collections import OrderedDict
import numpy as np
import dm_gate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P2 = json.load(open(os.path.join(OUT, "topk2.json")))
lines = []
def say(s=""): print(s, flush=True); lines.append(s)

def touch(c, e, C):
    if e in c: c.move_to_end(e); return
    c[e] = 1
    if len(c) > C: c.popitem(last=False)

LOADS = os.environ.get("LOADS", "kept")
say(f"Both top-2 experts resident at a check's start (LRU per layer, two-pass top-2 theta 0.99 replay; loads: {LOADS})")
say("| domain | C | (row, layer) pairs both resident | layers per row with both resident (mean) | rows with all 48 layers | rows with >= 40 layers | certified rows: all 48 layers |")
say("|---|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    for C in (51, 160):
        cache = [OrderedDict() for _ in range(48)]; pairs = both = 0; per_row = []; cert_rows = cert_all = 0
        for i, nm in enumerate(names):
            if nm not in base or base[nm]["dom"] != dom: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
            cert = np.exp(np.array(P2[nm]["lp"])[:, 0]).tolist()
            for p in range(a):
                for l in range(48):
                    for e in ids[p][l]: touch(cache[l], e, C)
            for k, m, A in Z[f"{i}_checks"].tolist():
                last, fm = min(A + m, n1 - 1), A + k
                for p in range(A, last + 1):
                    r = sum(1 for l in range(48) if ids[p][l][0] in cache[l] and ids[p][l][1] in cache[l])
                    pairs += 48; both += r; per_row.append(r)
                    j = p - a
                    if 0 <= j < len(cert) and cert[j] >= 0.99: cert_rows += 1; cert_all += r == 48
                for l in range(48):
                    st = set()
                    for p in range(A, (min(fm, last) if LOADS == "kept" else last) + 1):
                        j = p - a
                        kk = 10 if (j < 0 or j >= len(cert) or (cert[j] < 0.99 and p <= fm)) else 2
                        st.update(ids[p][l][:kk])
                    for e in st: touch(cache[l], e, C)
        pr = np.array(per_row)
        say(f"| {dom} | {C} | {100 * both / pairs:.1f} % | {pr.mean():.1f} | {100 * (pr == 48).mean():.1f} % | {100 * (pr >= 40).mean():.1f} % | "
            f"{100 * cert_all / max(cert_rows, 1):.1f} % ({cert_all}/{cert_rows}) |")
open(os.path.join(OUT, f"RESIDENT_read_{LOADS}.txt"), "w").write("\n".join(lines) + "\n")
