"""Is cache_replay.py's SSD count optimistic? (Breakthrough, Mon 2026-11-02 19:3x; CPU.) Its rows after a check's first exact
miss load the TRUE tokens' routing (at top-2), which pre-loads the next check's rows; the live engine holds wrong drafts
there. Same LRU replay (two-pass top-2 theta 0.99, prompt top-10 loads, cache carried across a domain's windows); the
after-miss rows load (true) their own routing as cache_replay.py does, or (proxy) the routing of row p + 97 of the same
window (a wrong draft's stand-in: domain-like experts, not the next check's). Misses per check and per kept token.
Usage (BigMoE venv): python dm_replay_bias.py -> out/REPLAY_bias.txt"""
import json, os
from collections import OrderedDict
import numpy as np
import dm_gate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); P2 = json.load(open(os.path.join(OUT, "topk2.json")))
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
say("| domain | C | misses per check: true (cache_replay.py) | proxy (wrong drafts) | ratio | kept per check |")
say("|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi"):
    Z = np.load(os.path.join(dm_gate.TD, dm_gate.WALK[dom])); names = [str(x) for x in Z["names"]]
    for C in (51, 160, 256):
        res = {}
        for mode in ("true", "proxy"):
            cache = [OrderedDict() for _ in range(48)]; miss = chk = kept = 0
            for i, nm in enumerate(names):
                if nm not in base or base[nm]["dom"] != dom: continue
                ids = np.load(os.path.join(OUT, "routing", nm))["ids"].tolist(); a = base[nm]["a"]; n1 = len(base[nm]["tk"]) + a
                cert = np.exp(np.array(P2[nm]["lp"])[:, 0]).tolist(); N = len(ids)
                def touch(l, e, count):
                    global miss
                    c = cache[l]
                    if e in c: c.move_to_end(e); return
                    miss += count; c[e] = 1
                    if len(c) > C: c.popitem(last=False)
                for p in range(a):
                    for l in range(48):
                        for e in ids[p][l]: touch(l, e, 0)
                for k, m, A in Z[f"{i}_checks"].tolist():
                    last, fm = min(A + m, n1 - 1), A + k; chk += 1; kept += k + 1
                    for l in range(48):
                        st = set()
                        for p in range(A, last + 1):
                            j = p - a
                            kk = 10 if (j < 0 or j >= len(cert) or (cert[j] < 0.99 and p <= fm)) else 2
                            src = p if (mode == "true" or p <= fm) else (p + 97) % N
                            st.update(ids[src][l][:kk])
                        for e in st: touch(l, e, 1)
            res[mode] = (miss / chk, kept / chk)
        say(f"| {dom} | {C} | {res['true'][0]:.0f} | {res['proxy'][0]:.0f} | {res['proxy'][0] / res['true'][0]:.2f} | {res['true'][1]:.1f} |")
open(os.path.join(OUT, "REPLAY_bias.txt"), "w").write("\n".join(lines) + "\n")
