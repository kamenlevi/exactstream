"""D-02a first-miss table (MTP Head, Thu 10-29; REPORT 'D-02a'): how sure the 125B is where DFlash2 L1-20 first misses.
Walks (sticky/d02b_walk.py, block 16, official 1-step loop; per check [k accepted, m checkable, a anchor]): data/
d02b_walk_{code,math}.npz (D-02b) and data/d02b_walk_d02a_{ifeval,multi,agent}.npz. A check with k < m first misses the
token at position a + k + 1 (depth k + 1). The 125B's top-32 next-token log-probs for that position sit on row p - 1:
code from ~/Leap/data/l118_tgt (dfl2c format, rows from position 0), the others from data/d02a (rows from off).
Each first miss whose text token is the 125B's own top-1 there (on-path) is split into sure (p1 >= 0.99), near-tie
(p1 - p2 <= 0.1) or between; off-path misses (text token != the 125B's top-1 in this runtime) are counted apart. The
base rate is the same split over every drafted position of the completion (positions gen + 1 ..).
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d02a_miss.py -> data/d02a_miss.txt"""
import os
import numpy as np

H = os.path.expanduser
TD = H("~/TrainLab/data")
SETS = [("code", "d02b_walk_code.npz", H("~/Leap/data/l118_tgt")), ("math", "d02b_walk_math.npz", TD + "/d02a"),
        ("ifeval", "d02b_walk_d02a_ifeval.npz", TD + "/d02a"), ("multi", "d02b_walk_d02a_multi.npz", TD + "/d02a"),
        ("agent", "d02b_walk_d02a_agent.npz", TD + "/d02a")]
BANDS = [(1, 3), (4, 7), (8, 11), (12, 15)]
CATS = ("sure", "tie", "mid")
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)

def cat(p1, p2):
    return "sure" if p1 >= 0.99 else "tie" if p1 - p2 <= 0.1 else "mid"

say("D-02a first misses of DFlash2 L1-20 (block 16, 1 step) by the 125B's confidence on the missed token's row:")
say("sure = p1 >= 0.99, tie = p1 - p2 <= 0.1, mid = the rest; shares of on-path first misses; off = text token is not")
say("the 125B's own top-1 there (counted apart). base = the same split over all drafted completion positions.")
summary = {}
for dom, wf, pdir in SETS:
    W = np.load(os.path.join(TD, wf))
    miss = {b: {c: 0 for c in CATS + ("off",)} for b in BANDS}
    base = {c: 0 for c in CATS + ("off",)}
    align = [0, 0]
    for s, nm in enumerate(W["names"]):
        nm = str(nm)
        ids, off, gen, checks = W[f"{s}_ids"], int(W[f"{s}_off"]), int(W[f"{s}_gen"]), W[f"{s}_checks"]
        P = np.load(os.path.join(pdir, nm))
        poff = int(P["off"]) if "off" in P.files else 0
        tk, lp = P["tk"], P["lp"].astype(np.float32)
        def row(p):                                      # the 125B's distribution for the token at position p
            r = p - 1 - poff
            return r if 0 <= r < tk.shape[0] else None
        def text(p):
            return int(ids[p - off])
        for p in range(gen + 1, off + len(ids)):         # base rate over the drafted completion positions
            r = row(p)
            if r is None:
                continue
            align[0] += int(tk[r, 0]) == text(p); align[1] += 1
            if int(tk[r, 0]) != text(p):
                base["off"] += 1
            else:
                base[cat(np.exp(lp[r, 0]), np.exp(lp[r, 1]))] += 1
        for k, m, a in checks:
            k, m, a = int(k), int(m), int(a)
            if k >= m:
                continue
            p, j = a + k + 1, k + 1
            r = row(p)
            if r is None:
                continue
            b = next(x for x in BANDS if x[0] <= j <= x[1])
            if int(tk[r, 0]) != text(p):
                miss[b]["off"] += 1
            else:
                miss[b][cat(np.exp(lp[r, 0]), np.exp(lp[r, 1]))] += 1
    nb = sum(base[c] for c in CATS)
    say(f"  {dom}: 125B top-1 = text token on {align[0] / max(align[1], 1):.3f} of {align[1]} completion positions; "
        f"base (on-path) sure / tie / mid = " + " / ".join(f"{base[c] / max(nb, 1):.2f}" for c in CATS))
    tot = {c: 0 for c in CATS + ("off",)}
    for b in BANDS:
        n = sum(miss[b][c] for c in CATS)
        for c in tot:
            tot[c] += miss[b][c]
        if n + miss[b]["off"] == 0:
            continue
        say(f"    depths {b[0]:2d}-{b[1]:2d}: {n:5d} on-path first misses: sure / tie / mid = "
            + " / ".join(f"{miss[b][c] / max(n, 1):.2f}" for c in CATS) + f"; off-path {miss[b]['off']}")
    deep = {c: sum(miss[b][c] for b in BANDS[1:]) for c in CATS}
    nd = sum(deep.values())
    n = sum(tot[c] for c in CATS)
    say(f"    all depths: {n} on-path first misses: sure / tie / mid = " + " / ".join(f"{tot[c] / max(n, 1):.2f}" for c in CATS)
        + f"; depths >= 4: sure {deep['sure'] / max(nd, 1):.2f}, tie {deep['tie'] / max(nd, 1):.2f} ({nd} misses)")
    summary[dom] = (deep["sure"] / max(nd, 1), deep["tie"] / max(nd, 1), tot["sure"] / max(n, 1), tot["tie"] / max(n, 1))
say("Route (pre-registered, NewIdeas'): sure >= 0.50 of the first misses at depths >= 4 on code or math -> knowledge "
    "problem; tie >= 0.50 -> the exact ceiling there; else mixed.")
for dom in ("code", "math"):
    sd, td, _, _ = summary[dom]
    say(f"  {dom}: depths >= 4 sure {sd:.2f}, tie {td:.2f} -> "
        + ("knowledge" if sd >= 0.5 else "ceiling" if td >= 0.5 else "mixed"))
open(os.path.join(TD, "d02a_miss.txt"), "w").write("\n".join(lines) + "\n")
