"""R-18 PAR24 (NewIdeas Tue ~19:40, QUEUE RAPID SCREENS): do lockstep parallel answer streams (SoT-style, own KV / GDN
state each) cut SSD bytes per token at 24 GB? CPU replay through MoonShots' wall24.py (engine LFU d 0.99, per-layer
pools C_m 147 MSB-3 / C_l 46 LSB-1, plan 4433333333) on the expA traces (bulgarian / chat / code / math, <= 800 tokens).
PASS: 2 same-answer streams cut bytes / token >= 20 % vs 1 stream (mean of the 4 sets, 16-row checks). KILL: < 8 %.
Design (fixed before the run): k streams in lockstep; at each step every stream runs one check (R rows accepting A),
one gated call per layer per step with the union of all streams' needs (LSB if the best rank among those rows is < 2).
Same answer: one trace cut into k contiguous segments. Control: segment s taken from another set's trace (set i + s).
Warm-up = the first half of the steps, measured = the second half; bytes / token = measured bytes / (k A steps).
k = 1 is the same rule on the whole trace. Primary R 16 / A 10 (wall24's 24 GB gap figure); secondary R 4 / A 3
(the shipped 24 GB line, S ~3). usage: par24.py  (CPU, one core)"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import wall24 as w                                   # loads the expA traces (IDS: set -> (n, 48, 10) top-10 ids)

SETS = list(w.IDS)
BIG = [0, 0]                                         # largest union per call seen (MSB, LSB), vs the pools 147 / 46


def lfu(seq, C, d=0.99):
    """wall24's 'lfu' (the engine's rule: per call w /= d, needed += w, forced admission, victim = lowest score not
    needed now), plus the pool cap: a call that needs more than C experts holds the excess as scratch, and after the
    call only the C best scores stay resident (wall24's run_pool lets the pool grow past C there, which would flatter
    k >= 2 streams). Identical to run_pool when no call needs more than C."""
    R, score, wt, miss = set(), {}, 1.0, []
    for N in seq:
        wt /= d
        for e in N:
            score[e] = score.get(e, 0.0) + wt
        m = [e for e in N if e not in R]
        miss.append(len(m))
        for e in m:
            if len(R) >= C:
                vic = [v for v in R if v not in N]
                if vic:
                    R.discard(min(vic, key=lambda v: score[v]))
            R.add(e)
        if len(R) > C:
            R = set(sorted(R, key=lambda v: -score[v])[:C])
    return miss


def arm(streams, R, A):
    """streams: list of (set, start, length). Returns (bytes, tokens) over the measured half of the lockstep steps."""
    per = [w.needs(w.IDS[s][a:a + n], R, A) for s, a, n in streams]
    steps = min(len(p) for p in per)
    tot = 0
    for l in range(w.NL):
        msb = [frozenset().union(*(p[j][1][l][0] for p in per)) for j in range(steps)]
        lsb = [frozenset().union(*(p[j][1][l][1] for p in per)) for j in range(steps)]
        BIG[0], BIG[1] = max(BIG[0], max(map(len, msb))), max(BIG[1], max(map(len, lsb)))
        mm, ml = lfu(msb, 147), lfu(lsb, 46)
        tot += sum(a * w.MSB + b * w.LSB for a, b in list(zip(mm, ml))[steps // 2:])
    return tot, len(streams) * A * (steps - steps // 2)


def run(R, A):
    res = {}
    for i, s in enumerate(SETS):
        n = len(w.IDS[s])
        res[(1, "same", s)] = arm([(s, 0, n)], R, A)
        for k in (2, 4):
            res[(k, "same", s)] = arm([(s, j * (n // k), n // k) for j in range(k)], R, A)
            ctl = [SETS[(i + j) % len(SETS)] for j in range(k)]
            res[(k, "ctrl", s)] = arm([(c, j * (len(w.IDS[c]) // k), len(w.IDS[c]) // k) for j, c in enumerate(ctl)], R, A)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(__doc__.split("\n")[3])
    ch = w.needs(w.IDS[SETS[0]], 16, 10)                # the capped LFU = wall24's run_pool where nothing overflows
    for l in (0, 24, 47):
        seq = [lay[l][0] for _, lay in ch]
        print(f"check vs wall24.run_pool, {SETS[0]} layer {l}: {'EQUAL' if lfu(seq, 147) == w.run_pool(seq, 147, 'lfu') else 'DIFFER'}")
    for R, A in ((16, 10), (4, 3)):
        BIG[:] = [0, 0]
        res = run(R, A)
        mb = {key: b / t / 1e6 for key, (b, t) in res.items()}
        base = sum(mb[(1, "same", s)] for s in SETS) / len(SETS)
        print(f"\nR {R} / A {A}: MB per token (" + ", ".join(SETS) + ") and mean; change vs k = 1")
        for k, kind in ((1, "same"), (2, "same"), (2, "ctrl"), (4, "same"), (4, "ctrl")):
            v = [mb[(k, kind, s)] for s in SETS]
            m = sum(v) / len(v)
            print(f"  k {k} {kind}: " + " ".join(f"{x:6.1f}" for x in v) + f"  mean {m:6.1f}  {100 * (m / base - 1):+6.1f} %")
        print(f"  largest union in one call: MSB {BIG[0]} (pool 147), LSB {BIG[1]} (pool 46); larger calls hold scratch")
    print(f"\n{time.time() - t0:.0f} s")
