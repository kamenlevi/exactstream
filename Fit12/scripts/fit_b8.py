"""Fit12 B8 (2026-10-25; NewIdeas' bet B8, forecast-guided eviction; MoonShots' prior: Belady needs 8-16 passes of foresight,
perfect 1-2-pass prediction got 7-13 %). Cheapest decisive test: the eviction bound for ANY forecast with a horizon of F passes
= Belady with perfect foresight capped at F passes. Exact routing (every top-10 expert of every row waited, as v3 / B12LL +
MTP), per-layer caches of C experts, demand insertion. Traces: the 125B decode routing of Breakthrough's expA (4 prompts,
2,926 tokens; the same files as NewIdeas' belady_frugal.py); a pass = A consecutive tokens (A = 3 ~ MTP's accepted tokens per
pass; A = 1 = one-token passes); its request = the union of their top-10 ids per layer.
Policies: LRU; OPT-F = evict the cached expert (not needed now) whose next use is farthest, where next uses beyond F passes are
unknown (those go first, least recently used first); OPT = OPT-inf (Belady). Misses per token per policy (and the change vs LRU, negative = fewer); 'steady' drops
each prompt's first 64 tokens (cold start). C = 51 (line 2's C_m), 128, 192 (24 GB). Kill line (NewIdeas): a forecast with
the horizon B8 can have (<= 1-2 passes) must cut misses >= 10 % vs LRU -> if OPT-2 < 10 % at C 128 and 192, B8 is KILLED.
Usage: python fit_b8.py   (CPU, numpy)"""
import glob, time
import numpy as np

NL, E, WARM = 48, 512, 64
FS = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
FH = (1, 2, 4, 8, 16, None)                      # None = unbounded (Belady)


def passes(ids, A):
    """ids [T, 10] -> list of (pass request mask [E] bool, tokens in the pass, first token index)"""
    out = []
    for t0 in range(0, ids.shape[0], A):
        m = np.zeros(E, bool)
        m[ids[t0:t0 + A].reshape(-1)] = True
        out.append((m, min(A, ids.shape[0] - t0), t0))
    return out


def replay(P, C, F):
    """misses per pass for one layer's pass list P; F = 0 -> LRU, F > 0 -> OPT-F, F = None -> OPT"""
    n = len(P)
    INF = 1 << 40
    nxt = np.full((n, E), INF, np.int64)          # nxt[p, e] = next pass > p that uses e
    cur = np.full(E, INF, np.int64)
    for p in range(n - 1, -1, -1):
        nxt[p] = cur
        cur = np.where(P[p][0], p, cur)
    cache = np.zeros(E, bool)
    last = np.full(E, -1, np.int64)
    miss = np.zeros(n, np.int64)
    for p, (need, _, _) in enumerate(P):
        mi = need & ~cache
        m = int(mi.sum())
        over = int(cache.sum()) + m - C
        if over > 0:
            cand = np.flatnonzero(cache & ~need)
            if F == 0:
                key = -last[cand].astype(np.float64)                       # evict the least recently used
            else:
                d = nxt[p, cand] - p
                known = d <= (F if F is not None else INF)
                key = np.where(known, d.astype(np.float64), 1e15 - last[cand])   # unknown first (LRU order), then farthest
                # (1e15: float64 keeps integer steps there; the first run's 1e18 rounded every unknown key to one value)
            vic = cand[np.argpartition(-key, over - 1)[:over]] if over < len(cand) else cand
            cache[vic] = False
        cache |= need
        last[need] = p
        miss[p] = m
    return miss


def main():
    t0 = time.time()
    tr = []
    for f in FS:
        d = np.load(f)
        tr.append([d[f"L{l}"][:, :10].astype(np.int64) for l in range(NL)])
    ntok = sum(x[0].shape[0] for x in tr)
    print(f"[b8] {len(tr)} traces, {ntok} tokens, {NL} layers ({time.time() - t0:.1f} s)", flush=True)
    for A in (3, 1):
        for C in (51, 128, 192):
            tot, st = {}, {}
            for F in (0,) + FH:
                a = s = 0
                for x in tr:
                    for l in range(NL):
                        P = passes(x[l], A)
                        mi = replay(P, C, F)
                        a += int(mi.sum())
                        s += int(sum(m for m, (_, _, t) in zip(mi, P) if t >= WARM))
                tot[F], st[F] = a, s
            nst = sum(max(0, x[0].shape[0] - WARM) for x in tr)
            row = " | ".join(f"{'LRU' if F == 0 else 'OPT' if F is None else f'OPT-{F}'} {tot[F] / ntok:.2f} "
                             f"({100 * (tot[F] / tot[0] - 1):+.1f} %; steady {100 * (st[F] / st[0] - 1):+.1f} %)"
                             for F in (0,) + FH)
            print(f"A={A} C={C}: misses / token (all layers): {row}; steady LRU {st[0] / nst:.2f} ({time.time() - t0:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
