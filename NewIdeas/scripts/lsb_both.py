"""How many LSB (sidecar) misses coincide with an MSB (base) miss of the same expert in the same check and layer?
Fable B's ROW_WALL sec 4 (adjacent MSB + LSB = one read). Same replay as band45_oea / calls_count (cap45, C 256, seed 0)."""
import os, sys, heapq, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.expanduser("~/NewIdeas/scripts"))
import band45_oea as B  # noqa: E402
LZ = B.LZ

def replay_sets(task):
    key, K, calls, pops = task
    out = []
    st, i = {}, 0
    for p, cl in enumerate(calls):
        pop = pops[p]; o = []
        for t, need in cl:
            i += 1
            if t >= 0: o.append(frozenset(e for e in need if e not in st))
            for e in need: st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + 30.0 * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf): del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf): del st[e]
        out.append(o)
    return out

if __name__ == "__main__":
    import pickle
    P, agr = pickle.load(open(B.CACHE, "rb"))   # laysz45.load()'s own output, written locally (trusted, as band45_oea)
    T = B.transplant(P, B.d23(), 0)
    for r in ("EXACT", "U3 = Q1 0.9 3b"):
        Q = B.needs(P, T, B.RULES["EXACT" if r == "EXACT" else "Q1 0.9 3b"])
        POP = {key: LZ.popularity(B.needs(P, T, B.RULES["EXACT"]), key) for key in ("m", "l")}
        with mp.get_context("fork").Pool(4) as pool:
            res = pool.map(replay_sets, B.tasks_of(Q, POP), chunksize=8)
        mm, ll = res[:B.NL], res[B.NL:]
        lsb = both = msb = 0
        for l in range(B.NL):
            for p in range(len(mm[l])):
                for a, b in zip(mm[l][p], ll[l][p]):
                    msb += len(a); lsb += len(b); both += len(a & b)
        print(f"{r}: MSB misses {msb}, LSB misses {lsb}, LSB misses whose MSB also missed {both} = {100*both/max(lsb,1):.1f} % of LSB misses -> adjacent MSB+LSB saves {100*both/(msb+lsb):.1f} % of reads", flush=True)
