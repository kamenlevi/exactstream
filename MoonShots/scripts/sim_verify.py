"""Event-driven model of SSD streaming in one S=3 verify pass (BigMoE 125B), on the MoEPrefetch proxy traces (35B routing:
40 layers x 256 experts, top-8; results scaled x48/40 layers). Per-layer LRU at 55 % resident plus a spare pool for prefetches.
Timeline per layer L: attention/GDN, router (needed experts known), wait for missing experts (stall), MoE compute.
SSD: FIFO, each read = LAT + bytes/BW, at most QD reads in flight. Policies for *when* the CPU learns which experts to read:
  none      : at L's router only (demand)
  v1(B)     : + layer L+1's look-ahead prediction (pmid1, top-B per row, union) issued at L's router (BigMoE prefetch v1)
  v1+d2(B)  : + layer L+2's prediction (pmid2) issued at L's router too
  runahead  : v1, plus during a stall at L the GPU runs ahead through L+1's attention with substitute experts and learns
              L+1's *exact* routing RA ms after the stall starts (if the stall lasts that long); then reads the rest
  oracle    : every layer's misses known at pass start (upper bound for any early-routing scheme)
Calibrate LAT/BW on BigMoE's measured points (none: 40.5 ms streaming, 36.5 stalled layers; v1 B=12: 26 ms, 27 stalled)."""
import os, sys
from collections import OrderedDict
import numpy as np

TR = os.path.expanduser("~/MoEPrefetch/traces")
NL, K, E, S = 40, 8, 256, 3
SETS = ["chat", "code", "realcode"]
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS]
C = int(0.55 * E)
COMPUTE = 38.0 / 48          # ms of GPU work per layer in an S=3 verify pass (BigMoE: 38 ms compute)
ATTN = 0.5 * COMPUTE         # share before the router (attention/GDN + HC)
GATE = 0.07                  # ms gate round trip on a stalled layer
SZ = 2.77                    # MB per 125B expert


def run(policy, B=12, LAT=0.35, BW=13.0, RA=None, spare=12, maxpass=400):
    RA = ATTN if RA is None else RA
    st_ms, st_layers, passes, misses = 0.0, 0, 0, 0
    for z in zs:
        T = len(z["doc"])
        acts = [z[f"L{L}_act"][:, :K] for L in range(NL)]
        p1 = [z[f"L{L}_pmid1"][:, :B] for L in range(NL - 1)]
        p2 = [z[f"L{L}_pmid2"][:, :B] for L in range(NL - 2)]
        main = [OrderedDict() for _ in range(NL)]
        for t0 in range(200, min(T - S, 200 + S * maxpass), S):     # skip the cold start
            passes += 1
            now, ssd_free = 0.0, 0.0
            arrive = [dict() for _ in range(NL)]                   # expert -> arrival time (in flight or landed this pass)

            def issue(L, experts, t):
                nonlocal ssd_free
                for e in experts:
                    if e in main[L] or e in arrive[L]:
                        continue
                    start = max(t, ssd_free - LAT)                  # FIFO: bandwidth shared, latency overlaps
                    done = max(t + LAT, ssd_free) + SZ / BW
                    ssd_free = done
                    arrive[L][e] = done
            need = [list(dict.fromkeys(acts[L][t0:t0 + S].reshape(-1).tolist())) for L in range(NL)]
            if policy == "oracle":
                for L in range(NL):
                    issue(L, need[L], 0.0)
            for L in range(NL):
                now += ATTN
                if policy in ("v1", "v1+d2", "runahead") and L + 1 < NL:     # at most `spare` prefetch reads per layer
                    pred = list(dict.fromkeys(p1[L][t0:t0 + S].reshape(-1).tolist()))
                    issue(L + 1, [e for e in pred if e not in main[L + 1] and e not in arrive[L + 1]][:spare], now)
                if policy == "v1+d2" and L + 2 < NL:
                    pred = list(dict.fromkeys(p2[L][t0:t0 + S].reshape(-1).tolist()))
                    issue(L + 2, [e for e in pred if e not in main[L + 2] and e not in arrive[L + 2]][:spare], now)
                miss = [e for e in need[L] if e not in main[L]]
                issue(L, miss, now)
                wait = max([arrive[L][e] for e in miss] + [now]) - now
                if miss and wait > 0:
                    wait += GATE
                    st_layers += 1
                    if policy == "runahead" and L + 1 < NL and wait > RA:
                        issue(L + 1, need[L + 1], now + RA)          # exact L+1 routing learned during the stall
                now += wait
                st_ms += wait
                misses += len(miss)
                for e in need[L]:                                    # LRU update / insert
                    main[L][e] = 1
                    main[L].move_to_end(e)
                while len(main[L]) > C:
                    main[L].popitem(last=False)
                now += COMPUTE - ATTN
    sc = 48 / NL
    return st_ms / passes * sc, st_layers / passes * sc, misses / passes * sc * (10 / 8)


if __name__ == "__main__":
    LAT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    BW = float(sys.argv[2]) if len(sys.argv) > 2 else 13.0
    print(f"S=3 verify pass, 55 % resident, LAT={LAT} ms, BW={BW} GB/s, compute {COMPUTE*48:.0f} ms/pass "
          f"(BigMoE measured: none 40.5 ms / 36.5 stalled; v1 B=12: 26 ms / 27 stalled)")
    for pol, B in (("none", 12), ("v1", 12), ("v1", 16), ("v1+d2", 12), ("v1+d2", 16), ("runahead", 12),
                   ("runahead", 16), ("oracle", 12)):
        ms, sl, mi = run(pol, B, LAT, BW)
        print(f"  {pol:9s} B={B:>2}: streaming {ms:5.1f} ms/pass, stalled layers {sl:4.1f}, demand misses {mi:5.1f}", flush=True)
