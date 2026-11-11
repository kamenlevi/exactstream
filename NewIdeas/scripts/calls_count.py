"""CALLS-COUNT (NewIdeas, Tue 3 Nov 2026 17:2x; CPU only, writes nothing): how Fable A's per-call SSD latency (SSD_WALL2:
busy = 0.283 ms per loader call with misses + MB / 12.7 GB/s, fitted on C50's six C 256 medians) moves when a FAST stack
cuts the bytes. Same replay as band45_oea.py (cap45's live checks, D-23 cf transplants, LRU + 30 x popularity, strict, res0
forced), but per (check, layer): a 'call' is a layer whose base row (or sidecar row) has >= 1 miss; JOINT = one call per
layer with any miss. Prints per rule: misses, calls (base + sidecar rows), JOINT calls, MB per check, and the fitted busy
split (fixed vs bytes) with and without JOINT. Usage (BigMoE venv): [BAND45_SIZES=laysz45_320.json] python calls_count.py"""
import os, sys, heapq, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45_oea as B  # noqa: E402
LZ = B.LZ
PER_CALL, GBS = 0.283, 12.7                              # Fable A's fit (2) [K six medians]


def replay_calls(task):
    """B.replay's rules for one layer's pool; misses per (prompt, check)"""
    key, K, calls, pops = task
    while os.path.exists(LZ.PAUSE):
        time.sleep(20)
    out = [np.zeros(sum(1 for t, _ in cl if t >= 0)) for cl in calls]
    st, i = {}, 0
    for p, cl in enumerate(calls):
        pop = pops[p]
        for t, need in cl:
            i += 1
            if t >= 0:
                out[p][t] = sum(1 for e in need if e not in st)
            for e in need:
                st[e] = float(i)
            if len(st) > K:
                kf = lambda e: st[e] + 30.0 * pop[e]
                for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                    del st[e]
                if len(st) > K:
                    for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                        del st[e]
    return out


if __name__ == "__main__":
    t0 = time.time()
    if os.path.exists(B.CACHE):
        import pickle; P, agr = pickle.load(open(B.CACHE, "rb"))  # laysz45.load()'s own output (trusted, as band45_oea)
    else:
        P, agr = LZ.load()
    D = B.d23(); T = B.transplant(P, D, 0)
    names = ["EXACT", "Q1 0.9", "Q1 0.9 3b", "HYB 0.9 u8 3b", "Q1 0", "Q1 0 3b"]
    rules = {r: B.RULES[r] for r in names if r in B.RULES}
    Qs = {r: B.needs(P, T, f) for r, f in rules.items()}
    POP = {key: LZ.popularity(Qs["EXACT"], key) for key in ("m", "l")}
    print(f"pools: {os.environ.get('BAND45_SIZES', 'laysz45_256.json')}; seed 0; per-call {PER_CALL} ms, {GBS} GB/s (Fable A's fit)")
    print("| rule | misses per check (base + LSB) | calls per check (base + LSB rows) | JOINT calls | MB per check | busy ms: fixed + bytes (fixed share) | with JOINT |")
    print("|---|---|---|---|---|---|---|")
    with mp.get_context("fork").Pool(4) as pool:
        for r, Q in Qs.items():
            res = pool.map(replay_calls, B.tasks_of(Q, POP), chunksize=8)
            mm, ll = res[: B.NL], res[B.NL:]                   # [layer][prompt] -> misses per check
            n = sum(len(x) for x in mm[0])
            bm = sum(float(np.sum(np.concatenate(mm[l]))) for l in range(B.NL)) / n
            bl = sum(float(np.sum(np.concatenate(ll[l]))) for l in range(B.NL)) / n
            cm = sum(float(np.sum(np.concatenate(mm[l]) > 0)) for l in range(B.NL)) / n
            cl = sum(float(np.sum(np.concatenate(ll[l]) > 0)) for l in range(B.NL)) / n
            cj = sum(float(np.sum((np.concatenate(mm[l]) + np.concatenate(ll[l])) > 0)) for l in range(B.NL)) / n
            mb = (bm * B.MR + bl * B.LR) / 1e6
            fx, by = PER_CALL * (cm + cl), mb / GBS
            fj = PER_CALL * cj
            print(f"| {r} | {bm + bl:.1f} ({bm:.1f} + {bl:.1f}) | {cm + cl:.1f} ({cm:.1f} + {cl:.1f}) | {cj:.1f} | {mb:.0f} | "
                  f"{fx:.1f} + {by:.1f} = {fx + by:.1f} ({100 * fx / (fx + by):.0f} %) | {fj:.1f} + {by:.1f} = {fj + by:.1f} |", flush=True)
    print(f"({n} checks, {time.time() - t0:.0f} s)")
