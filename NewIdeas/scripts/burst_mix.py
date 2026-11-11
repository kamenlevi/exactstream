"""BURST-MIX (NewIdeas, Tue 3 Nov 2026 22:5x; CPU only, writes data/sprint/burst_mix_<tag>.txt): two numbers for the
predictions of BigMoE's Wednesday lever-1 arms (docs/WED_TESTS.md item 2), from the calls_count.py replay (cap45's live
checks, LRU + 30 x popularity, the pools in BAND45_SIZES).
1. BIGMOE_COPY_PAR (w1td) splits a record's copy over 3 threads only in loader calls with < 4 jobs. Under JOINT the base
   row's misses are one call; every sidecar miss is its own 1-job call (joint_side's dispatch_async per id). So: base
   calls per check by size (1-3 = eligible, >= 4 = not), and sidecar reads per check.
2. BIGMOE_POST_SYS (w1tp): on a layer with NO resident hit the GPU posts its routing and spins at once, and Leap's
   W1-PICKUP probe found that a device-scope post followed only by a spin can stay unseen for ms. So: layers per check
   whose needed base (or sidecar) set misses entirely.
Usage (BigMoE venv): [BAND45_SIZES=<json>] [BM_TAG=c160] python burst_mix.py"""
import os, sys, heapq, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import band45_oea as B  # noqa: E402
LZ = B.LZ
TAG = os.environ.get("BM_TAG", "c256")
DST = os.path.expanduser(f"~/NewIdeas/data/sprint/burst_mix_{TAG}.txt")


def replay_mix(task):
    """calls_count.replay_calls's rules; per (prompt, check): (misses, needed ids) of this layer's pool"""
    key, K, calls, pops = task
    while os.path.exists(LZ.PAUSE):
        time.sleep(20)
    out = [np.zeros((sum(1 for t, _ in cl if t >= 0), 2)) for cl in calls]
    st, i = {}, 0
    for p, cl in enumerate(calls):
        pop = pops[p]
        for t, need in cl:
            i += 1
            if t >= 0:
                out[p][t] = (sum(1 for e in need if e not in st), len(need))
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
    Q = B.needs(P, T, B.RULES["EXACT"])
    POP = {key: LZ.popularity(Q, key) for key in ("m", "l")}
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(replay_mix, B.tasks_of(Q, POP), chunksize=8)
    mm = [np.concatenate(res[l]) for l in range(B.NL)]               # base pool: [checks, (misses, need)]
    ll = [np.concatenate(res[B.NL + l]) for l in range(B.NL)]        # sidecar pool
    n = len(mm[0])
    bm = np.stack([m[:, 0] for m in mm], 1); bn = np.stack([m[:, 1] for m in mm], 1)
    sm = np.stack([x[:, 0] for x in ll], 1); sn = np.stack([x[:, 1] for x in ll], 1)
    calls = bm > 0
    small = (bm > 0) & (bm < 4)
    lines = [f"BURST-MIX {TAG} (pools {os.environ.get('BAND45_SIZES', 'laysz45_256.json')}; EXACT; {n} checks, 48 layers; replay)"]
    lines.append(f"misses per check: base {bm.sum(1).mean():.1f} + sidecar {sm.sum(1).mean():.1f}; needed per layer: base "
                 f"{bn.mean():.1f}, sidecar {sn.mean():.1f}")
    lines.append(f"base loader calls per check {calls.sum(1).mean():.1f}: with 1-3 jobs (COPY_PAR-eligible) "
                 f"{small.sum(1).mean():.1f} ({100 * small.sum() / max(calls.sum(), 1):.0f} %), with >= 4 jobs "
                 f"{(bm >= 4).sum(1).mean():.1f}; jobs per base call: mean {bm[calls].mean():.1f}, median "
                 f"{np.median(bm[calls]):.0f}")
    lines.append(f"sidecar reads per check {sm.sum(1).mean():.1f} (under JOINT each is its own 1-job call: COPY_PAR applies)")
    hist = np.bincount(np.minimum(bm[calls].astype(int), 12), minlength=13)[1:]
    lines.append("base jobs per call, share 1..11, 12+: " + " ".join(f"{100 * h / calls.sum():.0f}" for h in hist))
    zb = (bn > 0) & (bm == bn); zs = (sn > 0) & (sm == sn)
    lines.append(f"zero-hit layers per check: base {zb.sum(1).mean():.2f} (checks with >= 1: {100 * (zb.sum(1) > 0).mean():.0f} %), "
                 f"sidecar {zs.sum(1).mean():.2f} (checks with >= 1: {100 * (zs.sum(1) > 0).mean():.0f} %); both on one layer "
                 f"{(zb & zs).sum(1).mean():.2f}")
    lines.append(f"({time.time() - t0:.0f} s)")
    print("\n".join(lines), flush=True)
    open(DST, "w").write("\n".join(lines) + "\n")
