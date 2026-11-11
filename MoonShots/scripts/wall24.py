"""24 GB SSD wall (NewIdeas Sun ~13:20): the ceiling for ANY residency policy (eviction, admission, prefetch-into-pool) at
v3's 24 GB split, on the expA traces (bulgarian / chat / code / math, 800 tokens, top-10 of the rank-ordered top-20).
v3 plan 4433333333: every needed expert needs its MSB-3 part (pool C_m 147), ranks 0-1 also the LSB-1 part (pool C_l 46;
LSB needed when the expert's best rank among the check's rows is < 2). 16-row checks accepting 10 (one gated call per
layer per check with the union of the rows' needs), tokens 0-399 warm the pools, checks starting at 400+ are measured.
Policies per pool (independent per layer):
  lru     the old default
  lfu     the engine's BIGMOE_EVICT_LFU=d rule (~/BigMoE/ext/bigmoe_ext.cpp ensure / make_room / lfu_victim): per call
          w /= d, every needed expert += w; a miss evicts the lowest-score resident not needed by this call
  belady  forced admission like the engine (every needed expert is resident after its call), the victim is the
          resident not needed now whose next use is farthest (perfect foresight)
  min     Belady's MIN with bypass: after each call keep the C experts with the nearest next use among residents and
          this call's needs (a needed expert may be read into scratch and dropped). Optimal for equal-size objects: no
          policy of any kind can read fewer bytes at this pool size
usage: wall24.py   (CPU only, one core)"""
import glob, os, time
import numpy as np

NL = 48
MSB, LSB = 2162688, 770048
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
SETS = {os.path.basename(f).split("_")[0]: np.load(f) for f in sorted(glob.glob(T + "*_800_g.npz"))}
IDS = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.int32) for k, d in SETS.items()}
INF = 1 << 30


MEAS = (400, 800)                                  # measured check starts [a, b); the trace (Belady's future) runs to 800


def needs(ids, R, A):
    """per check: (measured?, per layer (msb need set, lsb need set))"""
    n, out, pos = ids.shape[0], [], 0
    while pos + R <= n:
        lay = []
        for l in range(NL):
            best = {}
            for r in range(R):
                for k, e in enumerate(ids[pos + r, l]):
                    e = int(e)
                    if e not in best or k < best[e]:
                        best[e] = k
            lay.append((frozenset(best), frozenset(e for e, k in best.items() if k < 2)))
        out.append((MEAS[0] <= pos < MEAS[1], lay))
        pos += A
    return out


def run_pool(seq, C, policy, d=0.99):
    """seq: list of need sets (one per call). Returns misses per call."""
    miss = []
    H = None
    if policy.startswith("min") and policy != "min" or policy.startswith("ws"):
        H = int(policy[3:] if policy.startswith("min") else policy[2:])
    if policy.startswith("min") or policy == "belady" or policy.startswith("ws"):
        uses = {}
        for t, N in enumerate(seq):
            for e in N:
                uses.setdefault(e, []).append(t)
        ptr = {e: 0 for e in uses}

        def nxt(e, t):                               # first use strictly after t
            u, i = uses[e], ptr[e]
            while i < len(u) and u[i] <= t:
                i += 1
            ptr[e] = i
            return u[i] if i < len(u) else INF
    R = set()
    if policy == "lfub" or policy.startswith("gap"):
        return run_online(seq, C, policy, d)
    if policy == "lru":
        from collections import OrderedDict
        od = OrderedDict()
    score, w = {}, 1.0
    for t, N in enumerate(seq):
        if policy == "lru":
            m = [e for e in N if e not in od]
            miss.append(len(m))
            for e in N:
                if e in od:
                    od.move_to_end(e)
            for e in m:
                while len(od) >= C:
                    for v in od:                        # oldest not needed now
                        if v not in N:
                            del od[v]
                            break
                    else:
                        break
                od[e] = True
            continue
        if policy == "lfu" or H is not None:
            w /= d
            for e in N:
                score[e] = score.get(e, 0.0) + w
        m = [e for e in N if e not in R]
        miss.append(len(m))
        if policy.startswith("min") or policy.startswith("ws"):
            cand = R | N
            if len(cand) > C:
                if H is None:
                    key = lambda e: nxt(e, t)
                elif policy.startswith("min"):
                    key = lambda e: (nxt(e, t) if nxt(e, t) - t <= H else INF, -score[e])
                else:
                    key = lambda e: (0 if nxt(e, t) - t <= H else 1, -score[e])
                R = set(sorted(cand, key=key)[:C])
            else:
                R = cand
            continue
        for e in m:                                  # lfu / belady: forced admission, one victim per miss when full
            if len(R) >= C:
                vic = [v for v in R if v not in N]
                if vic:
                    if policy == "lfu":
                        v = min(vic, key=lambda v: score[v])
                    else:
                        v = max(vic, key=lambda v: nxt(v, t))
                    R.discard(v)
            R.add(e)
    return miss


_NEEDS = {}


def run_online(seq, C, policy, d=0.99):
    miss, R, score, w = [], set(), {}, 1.0
    forced = policy.endswith("_f")
    if policy.startswith("gap"):
        a = int(policy[3:].split("_")[0]) / 10
    last, g, G0 = {}, {}, 40.0
    for t, N in enumerate(seq):
        w /= d
        for e in N:
            score[e] = score.get(e, 0.0) + w
        m = [e for e in N if e not in R]
        miss.append(len(m))
        if policy == "lfub":
            cand = R | N
            R = set(sorted(cand, key=lambda e: -score[e])[:C]) if len(cand) > C else cand
            continue
        for e in N:
            if e in last:
                g[e] = a * (t - last[e]) + (1 - a) * g[e]
            else:
                g[e] = G0
            last[e] = t

        def pred(e):
            p = last[e] + g[e]
            return p if p > t else t + g[e]
        if forced:
            for e in m:
                if len(R) >= C:
                    vic = [v for v in R if v not in N]
                    if vic:
                        R.discard(max(vic, key=pred))
                R.add(e)
        else:
            cand = R | N
            R = set(sorted(cand, key=pred)[:C]) if len(cand) > C else cand
    return miss


def bytes_per_token(policy, Cm=147, Cl=46, R=16, A=10, d=0.99):
    res = {}
    for k, ids in IDS.items():
        if (k, R, A) not in _NEEDS:
            _NEEDS[(k, R, A)] = needs(ids, R, A)
        ch = _NEEDS[(k, R, A)]
        meas = [m for m, _ in ch]
        tot = 0
        for l in range(NL):
            mm = run_pool([lay[l][0] for _, lay in ch], Cm, policy, d)
            ml = run_pool([lay[l][1] for _, lay in ch], Cl, policy, d)
            tot += sum(a * MSB + b * LSB for a, b, f in zip(mm, ml, meas) if f)
        res[k] = tot / (A * sum(meas))
    return float(np.mean(list(res.values()))), res


PROB = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in SETS.items()}
IDS20 = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in SETS.items()}


def hint_seq(k, l, R=16, A=10, mode="r", beta=0.5):
    """per check at layer l: dict expert -> LFU increment factor from the check's rows' top-20 (the router's near misses).
    mode r: 1 for an expert in some row's top-10, beta for one only in ranks 10-19; mode p: sum over rows of its gate prob
    (normalised by the call's mean over its needed experts)"""
    ids, pr, out, pos, n = IDS20[k], PROB[k], [], 0, IDS20[k].shape[0]
    while pos + R <= n:
        h = {}
        for r in range(R):
            for j in range(20):
                e = int(ids[pos + r, l, j])
                if mode == "r":
                    h[e] = max(h.get(e, 0.0), 1.0 if j < 10 else beta)
                else:
                    h[e] = h.get(e, 0.0) + float(pr[pos + r, l, j])
        if mode == "p":
            nd = [v for e, v in h.items() if any(int(x) == e for x in ids[pos:pos + R, l, :10].reshape(-1))]
            mu = sum(nd) / max(1, len(nd))
            h = {e: v / mu for e, v in h.items()}
        out.append(h)
        pos += A
    return out


def run_lfu_hint(seq, hints, C, d=0.99):
    """the engine's LFU (forced admission, victim = lowest score not needed now), scores fed by the hint dicts"""
    miss, R, score, w = [], set(), {}, 1.0
    for N, h in zip(seq, hints):
        w /= d
        for e, f in h.items():
            score[e] = score.get(e, 0.0) + w * f
        for e in N:
            if e not in h:
                score[e] = score.get(e, 0.0) + w
        m = [e for e in N if e not in R]
        miss.append(len(m))
        for e in m:
            if len(R) >= C:
                vic = [v for v in R if v not in N]
                if vic:
                    R.discard(min(vic, key=lambda v: score[v]))
            R.add(e)
    return miss


def hints():
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say("LFU fed by the router's near misses (top-20), MSB pool only (LSB pool: plain LFU), 16-row checks accepting 10,"
        " measured tokens [400, 800); MB/token")
    lfu, _ = bytes_per_token("lfu")
    say(f"  lfu (engine) {lfu / 1e6:6.1f}")
    for mode, beta in (("r", 0.0), ("r", 0.25), ("r", 0.5), ("r", 1.0), ("p", 0.0)):
        res = {}
        for k, ids in IDS.items():
            if (k, 16, 10) not in _NEEDS:
                _NEEDS[(k, 16, 10)] = needs(ids, 16, 10)
            ch = _NEEDS[(k, 16, 10)]
            meas = [m for m, _ in ch]
            tot = 0
            for l in range(NL):
                hs = hint_seq(k, l, mode=mode, beta=beta)[:len(ch)]
                mm = run_lfu_hint([lay[l][0] for _, lay in ch], hs, 147)
                ml = run_pool([lay[l][1] for _, lay in ch], 46, "lfu")
                tot += sum(a * MSB + b * LSB for a, b, f in zip(mm, ml, meas) if f)
            res[k] = tot / (10 * sum(meas))
        b = float(np.mean(list(res.values())))
        say(f"  mode {mode} beta {beta:.2f}: {b / 1e6:6.1f}  {100 * (b / lfu - 1):+5.1f} % vs lfu  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in res.items()) + ")")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24_hints.txt"), "w").write("\n".join(lines) + "\n")


def run_global(chs, Cm, Cl, policy="lfu", d=0.99, protect=False, prev=False):
    """one MSB pool and one LSB pool shared by all 48 layers (records are the same size in every layer): calls in pass
    order (check t: layers 0..47). lfu: per (layer, expert) decayed counts (each layer's w advances once per check, so
    scores compare across layers); a miss evicts the globally lowest score not needed by this call (lazy min-heap).
    Returns bytes over measured checks."""
    import heapq
    tot = 0
    for part, C, sz in ((0, Cm, MSB), (1, Cl, LSB)):
        R, score, w, hp = set(), {}, 1.0, []
        last = [set() for _ in range(NL)]                 # prev: the previous check's needs per layer (online proxy)
        for meas, lay in chs:
            w /= d
            P = {(l, e) for l in range(NL) for e in lay[l][part]} if protect else None   # the whole check's needs
            for l in range(NL):
                if prev:                                  # upcoming layers of this check, as last check needed them
                    P = {(j, e) for j in range(l + 1, NL) for e in last[j]}
                    last[l] = set(lay[l][part])
                N = {(l, e) for e in lay[l][part]}
                for x in N:
                    score[x] = score.get(x, 0.0) + w
                m = [x for x in N if x not in R]
                if meas:
                    tot += len(m) * sz
                for x in m:
                    if len(R) >= C:
                        aside = []
                        while hp:
                            sc, v = heapq.heappop(hp)
                            if v not in R or sc != score[v]:
                                continue                     # stale entry
                            if v in N or (P is not None and v in P):
                                aside.append((sc, v)); continue
                            R.discard(v); break
                        for a in aside:
                            heapq.heappush(hp, a)
                    R.add(x)
                for x in N:                                  # needed experts' scores changed: fresh heap entries
                    heapq.heappush(hp, (score[x], x))
    return tot


def shared():
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    lfu, per = bytes_per_token("lfu")
    say(f"per-layer pools (engine) lfu: {lfu / 1e6:6.1f} MB/token")
    res = {}
    for k, ids in IDS.items():
        if (k, 16, 10) not in _NEEDS:
            _NEEDS[(k, 16, 10)] = needs(ids, 16, 10)
        ch = _NEEDS[(k, 16, 10)]
        res[k] = run_global(ch, 147 * NL, 46 * NL) / (10 * sum(m for m, _ in ch))
    b = float(np.mean(list(res.values())))
    say(f"one shared pool (C_m {147 * NL}, C_l {46 * NL}) lfu: {b / 1e6:6.1f}  {100 * (b / lfu - 1):+5.1f} %  (" +
        ", ".join(f"{k} {v / 1e6:.1f}" for k, v in res.items()) + ")")
    res = {}
    for k, ids in IDS.items():
        ch = _NEEDS[(k, 16, 10)]
        res[k] = run_global(ch, 147 * NL, 46 * NL, protect=True) / (10 * sum(m for m, _ in ch))
    b = float(np.mean(list(res.values())))
    res2 = {}
    for k, ids in IDS.items():
        ch = _NEEDS[(k, 16, 10)]
        res2[k] = run_global(ch, 147 * NL, 46 * NL, prev=True) / (10 * sum(m for m, _ in ch))
    b2 = float(np.mean(list(res2.values())))
    say(f"shared pool, victims exclude upcoming layers' needs of the PREVIOUS check (online): {b2 / 1e6:6.1f}  "
        f"{100 * (b2 / lfu - 1):+5.1f} %  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in res2.items()) + ")")
    say(f"shared pool, victims exclude the whole check's needs (oracle within the check): {b / 1e6:6.1f}  "
        f"{100 * (b / lfu - 1):+5.1f} %  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in res.items()) + ")")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24_shared.txt"), "w").write("\n".join(lines) + "\n")


def real_rows():
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    for R, A in ((16, 10), (11, 9)):
        lfu, _ = bytes_per_token("lfu", R=R, A=A)
        say(f"R {R} A {A}: lfu {lfu / 1e6:6.1f} MB/token")
        for pol in ["min1", "min2", "min5", "belady", "min"]:
            b, _ = bytes_per_token(pol, R=R, A=A)
            say(f"  {pol:6s} {b / 1e6:6.1f}  {100 * (b / lfu - 1):+5.1f} % vs lfu")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24_rows.txt"), "w").write("\n".join(lines) + "\n")


def online():
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say("online policies (no foresight), 16-row checks accepting 10, measured tokens [400, 800); MB/token")
    lfu, _ = bytes_per_token("lfu")
    say(f"  lfu (engine) {lfu / 1e6:6.1f}")
    for pol in ["lfub", "gap3", "gap5", "gap8", "gap3_f", "gap5_f"]:
        b, per = bytes_per_token(pol)
        say(f"  {pol:7s} {b / 1e6:6.1f}  {100 * (b / lfu - 1):+5.1f} % vs lfu  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in per.items()) + ")")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24_online.txt"), "w").write("\n".join(lines) + "\n")


def sweep():
    global MEAS
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    for win in ((400, 800), (400, 600)):
        MEAS = win; _NEEDS.clear()
        say(f"measured checks starting in tokens [{win[0]}, {win[1]}) (future known to 800); 16-row checks accepting 10; MB/token")
        lfu, _ = bytes_per_token("lfu")
        say(f"  lfu (engine)      {lfu / 1e6:6.1f}")
        for pol in ["min2", "min5", "min10", "min20", "min40", "ws5", "ws10", "ws20", "ws40", "min"]:
            b, per = bytes_per_token(pol)
            say(f"  {pol:6s} {b / 1e6:6.1f}  {100 * (b / lfu - 1):+5.1f} % vs lfu  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in per.items()) + ")")
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24_horizon.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__" and os.environ.get("WALL_SHARED"):
    shared()
elif __name__ == "__main__" and os.environ.get("WALL_ROWS"):
    real_rows()
elif __name__ == "__main__" and os.environ.get("WALL_HINTS"):
    hints()
elif __name__ == "__main__" and os.environ.get("WALL_ONLINE"):
    online()
elif __name__ == "__main__" and os.environ.get("WALL_SWEEP"):
    sweep()
elif __name__ == "__main__":
    t0 = time.time()
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    say("v3 24 GB split C_m 147 / C_l 46, plan 4433333333, MB per token (mean of 4 sets: bulgarian chat code math)")
    base = {}
    for R, A in ((1, 1), (16, 10)):
        for pol in ("lru", "lfu", "belady", "min"):
            b, per = bytes_per_token(pol, R=R, A=A)
            base[(pol, R)] = b
            say(f"R {R:2d} A {A:2d} {pol:6s}: {b / 1e6:6.1f}  (" + ", ".join(f"{k} {v / 1e6:.1f}" for k, v in per.items()) + ")"
                + ("" if pol == "lru" else f"  {100 * (b / base[('lru', R)] - 1):+.1f} % vs lru"))
    lfu16, min16 = base[("lfu", 16)], base[("min", 16)]
    say(f"MIN (any policy's floor) vs the engine's LFU at 16-row checks: {100 * (min16 / lfu16 - 1):+.1f} %")
    # RAM worth of the policy headroom: LFU pool scale f (C_m and C_l x f) that reads as little as MIN at 147 / 46
    for f in (1.1, 1.2, 1.3, 1.4, 1.6):
        b, _ = bytes_per_token("lfu", Cm=int(147 * f), Cl=int(46 * f))
        say(f"lfu with pools x{f:.1f} (C_m {int(147 * f)}, C_l {int(46 * f)}, +{(f - 1) * 16.96:.1f} GB): {b / 1e6:6.1f}")
        if b <= min16:
            break
    say(f"({time.time() - t0:.0f} s)")
    open(os.path.expanduser("~/MoonShots/data/wall24.txt"), "w").write("\n".join(lines) + "\n")
