"""Speed of half-layer deferral (dh_wW) on BigMoE v3 at 24 GB, replayed on the traced passes (anatomy.py STALL_TRACE=1):
per layer the measured demand reads (from the SSD busy time) split into waited (ranks < W, read before phase A) and late
(read after phase A, needed before the next layer's MLP read); the GPU's compute windows between layers are the measured
gaps. One SSD (FIFO, waited before late, 0.191 ms latency per batch + 2.16 MB at 13.1 GB/s per read). GPU timeline per layer:
routing -> wait for the waited batch -> compute window (this layer's MoE + next layer's attention) -> wait for this layer's
late batch before the next layer's MLP read -> next routing. Compares with the same model at f_late = 0 (today, all waited).
usage: defer_sim.py [f_late=0.72 (dh_w4 at C=128: 59.4 of 82.3 reads late)] [glob]"""
import glob, json, os, sys
import numpy as np

LAT, BW, MB = 0.191, 13.1, 2.162688
RD = MB / BW
FL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.72
files = sorted(glob.glob(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/stall/anatomy_*_trace.json")))


def passes(r):
    ev = np.array(r["ext_trace"])
    L, seq, kind, t0, t1 = ev[:, 0].astype(int), ev[:, 1].astype(int), ev[:, 2].astype(int), ev[:, 3] * 1e3, ev[:, 4] * 1e3
    for s in np.unique(seq):
        m = (seq == s) & (kind == 1)
        ml = L[m] % 49
        if len(np.unique(ml[ml < 48])) < 40:
            continue
        slow = set(zip(L[(seq == s) & (kind == 4)], np.round(t0[(seq == s) & (kind == 4)], 6)))
        lay = {}
        for li, a, b, raw in zip(ml, t0[m], t1[m], L[m]):
            if li >= 48:
                continue
            e = lay.setdefault(li, [a, b, 0.0])
            e[0], e[1] = min(e[0], a), max(e[1], b)
            if (raw, round(a, 6)) in slow:
                e[2] = max(e[2], b - a)
        ls = sorted(lay)
        busy = np.array([lay[i][2] for i in ls])
        n = np.maximum(0, np.round((busy - LAT) / RD)) * (busy > 0)
        gaps = np.array([max(0.0, lay[ls[j + 1]][0] - lay[ls[j]][1]) for j in range(len(ls) - 1)] + [0.0])
        yield n, gaps, lay[ls[-1]][1] - lay[ls[0]][0]


def sim(n, gaps, fl, d1=False):
    """d1=False: half a layer late (late reads due before the next layer's MLP read); d1=True: one full layer late (due
    before the routing of layer L+2, i.e. one more compute window of slack)"""
    t = s = 0.0
    prev_sl = 0.0
    for nl, g in zip(n, gaps):
        nw = int(round(nl * (1 - fl))); nlate = int(nl) - nw
        sw = (max(s, t) + LAT + nw * RD) if nw > 0 else max(s, t)
        sl = sw + nlate * RD + (LAT if (nw == 0 and nlate > 0) else 0.0)
        s = sl if nlate > 0 else sw
        t = max(t, sw) if nw > 0 else t                      # phase A: wait for the waited reads
        if d1:
            t = max(t + g, prev_sl)                          # layer L-1's late reads due before this window ends
            prev_sl = sl if nlate > 0 else 0.0
        else:
            t = max(t + g, sl if nlate > 0 else 0.0)        # compute window, then the late reads before the next MLP read
    return max(t, prev_sl)


for f in files:
    r = json.load(open(f))
    if not r.get("ext_trace"):
        continue
    tag = os.path.basename(f).replace("anatomy_", "").replace("_trace.json", "")
    base, dfr, d1r, span = [], [], [], []
    for n, gaps, sp in passes(r):
        base.append(sim(n, gaps, 0.0)); dfr.append(sim(n, gaps, FL)); d1r.append(sim(n, gaps, FL, d1=True)); span.append(sp)
    ver = np.mean([p["t_verify"] for p in r["passes"]]) * 1e3
    b, d = np.mean(base), np.mean(dfr)
    print(f"{tag}: measured verify {ver:.1f} ms (layer span {np.mean(span):.1f}); model today (all waited) {b:.1f} ms -> "
          f"half-layer deferral, {100 * FL:.0f} % of reads late: {d:.1f} ms; predicted verify {ver - (b - d):.1f} ms "
          f"({ver / (ver - (b - d)):.2f}x); one full layer late: {ver - (b - np.mean(d1r)):.1f} ms "
          f"({ver / (ver - (b - np.mean(d1r))):.2f}x)")
