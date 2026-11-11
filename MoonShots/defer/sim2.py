"""dh_w4 speed on BigMoE v3 at 24 GB, trace replay with the costs the first model (stall/defer_sim.py) left out:
LATE_LAT  the late reads form their own SSD batch (v3f loader: ensure(waited) returns, phase A goes out, then
          ensure(late)) and pay a second batch latency LAT (1) or stream right behind the waited ones (0, the old model)
TAIL      ms of GPU work between the late part's HC write and layer L+1's routing post being seen by the loader (late
          combine/HC write -> L+1 MLP HC read -> router -> post kernel -> commit -> loader poll): the late reads are due
          TAIL before the next routing, not at it
OVH       ms of phase-B GPU overhead per layer (spin B, phase-3 GLU dispatches, late combine, HC write) on every layer
Same SSD model as defer_sim.py (0.191 ms per batch + 2.16 MB at 13.1 GB/s per read), waited fraction 1 - FL.
usage: sim2.py [FL=0.72]"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/stall"))
LAT, BW, MB = 0.191, 13.1, 2.162688
RD = MB / BW
FL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.72


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
        yield n, gaps


def sim(n, gaps, fl, late_lat=0, tail=0.0, ovh=0.0, d1=False):
    """d1: the late part is due one layer later (before layer L+2's routing), so layer L+1's routing does not wait for it;
    the single loader thread still serves L's late batch before L+1's row (SSD order unchanged)"""
    t = s = 0.0
    prev = 0.0
    for j, (nl, g) in enumerate(zip(n, gaps)):
        last = j == len(n) - 1
        nw = int(round(nl * (1 - fl))) if not last else int(nl); nlate = int(nl) - nw
        sw = (max(s, t) + LAT + nw * RD) if nw > 0 else max(s, t)
        sl = sw + nlate * RD + (LAT if nlate > 0 and (nw == 0 or late_lat) else 0.0)
        s = sl if nlate > 0 else sw
        t = max(t, sw) if nw > 0 else t                      # phase A
        if fl > 0 and not last and not d1:
            t = max(t + g + ovh, (sl + tail + ovh) if nlate > 0 else 0.0)   # late part written TAIL before the next post
        elif fl > 0 and d1:
            t = max(t + g + ovh, prev)                       # layer L-1's late part due before this layer's next post
            prev = (sl + tail + ovh) if (nlate > 0 and not last) else 0.0
        else:
            t = t + g
    return max(t, prev)


rows = []
for f in sorted(glob.glob(os.path.expanduser("~/MoonShots/data/stall/anatomy_*_tier_*trace.json"))):
    r = json.load(open(f))
    tag = os.path.basename(f).split("_")[1]
    P = list(passes(r))
    ver = np.mean([p["t_verify"] for p in r["passes"]]) * 1e3
    base = np.mean([sim(n, g, 0.0) for n, g in P])
    print(f"{tag}: measured verify {ver:.1f} ms; model today {base:.1f} ms")
    for ll, tail, ovh in ((0, 0, 0), (1, 0, 0), (0, 0.12, 0), (1, 0.12, 0), (0, 0.12, 0.04), (1, 0.12, 0.04), (1, 0.2, 0.06)):
        d = np.mean([sim(n, g, FL, ll, tail, ovh) for n, g in P])
        d1 = np.mean([sim(n, g, FL, ll, tail, ovh, d1=True) for n, g in P])
        pred, pred1 = ver - (base - d), ver - (base - d1)
        print(f"   late batch latency {'own' if ll else 'none'}, tail {tail:.2f} ms, phase-B overhead {ovh:.2f} ms/layer: "
              f"dh {pred:.1f} ms ({ver / pred:.2f}x) | d1 {pred1:.1f} ms ({ver / pred1:.2f}x)")
