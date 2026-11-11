"""B10 v2 residual policy replay (NewIdeas Sun 25 Oct): where the SSD bytes go at 24 / 45 GB, and whether reading the
rank-0-1 residual by routing WEIGHT instead of by rank cuts them at equal (or better) quality proxy.
Model of B10 v2 (QUEUE B10-v2): every expert has a trellis base (B bytes, ~MSB-3 error) and a residual (R bytes) that
lifts it to the 4-bit grid; per layer two LRU pools (base slots Cb, residual slots Cr) share the byte budget of today's
24 GB split (147 MSB + 46 LSB = 353.3 MB / layer) or the 45 GB one (791 MB / layer, MoonShots budget24_b10v2).
A use (token, layer, rank k, normalized top-10 weight w) needs the base; it also needs the residual when the policy says:
  rank2      ranks 0-1 (B10 v2 as specified; reproduces MoonShots' 47.7 MB / token at B 1.29 R 0.9)
  rank1      rank 0 only
  w>=t       w >= t, any rank
  cum<m      ranks whose preceding cumulative weight is < m (rank 0 always)
Quality proxy (relative only; TrainLab's screen judges): Q = sum over uses of w^2 * e, e = 1 for base-only (taken as
MSB-3's error), e = 0.35 with the residual (E4 + residual code error); reported as Q / Q(v3 = ranks 0-1 at E4). The proxy
UNDER-states rank-0-1 damage (3B: proxy ~1.7x, TrainLab measured 2.25-3.97x), so only policies with proxy <= rank2's
are candidates.
Traces: Breakthrough's expA traces125 (4 sets x 800 tokens, top-20 ids + softmax gates per layer); tokens 0-399 warm the
pools, 400-799 measured. R rows per check, A accepted (the union per check is fetched once; residual if any row needs it).
usage: b10_resid.py [quick]   (CPU, one process; run with taskpolicy -b)"""
import glob, os, sys, time
from collections import OrderedDict
import numpy as np

NL = 48
T = "/Users/kamen/Breakthrough/data/expA/traces125/"
SETS = {}
for f in sorted(glob.glob(T + "*_800_g.npz")):
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.int32)          # tok x layer x rank
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1)[:, :, :10].astype(np.float64)
    w = g / g.sum(-1, keepdims=True)                                                        # normalized top-10 weights
    SETS[os.path.basename(f).split("_")[0]] = (ids, w)
BUD = {24: 147 * 2162688 + 46 * 770048, 45: 791e6}


def need_res(pol, w):
    """bool per use (tok x layer x rank) from the normalized weights (rank order)"""
    kind, x = pol
    if kind == "rank":
        return np.broadcast_to(np.arange(10) < x, w.shape)
    if kind == "w":
        return w >= x
    cum = np.cumsum(w, -1) - w
    return cum < x


def replay(ids, w, pol, B, R_b, Cb, Cr, R=1, A=1):
    n = ids.shape[0]
    pb = [OrderedDict() for _ in range(NL)]
    pr = [OrderedDict() for _ in range(NL)]
    nres = need_res(pol, w)
    st = dict(bytes=0.0, base=0, res=0, tok=0, q=0.0, qv3=0.0, resuse=0, uses=0)

    def touch(od, e, C):
        if e in od:
            od.move_to_end(e)
            return 0
        if C <= 0:
            return 1
        od[e] = True
        if len(od) > C:
            od.popitem(last=False)
        return 1
    pos = 0
    while pos + R <= n:
        meas = pos >= n // 2
        for l in range(NL):
            nb, nr = set(), set()
            for r in range(R):
                row, rr = ids[pos + r, l], nres[pos + r, l]
                for k in range(10):
                    e = int(row[k])
                    nb.add(e)
                    if rr[k]:
                        nr.add(e)
            b = 0.0
            for e in nb:
                m = touch(pb[l], e, Cb)
                b += m * B
                st["base"] += m * meas
            for e in nr:
                m = touch(pr[l], e, Cr)
                b += m * R_b
                st["res"] += m * meas
            if meas:
                st["bytes"] += b
        if meas:
            for t in range(pos, pos + A):
                ww = w[t] ** 2
                st["q"] += float((ww * np.where(nres[t], 0.35, 1.0)).sum())
                st["qv3"] += float((ww[:, :2] * 0.35).sum() + ww[:, 2:].sum())
                st["resuse"] += int(nres[t].sum())
                st["uses"] += nres[t].size
            st["tok"] += A
        pos += A
    return st


def replay_lfu(ids, w, pol, B, R_b, Cb, Cr, R=1, A=1, d=0.99):
    """as replay(), but victims are the lowest decayed use count (the engine's BIGMOE_EVICT_LFU=d, bigmoe_ext.cpp:270-315:
    per layer call the weight grows by 1/d and every needed expert gets +weight; victims exclude the call's needs)"""
    n = ids.shape[0]
    nres = need_res(pol, w)
    st = dict(bytes=0.0, base=0, res=0, tok=0, q=0.0, qv3=0.0, resuse=0, uses=0)
    E = 512
    pools = []
    for C in (Cb, Cr):
        pools.append([dict(C=C, sc=np.zeros(E), res=np.zeros(E, bool), n=0, wt=1.0) for _ in range(NL)])

    def call(p, need):
        """need: unique expert ids (np array); returns misses"""
        if need.size == 0:
            return 0
        p["wt"] /= d
        p["sc"][need] += p["wt"]
        miss = need[~p["res"][need]]
        k = miss.size
        if k == 0 or p["C"] <= 0:
            return k
        free = p["C"] - p["n"]
        if k > free:
            cand = p["res"].copy()
            cand[need] = False
            ci = np.flatnonzero(cand)
            v = ci[np.argpartition(p["sc"][ci], k - free - 1)[:k - free]] if k - free < ci.size else ci
            p["res"][v] = False
            p["n"] -= v.size
        take = miss[:min(k, p["C"] - p["n"])]
        p["res"][take] = True
        p["n"] += take.size
        return k
    pos = 0
    while pos + R <= n:
        meas = pos >= n // 2
        for l in range(NL):
            blk = ids[pos:pos + R, l]
            nb = np.unique(blk)
            nr = np.unique(blk[nres[pos:pos + R, l]])
            mb = call(pools[0][l], nb)
            mr = call(pools[1][l], nr)
            if meas:
                st["bytes"] += mb * B + mr * R_b
                st["base"] += mb
                st["res"] += mr
        if meas:
            for t in range(pos, pos + A):
                ww = w[t] ** 2
                st["q"] += float((ww * np.where(nres[t], 0.35, 1.0)).sum())
                st["qv3"] += float((ww[:, :2] * 0.35).sum() + ww[:, 2:].sum())
                st["resuse"] += int(nres[t].sum())
                st["uses"] += nres[t].size
            st["tok"] += A
        pos += A
    return st


def run(pol, B, R_b, Cb, Cr, R=1, A=1, lfu=0.0):
    tot = dict(bytes=0.0, base=0, res=0, tok=0, q=0.0, qv3=0.0, resuse=0, uses=0)
    per = {}
    for k, (ids, w) in SETS.items():
        s = replay_lfu(ids, w, pol, B, R_b, Cb, Cr, R, A, lfu) if lfu else replay(ids, w, pol, B, R_b, Cb, Cr, R, A)
        per[k] = s["bytes"] / s["tok"]
        for x in tot:
            tot[x] += s[x]
    tok = tot["tok"]
    # MB / token as the mean of the four sets (MoonShots' budget24 convention; the sets differ in length)
    return dict(mbt=float(np.mean(list(per.values()))) / 1e6, base_rt=tot["base"] / tok, res_rt=tot["res"] / tok,
                proxy=tot["q"] / tot["qv3"], res_share=tot["resuse"] / tot["uses"], per=per)


def best_split(pol, B, R_b, gb, R=1, A=1, crs=(0, 16, 32, 48, 64, 96, 128, 160, 208, 256), lfu=0.0):
    best = None
    for Cr in crs:
        Cb = int((BUD[gb] - Cr * R_b) // B)
        if Cb < 10 or Cb > 512:
            if Cb > 512:
                Cb = 512
            else:
                continue
        o = run(pol, B, R_b, Cb, Cr, R, A, lfu)
        o.update(Cb=Cb, Cr=Cr)
        if best is None or o["mbt"] < best["mbt"]:
            best = o
    return best


if __name__ == "__main__":
    t0 = time.time()
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    # reproduce MoonShots' B10 v2 row first (24 GB, B 1.29, R 0.9, base 229 + residual 64 -> 47.7 MB / token)
    o = run(("rank", 2), 1.29e6, 0.9e6, 229, 64)
    say(f"check vs MoonShots (24 GB, rank2, B 1.29 R 0.9, Cb 229 Cr 64): {o['mbt']:.1f} MB/token (MoonShots 47.7), "
        f"base reads {o['base_rt']:.1f} + residual reads {o['res_rt']:.1f} per token, proxy {o['proxy']:.2f}")
    pols = [("rank", 2), ("rank", 1), ("rank", 3), ("w", 0.25), ("w", 0.2), ("w", 0.15), ("w", 0.12), ("w", 0.1),
            ("cum", 0.2), ("cum", 0.3), ("cum", 0.4)]
    rates = [(1.29e6, 0.9e6, "TRL-2 2.06 b + R 0.9"), (1.29e6, 0.65e6, "TRL-2 + R k_r=1 0.65"),
             (1.49e6, 0.9e6, "gu2+d3 2.39 b + R 0.9")]
    if quick:
        pols, rates = pols[:4], rates[:1]
    for gb in (24, 45):
        for B, R_b, lab in rates:
            say(f"\n== {gb} GB, {lab} (B {B / 1e6:.2f} MB, R {R_b / 1e6:.2f} MB)")
            for pol in pols:
                o = best_split(pol, B, R_b, gb)
                say(f"  {pol[0]}{pol[1]:<5}: {o['mbt']:5.1f} MB/token (base {o['base_rt']:4.1f} + res {o['res_rt']:4.1f} reads/token; "
                    f"Cb {o['Cb']} Cr {o['Cr']}) | residual on {100 * o['res_share']:4.1f} % of uses, proxy {o['proxy']:.2f} | "
                    + " ".join(f"{k} {v / 1e6:.1f}" for k, v in o["per"].items()))
    say(f"({time.time() - t0:.0f} s)")
    od = os.path.expanduser("~/NewIdeas/data/b10r")
    os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "b10_resid" + ("_quick" if quick else "") + ".txt"), "w").write("\n".join(lines) + "\n")
