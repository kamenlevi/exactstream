"""xfront_time.py (Breakthrough, Tue 17 Nov 2026 21:1x; MOE BLOCK front fold, STACK FORM; the ship rule of docs/MOEBLOCK.md
"FRONT FOLD, STACK FORM (mode tgl16ef)", pre-registered 1b71659): GPU interval of one command buffer (libmtlprof), min
over real blocks x rounds, at P = 160 / 320 (16 / 32 rows), layers 20 and 44:
  today = ooo1cq.gpu_grp_1cq + rowsum64(x) evaluated together, as tgl16e issues them; also each alone (info);
  fold  = xfront.gpu_front, variants v / r / s (one dispatch).
Rule: ship the faster bitwise variant iff fold < today at both P on L20 (min). Arms interleaved, order flipped per round.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xfront_time.py [rounds]   (quiet lock, < 1 GB)
Writes data/xrw/xfront_time.txt and .json."""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                        # noqa: E402

MP.load(2)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                          # noqa: E402
import xfront_check as FC                                   # noqa: E402

mx, OQ, XF, rowsum64 = FC.mx, FC.OQ, FC.XF, FC.rowsum64
OUT = os.path.join(HERE, "..", "..", "data", "xrw", "xfront_time")
L_ = []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


RAW = []


def gpu_us(f):
    """(GPU-busy = the union of this eval's command-buffer intervals in us, the largest single interval, their count).
    21:1x: the first pass used the span (first start to last end); with 2 command buffers per eval that counted the host
    gap between them (70-110 us at P = 160), so it was discarded."""
    mx.synchronize()
    c0 = MP.counts()["ncb"]
    mx.eval(f())
    mx.synchronize()
    cb = MP.cbs(c0, MP.counts()["ncb"])
    iv = sorted((s, e) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s)
    if not iv:
        return float("nan"), float("nan"), 0
    if len(RAW) < 12:
        RAW.append([(round(1e6 * (s - iv[0][0]), 1), round(1e6 * (e - iv[0][0]), 1)) for s, e in iv])
    u, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            u += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    u += ce - cs
    return 1e6 * u, 1e6 * max(e - s for s, e in iv), len(iv)


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    arms = ("today", "grp", "rs", "v", "r", "s")
    say(f"XFRONT TIME: GPU-busy us per launch (one eval; the union of its command-buffer intervals), min / p25 / median over 12 real blocks x "
        f"{rounds} rounds; today = gpu_grp_1cq + rowsum64(x) together")
    res = {}
    ncb = {}
    for Lm in (20, 44):
        bl = FC.real_blocks(Lm, 12)
        rr = np.random.default_rng(Lm)
        slot_of = rr.permutation(512).astype(np.int32)
        side_of = rr.permutation(512).astype(np.int32)
        for M in (16, 32):
            data = []
            for (f, b0, xf, idr) in bl:
                idm = idr[:M]
                d = dict(pm=mx.array(slot_of[idm]), pl=mx.array(side_of[idm[:, :2]]), ids=mx.array(idm),
                         x=mx.array(xf[:M]).astype(mx.bfloat16))
                mx.eval(*d.values())
                data.append(d)
            fn = {"today": lambda d: list(OQ.gpu_grp_1cq(d["pm"], d["pl"], d["ids"], "4433333333")[:3]) + [rowsum64(d["x"])],
                  "grp": lambda d: list(OQ.gpu_grp_1cq(d["pm"], d["pl"], d["ids"], "4433333333")[:3]),
                  "rs": lambda d: rowsum64(d["x"])}
            for v in ("v", "r", "s"):
                fn[v] = (lambda v_: lambda d: (lambda g: list(g[0][:3]) + [g[1]])(
                    XF.gpu_front(d["pm"], d["pl"], d["ids"], "4433333333", d["x"], v_)))(v)
            for d in data[:2]:                              # warm: build every kernel
                for a in arms:
                    gpu_us(lambda: fn[a](d))
            r = {a: [] for a in arms}
            n = {a: set() for a in arms}
            mxs = {a: [] for a in arms}
            for rd in range(rounds):
                for d in data:
                    for a in (arms if rd % 2 == 0 else arms[::-1]):
                        t, tm, c = gpu_us(lambda: fn[a](d))
                        r[a].append(t)
                        mxs[a].append(tm)
                        n[a].add(c)
            key = f"L{Lm}_P{10 * M}"
            res[key] = r
            ncb[key] = {a: sorted(n[a]) for a in arms}
            say(f"  {key} busy: " + "; ".join(f"{a} {np.min(r[a]):.1f} / {np.percentile(r[a], 25):.1f} / "
                                               f"{np.median(r[a]):.1f} (cbs {ncb[key][a]})" for a in arms))
            say(f"  {key} largest cb (info): " + "; ".join(f"{a} {np.min(mxs[a]):.1f} / {np.median(mxs[a]):.1f}"
                                                         for a in arms))
    best = {}
    for key in res:
        best[key] = {a: float(np.min(res[key][a])) for a in arms}
    l20 = [k for k in best if k.startswith("L20")]
    win = {v: all(best[k][v] < best[k]["today"] for k in l20) for v in ("v", "r", "s")}
    order = sorted(("v", "r", "s"), key=lambda v: (best["L20_P320"][v], best["L20_P160"][v]))
    pick = next((v for v in order if win[v]), None)
    say("SHIP RULE (L20, min; fold < today at P = 160 and 320): " +
        "; ".join(f"{v}: {best['L20_P160'][v]:.1f} / {best['L20_P320'][v]:.1f} vs today {best['L20_P160']['today']:.1f} / "
                  f"{best['L20_P320']['today']:.1f} -> {'pass' if win[v] else 'no'}" for v in ("v", "r", "s")))
    if pick:
        say(f"  -> SHIP variant {pick}; saving per call (min, today - fold): " +
            "; ".join(f"{k} {best[k]['today'] - best[k][pick]:.1f} us" for k in best) +
            "; x 48 layers per check [P]: " +
            "; ".join(f"{k} {48 * (best[k]['today'] - best[k][pick]) / 1000:.2f} ms" for k in best))
    else:
        say("  -> NO SHIP: no variant beats today at both P on L20")
    say(f"  raw cb intervals of the first evals (us from the first start): {RAW}")
    json.dump({"res": res, "ncb": ncb, "best": best, "pick": pick, "raw": RAW}, open(OUT + ".json", "w"))
    open(OUT + ".txt", "w").write("\n".join(L_) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
