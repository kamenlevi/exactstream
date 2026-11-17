"""xrw_bench.py (Breakthrough, Tue 17 Nov 2026 10:2x; EXPERT REWRITE, Leap's item f): the candidate vs the live COH
one-call, INTERLEAVED in one process (Leap 09:34): the same real blocks (exp32's: the decision map's code windows, layer
LM, 32-row blocks and their first 16 rows), alternating rounds, COH on. Per round and arm: NB calls independent in one
graph (exp32's way), us per call = wall / NB; plus libmtlprof's GPU busy over the round. Report: median over rounds of
us per call (32 and 16 rows), GB/s vs 280 on the real piece bytes, the candidate / original ratio per paired round.
Kernel-level only; the engine number is Leap's c25 A/B.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xrw_bench.py [--rounds 9] [--nb 24] [--layer 20] [--tag T]
  (share lock for a rough look, quiet ticket for the number; MEM ~3 GB) -> data/xrw/xrw_bench[_T].txt / .json"""
import argparse
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                        # before mlx

MP.load(2)
import numpy as np                                          # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xrw_check as XC                                      # noqa: E402  (paths, ref2.apply(), bigmoe, Leap's modules, w4a8)
import mlx.core as mx                                       # noqa: E402

BG, T, CQ, XK = XC.BG, XC.T, XC.CQ, XC.XK
OUTD = os.environ.get("XRW_OUTD") or os.path.join(HERE, "..", "..", "data", "xrw")
L_ = []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--nb", type=int, default=24)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cands", default="w4a8,tgl16")
    ap.add_argument("--sizes", default="32,16")
    ap.add_argument("--minmode", type=int, default=0, help="N > 0: each call alone in its command buffer, N interleaved "
                    "repeats per block and arm; per arm the median over blocks of the MIN GPU interval (steering on a "
                    "shared GPU only)")
    a = ap.parse_args()
    arms = ["orig"] + [c for c in a.cands.split(",") if c]
    SZ = tuple(int(v) for v in a.sizes.split(","))
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    r = ix["layers"][str(a.layer)]
    base, _ = T.make_pools(ext, 0, 512, 512, XC.SH4, d_, r)
    Ls, side, _ = T.make_sidecar(ext, 512, 512, XC.SH4, d_, r)
    allids = np.arange(512, dtype=np.int32)
    for j in range(0, 512, 64):
        ext.ensure(0, mx.array(allids[j:j + 64]))
        ext.ensure(Ls, mx.array(allids[j:j + 64]))
    sm_of = np.array(ext.lookup(0, allids.tolist()), np.int32)
    sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
    assert (sm_of >= 0).all() and (sl_of >= 0).all()
    MSB_MB = sum(r["msb_piece_size"]) / 1e6
    LSB_MB = sum(r["lsb_piece_size"]) / 1e6
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    BL = XC.blocks_for(a.layer, a.nb)
    blk = {32: [], 16: []}
    gb = {32: [], 16: []}
    for (f, b0, xf, idk, nolsb) in BL:
        for m in (32, 16):
            x = mx.array(xf[:m]).astype(mx.bfloat16)
            pm = mx.array(sm_of[idk[:m]].astype(np.int32))
            pl = mx.array(np.where(nolsb[:m], -2, sl_of[idk[:m, :XC.R4]]).astype(np.int32))
            ids = mx.array(idk[:m].astype(np.int32))
            blk[m].append((x, pm, pl, ids))
            ex = np.unique(idk[:m])
            lsb = np.unique(idk[:m, :XC.R4][~nolsb[:m]])
            gb[m].append((len(ex) * MSB_MB + len(lsb) * LSB_MB) / 1e3)
    mx.eval([t for m in blk for b in blk[m] for t in b] + [region, status, offs])
    nb = len(BL)

    def arm(nm):
        def f(b):
            XK.ON[0] = nm != "orig"
            if nm != "orig":
                XK.MODE[0] = nm
            return CQ.tier_glu_nax6n_1cq_coh(b[0], base, side, b[1], b[2], b[3], XC.PLAN, region, offs, status)[0]
        return f
    XK.install()
    t0 = time.time()
    for nm in arms:                                         # build + warm every arm
        for m in SZ:
            mx.eval([arm(nm)(b) for b in blk[m][:2]])
    mx.synchronize()
    tw = time.time() - t0
    if a.minmode > 0:
        mins = {f"{nm}{m}": [[] for _ in range(nb)] for nm in arms for m in SZ}
        for rp in range(a.minmode):
            for m in SZ:
                for bi, b in enumerate(blk[m]):
                    for nm in (arms if (rp + bi) % 2 == 0 else arms[::-1]):
                        f = arm(nm)
                        mx.synchronize()
                        c0 = MP.counts()["ncb"]
                        mx.eval(f(b))
                        mx.synchronize()
                        cb = MP.cbs(c0, MP.counts()["ncb"])
                        gi = [(s_, e_) for s_, e_, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e_ > s_]
                        if gi:
                            mins[f"{nm}{m}"][bi].append(1e6 * (max(e_ for _, e_ in gi) - min(s_ for s_, _ in gi)))
        XK.ON[0] = False
        XK.uninstall()
        say(f"\nXRW BENCH MINMODE {time.strftime('%a %d %b %H:%M:%S')} (STEERING ONLY, shared GPU): arms {arms}; layer "
            f"{a.layer}; {nb} blocks x {a.minmode} interleaved repeats; per arm the median over blocks of the min GPU "
            f"interval of one call (its command buffers' first start to last end)")
        for m in SZ:
            o = [min(v) for v in mins[f"orig{m}"]]
            for nm in arms:
                mm = [min(v) for v in mins[f"{nm}{m}"]]
                rr = [x / y for x, y in zip(mm, o)]
                say(f"  {m} rows {nm:6s}: {np.median(mm):7.1f} us (min-of-{a.minmode}, median over blocks); / orig per block: "
                    f"median {np.median(rr):.3f} [p10 {np.percentile(rr, 10):.3f}, p90 {np.percentile(rr, 90):.3f}]")
        sfx = f"_{a.tag}" if a.tag else ""
        os.makedirs(OUTD, exist_ok=True)
        open(os.path.join(OUTD, f"xrw_bench{sfx}.txt"), "a").write("\n".join(L_) + "\n")
        json.dump(mins, open(os.path.join(OUTD, f"xrw_bench{sfx}.json"), "w"))
        return
    res = {f"{nm}{m}": [] for nm in arms for m in SZ}
    gpu = {k: [] for k in res}
    for rd in range(a.rounds):
        order = arms[rd % len(arms):] + arms[:rd % len(arms)]   # rotate the arm order every round
        for m in SZ:
            for nm in order:
                f = arm(nm)
                mx.synchronize()
                c0 = MP.counts()["ncb"]
                m0 = MP.now()
                t1 = time.perf_counter()
                ys = [f(b) for b in blk[m]]
                mx.eval(ys)
                mx.synchronize()
                dt = time.perf_counter() - t1
                m1 = MP.now()
                cb = MP.cbs(c0, MP.counts()["ncb"])
                res[f"{nm}{m}"].append(1e6 * dt / nb)
                gpu[f"{nm}{m}"].append(1e6 * MP.busy(cb, m0, m1) / nb)
    XK.ON[0] = False
    XK.uninstall()
    med = {k: float(np.median(v)) for k, v in res.items()}
    gmed = {k: float(np.median(v)) for k, v in gpu.items()}
    say(f"\nXRW BENCH {time.strftime('%a %d %b %H:%M:%S')}: arms {arms} (orig = coh1cq, the live call), interleaved with "
        f"the order rotated per round; layer {a.layer}; {nb} real blocks; {a.rounds} rounds; MLX {mx.__version__}; "
        f"build + warm {tw:.1f} s")
    for m in SZ:
        g = float(np.mean(gb[m]))
        for nm in arms:
            u = med[f"{nm}{m}"]
            v = res[f"{nm}{m}"]
            say(f"  {m} rows {nm:6s}: {u:7.1f} us/call (GPU busy {gmed[f'{nm}{m}']:7.1f}); {g / (u * 1e-6):5.1f} GB/s "
                f"(x{280 / (g / (u * 1e-6)):.2f} the 280 floor); spread {100 * (max(v) - min(v)) / u:.0f} %")
        for nm in arms[1:]:
            rr = [c / o for c, o in zip(res[f"{nm}{m}"], res[f"orig{m}"])]
            say(f"  {m} rows {nm} / orig per round: median {np.median(rr):.3f} [min {min(rr):.3f}, max {max(rr):.3f}]")
    sfx = f"_{a.tag}" if a.tag else ""
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, f"xrw_bench{sfx}.txt"), "a").write("\n".join(L_) + "\n")
    json.dump(dict(res=res, gpu=gpu, gb={m: float(np.mean(gb[m])) for m in gb}), open(os.path.join(OUTD, f"xrw_bench{sfx}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
