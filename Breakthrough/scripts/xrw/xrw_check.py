"""xrw_check.py (Breakthrough, Tue 17 Nov 2026 10:1x; EXPERT REWRITE, Leap's FAST report items a-d + g for a candidate):
the candidate (w4a8.tier_glu_nax6n_1cq_w4a8) vs the live COH one-call (coh1cq.tier_glu_nax6n_1cq_coh -> ooo1cq) on REAL
routing: the decision map's 12 code windows (xcap: each layer's MoE input x (fp16 -> bf16), top-10 ids + weights;
bt/tiers_1c keep / keep3), 32-row blocks of answer rows, layers 20 and 44, pools of BigMoE's loader (read-only, all 512
resident, native+p11+pad). Both kernels get the same per-pair slots (the km path: a dropped use reads its row's rank-0
id; nolsb = rank 0-1 uses not kept at 4 bits -> sidecar -2).
  a) install() with ON off == the original call, bitwise (y, rs_m, rs_l)
  b) 5 repeats of the candidate, bitwise identical
  c) the 32-row call == its two 16-row halves, bitwise (rows are independent: int32 sums are exact)
  d) per pair (row, rank): rel L2 error of y vs an fp32 dequant reference (numpy, from the pools via p11.unpack and the
     native maps; h kept in fp32), for the candidate and for ooo1cq; and candidate vs ooo1cq; max (worst pair named)
     and mean; plus elementwise max |diff| / max |ref| per pair
  g) the candidate's compile + first-call time and its extra buffers
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xrw_check.py [--blocks 4] [--tag T]  (share lock, MEM ~4 GB)
  -> data/xrw/xrw_check[_T].txt / .json"""
import argparse
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1cq as OQ
import coh1cq as CQ
import p11 as P11
import tierglu_nax6n as NN

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w4a8 as W                                            # noqa: E402
import xknob as XK                                          # noqa: E402

OUTD = os.path.join(HERE, "..", "..", "data", "xrw")
PLAN, KR, R4, NL, D = "4433333333", 10, 2, 96, 2560
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
MAP = os.path.expanduser("~/FableGod/decision_map/out")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
PROJ = ("gate_proj", "up_proj", "down_proj")
SHP = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
L_ = []


def say(s=""):
    print(s, flush=True)
    L_.append(s)


def bf16f(u16):
    return (np.asarray(u16, np.uint32) << 16).view(np.float32)


class Deq:
    """fp32 weights of one (layer, slot pair, tier) from the pools: W = s' q + b (tier 4: s' = s / 2, q = 2 m + l, b = the
    sidecar's bias; tier 3: s' = s, q = m, b = b3)"""

    def __init__(self, base, side):
        self.base, self.side = base, side

    def proj(self, p, sm, sl, tier):
        O, I = SHP[p]
        KG = I // 64
        w3 = np.array(self.base[p]["w3"][sm]).reshape(-1)[:O * KG * 6].astype(np.uint32)
        w3 = w3.reshape(O // 32, KG, 32, 6).transpose(0, 2, 1, 3).reshape(O, KG, 6)
        bits = np.unpackbits(w3.astype("<u4").view(np.uint8).reshape(O, KG, 24), axis=-1, bitorder="little")
        m = bits.reshape(O, KG, 64, 3) @ np.array([1, 2, 4], np.uint8)
        n = O * KG
        nm = NN.native_map(O, KG, 1)
        pw = P11.piece_words(n)

        def piece(arr, slot):
            v = P11.unpack(np.array(arr[slot]).reshape(1, -1)[:, :pw].astype(np.uint32), n)[0]
            rm = np.empty_like(v)
            rm[nm] = v
            return bf16f(rm).reshape(O, KG)
        s = piece(self.base[p]["s3"], sm)
        if tier == 4:
            l1 = np.array(self.side[p]["l1"][sl]).reshape(-1)[:O * KG * 2].astype(np.uint32)
            l1 = l1.reshape(O // 32, KG, 32, 2).transpose(0, 2, 1, 3).reshape(O, KG, 2)
            lb = np.unpackbits(l1.astype("<u4").view(np.uint8).reshape(O, KG, 8), axis=-1, bitorder="little")
            q = 2 * m.astype(np.float32) + lb.reshape(O, KG, 64).astype(np.float32)
            b = piece(self.side[p]["b"], sl)
            Wt = 0.5 * s[:, :, None] * q + b[:, :, None]
        else:
            b = piece(self.base[p]["b3"], sm)
            Wt = s[:, :, None] * m.astype(np.float32) + b[:, :, None]
        return Wt.reshape(O, I)


def blocks_for(Lm, nb):
    basej = json.load(open(os.path.join(MAP, "baseline.json")))
    code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
    out = []
    for f in code:
        z = np.load(os.path.join(MAP, "xcap", f))
        tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
        a = int(z["a"])
        n = z["ids"].shape[0]
        for b0 in range(0, n - 31, 32):
            rows = slice(b0, b0 + 32)
            ids = z["ids"][rows, Lm].astype(np.int64)
            k4 = tz["keep"][a + b0:a + b0 + 32, Lm].astype(bool)
            k3 = tz["keep3"][a + b0:a + b0 + 32, Lm].astype(bool)
            kept = k4 | k3
            idk = np.where(kept, ids, ids[:, :1])
            out.append((f, b0, z["x"][rows, Lm].astype(np.float32), idk, ~k4[:, :R4]))
            if len(out) >= nb:
                return out
    return out


def call(f, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs):
    pm = mx.array(sm_of[idk].astype(np.int32))
    pl = mx.array(np.where(nolsb, -2, sl_of[idk[:, :R4]]).astype(np.int32))
    y, rs_m, rs_l = f(x, base, side, pm, pl, mx.array(idk.astype(np.int32)), PLAN, region, offs, status)
    mx.eval(y, rs_m, rs_l)
    return y, rs_m, rs_l


def bits_eq(a, b):
    if a.shape != b.shape:
        return False
    if a.dtype == mx.bfloat16:
        return bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
    return bool(mx.all(a == b).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--layers", default="20,44")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cand", default="w4a8")
    a = ap.parse_args()
    XK.MODE[0] = a.cand
    layers = [int(v) for v in a.layers.split(",")]
    ext = BG.ext
    d_, ix = T.load_index(BARE, layout_ok=True)
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    mx.eval(region, status, offs)
    say(f"\nXRW CHECK {time.strftime('%a %d %b %H:%M:%S')}: candidate {a.cand} vs coh1cq (live COH one-call); MLX {mx.__version__}; "
        f"layers {layers}; {a.blocks} real 32-row blocks per layer (decision-map code windows)")
    XK.install()
    res = {"layers": {}}
    nxt = 0
    allok = True
    for Lm in layers:
        r = ix["layers"][str(Lm)]
        assert r.get("layout") == "native+p11+pad", r.get("layout")
        base, _ = T.make_pools(ext, nxt, 512, 512, SH4, d_, r)
        Ls, side, _ = T.make_sidecar(ext, 512, 512, SH4, d_, r)
        Lb = nxt
        nxt = Ls + 1
        allids = np.arange(512, dtype=np.int32)
        for j in range(0, 512, 64):
            ext.ensure(Lb, mx.array(allids[j:j + 64]))
            ext.ensure(Ls, mx.array(allids[j:j + 64]))
        sm_of = np.array(ext.lookup(Lb, allids.tolist()), np.int32)
        sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
        assert (sm_of >= 0).all() and (sl_of >= 0).all()
        dq = Deq(base, side)
        BL = blocks_for(Lm, a.blocks)
        st = dict(rel_c=[], rel_o=[], rel_co=[], elm_c=[], elm_o=[], worst=None, a=True, b=True, c=True, beq=0, bn=0)
        for bi, (f, b0, xf, idk, nolsb) in enumerate(BL):
            x = mx.array(xf).astype(mx.bfloat16)
            mx.eval(x)
            XK.ON[0] = False
            yo, rmo, rlo = call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs)
            XK.uninstall()
            yo2, rmo2, rlo2 = call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs)
            XK.install()
            st["a"] &= bits_eq(yo, yo2) and bits_eq(rmo, rmo2) and bits_eq(rlo, rlo2)
            XK.ON[0] = True
            t0 = time.time()
            yc, rmc, rlc = call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs)
            if bi == 0 and Lm == layers[0]:
                res["first_call_s"] = time.time() - t0
            st["b"] &= all(bits_eq(yc, call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region,
                                             status, offs)[0]) for _ in range(5))
            st["b"] &= bits_eq(rmc, rmo) and bits_eq(rlc, rlo)       # the same resolved slots as the original
            ya = call(CQ.tier_glu_nax6n_1cq_coh, x[:16], base, side, sm_of, sl_of, idk[:16], nolsb[:16], region, status,
                      offs)[0]
            yb = call(CQ.tier_glu_nax6n_1cq_coh, x[16:], base, side, sm_of, sl_of, idk[16:], nolsb[16:], region, status,
                      offs)[0]
            st["c"] &= bits_eq(yc, mx.concatenate([ya, yb], 0))
            XK.ON[0] = False
            eq = np.array(mx.all(yc.view(mx.uint16) == yo.view(mx.uint16), axis=-1)).reshape(-1)
            st["beq"] += int(eq.sum())
            st["bn"] += int(eq.size)
            # d) the fp32 reference per pair
            Yc = np.array(yc.astype(mx.float32)).reshape(32, KR, D)
            Yo = np.array(yo.astype(mx.float32)).reshape(32, KR, D)
            xb = np.array(x.astype(mx.float32))
            cache = {}
            for m in range(32):
                for rk in range(KR):
                    e = int(idk[m, rk])
                    tier = 4 if (rk < R4 and not nolsb[m, rk]) else 3
                    key = (e, tier)
                    if key not in cache:
                        cache[key] = tuple(dq.proj(p, int(sm_of[e]), int(sl_of[e]), tier) for p in PROJ)
                    Wg, Wu, Wd = cache[key]
                    g_ = Wg @ xb[m]
                    hh = (g_ / (1.0 + np.exp(-g_))) * (Wu @ xb[m])
                    yr = Wd @ hh
                    nr = np.linalg.norm(yr) + 1e-30
                    rc = float(np.linalg.norm(Yc[m, rk] - yr) / nr)
                    ro = float(np.linalg.norm(Yo[m, rk] - yr) / nr)
                    st["rel_c"].append(rc)
                    st["rel_o"].append(ro)
                    st["rel_co"].append(float(np.linalg.norm(Yc[m, rk] - Yo[m, rk]) / (np.linalg.norm(Yo[m, rk]) + 1e-30)))
                    mr = float(np.abs(yr).max()) + 1e-30
                    st["elm_c"].append(float(np.abs(Yc[m, rk] - yr).max() / mr))
                    st["elm_o"].append(float(np.abs(Yo[m, rk] - yr).max() / mr))
                    if st["worst"] is None or rc > st["worst"][0]:
                        st["worst"] = (rc, ro, f, b0 + m, rk, e, tier)
            del cache
        allok &= st["a"] and st["b"] and st["c"]
        q = {k: (float(np.max(v)), float(np.mean(v))) for k, v in st.items() if k.startswith(("rel", "elm"))}
        res["layers"][Lm] = dict(q, worst=st["worst"], a=st["a"], b=st["b"], c=st["c"], pairs=len(st["rel_c"]),
                                 bitwise_eq=st["beq"], bitwise_n=st["bn"])
        w = st["worst"]
        say(f"  L{Lm}: {len(BL)} blocks, {len(st['rel_c'])} pairs. a) ON off == original bitwise: {st['a']}; b) 5 repeats "
            f"bitwise + same resolved slots: {st['b']}; c) 32 == two 16-row halves bitwise: {st['c']}")
        say(f"    d) rel L2 vs fp32 ref: candidate max {q['rel_c'][0]:.4f} mean {q['rel_c'][1]:.4f} | ooo1cq max "
            f"{q['rel_o'][0]:.4f} mean {q['rel_o'][1]:.4f} | candidate vs ooo1cq max {q['rel_co'][0]:.4f} mean "
            f"{q['rel_co'][1]:.4f}; elementwise max|diff|/max|ref|: candidate {q['elm_c'][0]:.4f} / ooo1cq {q['elm_o'][0]:.4f}")
        say(f"    pairs bitwise == ooo1cq: {st['beq']} of {st['bn']}")
        rc_ = np.array(st["rel_c"]); ro_ = np.array(st["rel_o"])
        say(f"    share of pairs with rel L2 vs fp32 above 2 / 5 / 10 %: candidate {100 * np.mean(rc_ > 0.02):.1f} / "
            f"{100 * np.mean(rc_ > 0.05):.1f} / {100 * np.mean(rc_ > 0.10):.2f} % | ooo1cq {100 * np.mean(ro_ > 0.02):.1f} / "
            f"{100 * np.mean(ro_ > 0.05):.1f} / {100 * np.mean(ro_ > 0.10):.2f} %; candidate p50 / p90 / p99 "
            f"{np.percentile(rc_, 50):.4f} / {np.percentile(rc_, 90):.4f} / {np.percentile(rc_, 99):.4f}")
        res["layers"][Lm]["share_gt"] = {t: float(np.mean(rc_ > t)) for t in (0.02, 0.05, 0.10)}
        say(f"    worst pair (candidate): rel {w[0]:.4f} (ooo1cq {w[1]:.4f}) at {w[2]} row {w[3]} rank {w[4]} expert {w[5]} tier {w[6]}")
    XK.uninstall()
    say(f"  g) first candidate call incl. compile: {res.get('first_call_s', float('nan')):.2f} s; extra buffers per call: "
        f"x int8 {32 * 2560 / 1e3:.0f} KB + h int8 {320 * 640 / 1e3:.0f} KB + scales / sums {4 * 2 * (32 * 40 + 320 * 10) / 1e3:.0f} KB")
    say(f"XRW CHECK items a-c: {'PASS' if allok else 'FAIL'}; d) reported above")
    sfx = f"_{a.tag}" if a.tag else ""
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, f"xrw_check{sfx}.txt"), "a").write("\n".join(L_) + "\n")
    json.dump(res, open(os.path.join(OUTD, f"xrw_check{sfx}.json"), "w"), indent=1, default=str)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
