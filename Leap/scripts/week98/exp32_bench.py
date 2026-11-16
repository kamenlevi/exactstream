"""exp32_bench.py (Leap, Mon 16 Nov 2026 13:0x; CHECK25: the experts, ~20 ms of a 32-row lossy check at ~1.43x their byte
floor): the FAST-NAX expert one-call (ref2._ooo_sel -> ooo1cq / coh1cq, b43efc4's 2-D nolsb) on layer 20's converted pools
(BigMoE's loader, read-only; all 512 experts resident), on REAL 1c-tiered routing: the decision map's captured code
windows (xcap: the layer-20 MoE inputs x, top-10 ids and weights; bt/tiers_1c: keep = 4-bit tier, keep3 = MSB-3),
32-row blocks (and their first 16 rows for the 16-row cost). Per block, the km path (a dropped use reads its row's
rank-0 id at weight 0) and nolsb = rank 0-1 uses at MSB-3 or drop, as BigMoE's _t1c builds them.
Timed: NB blocks' calls independent in one graph per round, 7 rounds, median of 6 (us per call); bytes per call = per
distinct kept expert its MSB-3 plane (1.661 MB) + its LSB plane (0.474 MB) when a rank 0-1 use reads it -> GB/s and x the
280 GB/s floor. Variants: the one-call's kernel splits (ks for kernel A, nsg_b for kernel B), COH on / off.
usage: [BigMoE venv] exp32_bench.py   (MEM ~3 GB; quiet lock for clean us)"""
import json
import os
import sys
import time

os.environ["BIGMOE_MAXK"] = "320"
if "prof" in sys.argv[1:]:                               # libmtlprof before mlx: kernels per call
    sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
    import mtlprof as MP
    MP.load(2)
import numpy as np
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("~/Leap/third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx
import ref2
ref2.apply()
import bigmoe.gated as BG
import bigmoe.tier as T
import ooo1c as OC
import ooo1cq as OQ

OUT = os.path.expanduser("~/Leap/data/week98/check25/exp32_bench.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


ext = BG.ext
PLAN, K, R4, NL, D, LM = "4433333333", 10, 2, 96, 2560, 20
T.PLAN = PLAN
BARE = os.path.expanduser("~/BigMoE/models/bare")
SH4 = []
for o, i in ((640, 2560), (640, 2560), (2560, 640)):
    SH4 += [((o, i // 8), mx.uint32), ((o, i // 64), mx.bfloat16), ((o, i // 64), mx.bfloat16)]
d_, ix = T.load_index(BARE, layout_ok=True)
r = ix["layers"][str(LM)]
base, _ = T.make_pools(ext, 0, 512, 512, SH4, d_, r)
Ls, side, _ = T.make_sidecar(ext, 512, 512, SH4, d_, r)
allids = np.arange(512, dtype=np.int32)
for j in range(0, 512, 64):
    ext.ensure(0, mx.array(allids[j:j + 64])); ext.ensure(Ls, mx.array(allids[j:j + 64]))
sm_of = np.array(ext.lookup(0, allids.tolist()), np.int32)
sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
assert (sm_of >= 0).all() and (sl_of >= 0).all()
RS = 32 * (2 + BG.XLINES)
MSB_MB = sum(r["msb_piece_size"]) / 1e6                  # the records3 pieces the one-call reads per expert
LSB_MB = sum(r["lsb_piece_size"]) / 1e6


class Gate:
    def __init__(self):
        self.nl, self.nreg, self.seq = NL, 128, 9
        self.region = mx.zeros((self.nreg * BG.PAIRS * RS,), mx.int32)
        self.status = mx.zeros((4096,), mx.int32)
        mx.eval(self.region, self.status)
        self.REG = np.frombuffer(memoryview(self.region), dtype=np.int32)
        self.REG.setflags(write=True)


GATE = Gate()


CUR = [None]


def post(Lx, ids_, after=None):
    """sync-free: the block's slots were precomputed on the host (the loader's posts are async in the engine; a
    np.array(ids_) here would sync the GPU on every call, as the first q3 run did: ~2.6x the floor, a harness artifact)"""
    pb, ps, nb, ns = CUR[0]
    return (pb, nb) if Lx == 0 else (ps, ns)


def call(blk):
    x, ids, sc, nl, pb, ps, nb, ns = blk
    CUR[0] = (pb, ps, nb, ns)
    GATE.seq += 1
    t = {"base": base, "sideP": side, "segs": [(0, 2, 4), (2, 10, 3)], "side": NL // 2 + LM, "layout": 2,
         "nolsb": (nl, None)}
    return ref2._ooo_sel(x, ids, {"gate": GATE, "L": 0}, t, post, sc, None)


# real routing: the map's code windows, layer LM, 32-row blocks of the answer rows
MAP = os.path.expanduser("~/FableGod/decision_map/out")
basej = json.load(open(os.path.join(MAP, "baseline.json")))
code = [f for f in sorted(os.listdir(os.path.join(MAP, "xcap"))) if basej[f]["dom"] == "code"]
B32, B16, GB32, GB16, NK, RAW = [], [], [], [], [], []
for f in code:
    z = np.load(os.path.join(MAP, "xcap", f))
    tz = np.load(os.path.join(MAP, "bt", "tiers_1c", f))
    a = int(z["a"])
    n = z["ids"].shape[0]
    for b0 in range(0, n - 31, 32):
        rows = slice(b0, b0 + 32)
        ids = z["ids"][rows, LM].astype(np.int64)
        w = z["w"][rows, LM].astype(np.float32)
        k4 = tz["keep"][a + b0:a + b0 + 32, LM].astype(bool)
        k3 = tz["keep3"][a + b0:a + b0 + 32, LM].astype(bool)
        kept = k4 | k3
        idk = np.where(kept, ids, ids[:, :1])
        wk = np.where(kept, w, 0.0)
        nl = ~k4[:, :R4]                                     # rank 0-1 uses at MSB-3 or dropped: no LSB read
        for (m, dst, gdst) in ((32, B32, GB32), (16, B16, GB16)):
            kk = kept[:m]
            ex = np.unique(idk[:m][kk]) if kk.any() else np.unique(idk[:m, :1])
            lsb = np.unique(ids[:m, :R4][k4[:m, :R4]])
            nlm, idm = nl[:m], idk[:m]
            fl = np.flatnonzero(~nlm.reshape(-1))            # the 2-D dup pick (ref2 b43efc4): marked -> first unmarked id
            dpk = idm[:, :R4].reshape(-1)[fl[0]] if len(fl) else idm[0, 0]
            ids_s = np.where(nlm, dpk, idm[:, :R4])
            pb = mx.array(np.concatenate([[0], sm_of[idm.reshape(-1)]]).astype(np.int32))
            ps = mx.array(np.concatenate([[0], sl_of[ids_s.reshape(-1)]]).astype(np.int32))
            dst.append((mx.array(z["x"][rows, LM][:m].astype(np.float32)).astype(mx.bfloat16),
                        mx.array(idm.astype(np.int32)), mx.array(wk[:m]).astype(mx.bfloat16),
                        mx.array(nlm), pb, ps, int(idm.size), int(ids_s.size)))
            if m == 32:
                RAW.append((idm.copy(), kk.copy(), nlm.copy()))
            gdst.append((len(ex) * MSB_MB + len(lsb) * LSB_MB) / 1e3)   # the real pieces (11-16 13:4x; was 1.661 / 0.474)
            if m == 32:
                NK.append(len(ex))
mx.eval([t_ for blk in B32 + B16 for t_ in blk if isinstance(t_, mx.array)])
NB = min(24, len(B32))
say(f"\nEXP32 BENCH {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; layer {LM}'s pools (all 512 resident); "
    f"{len(code)} code windows -> {len(B32)} 32-row blocks (timing {NB}); kept experts per 32-row block mean "
    f"{np.mean(NK):.1f}; GB per call mean {np.mean(GB32[:NB]):.4f} (32 rows) / {np.mean(GB16[:NB]):.4f} (16)")
ref2.CFG.update(ooo_1c=True, tier_nax_ooo=True, nax_1cq=True, nax_native=True)


def run(blocks, rounds=7):
    ts = []
    for r_ in range(rounds):
        mx.synchronize()
        t0 = time.perf_counter()
        ys = [call(b) for b in blocks]
        mx.eval(ys)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(ts[1:])) / len(blocks)


if "prof" in sys.argv[1:]:                               # one call per block: dispatches, kernels, GPU busy vs wall
    ref2.CFG["nax_coh"] = True
    for b_ in B32[:2]:
        mx.eval(call(b_)); mx.synchronize()             # warm (pipelines built)
    for b_ in B32[2:5]:
        mx.synchronize()
        p0, c0, m0 = MP.psos(), MP.counts(), MP.now()
        y_ = call(b_)
        mx.eval(y_)
        mx.synchronize()
        m1, c1 = MP.now(), MP.counts()
        cb = MP.cbs(c0["ncb"], c1["ncb"])
        say(f"  one 32-row call: {c1['disp'] - c0['disp']} dispatches, {c1['ncb'] - c0['ncb']} command buffers, wall "
            f"{1e6 * (m1 - m0):.0f} us, GPU busy {1e6 * MP.busy(cb, m0, m1):.0f} us; kernels "
            + "; ".join(f"{k[:60]} {v}" for k, v in sorted(MP.pso_delta(p0, MP.psos()).items(), key=lambda kv: -kv[1])))
    open(OUT, "a").write("\n".join(L_) + "\n")
    sys.exit(0)
if "nlv" in sys.argv[1:]:                                # the one-call with three nolsb masks on the same blocks
    def variant(kind):
        out, gbs = [], []
        for (blk, (idm, kk, nlm)) in zip(B32[:NB], RAW[:NB]):
            nl2 = np.ones_like(nlm) if kind == "msb3" else (np.zeros_like(nlm) if kind == "lsb_all" else nlm)
            fl = np.flatnonzero(~nl2.reshape(-1))
            dpk = idm[:, :R4].reshape(-1)[fl[0]] if len(fl) else idm[0, 0]
            ids_s = np.where(nl2, dpk, idm[:, :R4])
            ps_ = mx.array(np.concatenate([[0], sl_of[ids_s.reshape(-1)]]).astype(np.int32))
            out.append((blk[0], blk[1], blk[2], mx.array(nl2), blk[4], ps_, blk[6], int(ids_s.size)))
            ex = np.unique(idm[kk]) if kk.any() else np.unique(idm[:, :1])
            lsb = np.unique(idm[:, :R4][~nl2])
            gbs.append((len(ex) * MSB_MB + len(lsb) * LSB_MB) / 1e3)
        mx.eval([t_ for b_ in out for t_ in b_ if isinstance(t_, mx.array)])
        return out, float(np.mean(gbs))
    ref2.CFG["nax_coh"] = False
    for kind in ("real", "msb3", "lsb_all", "real"):
        blks, g = variant(kind)
        u = run(blks)
        say(f"  nolsb {kind:7s}: {u:7.1f} us/call, {g:.4f} GB -> {g / (u * 1e-6):5.1f} GB/s (x{280 / (g / (u * 1e-6)):.2f} the 280 floor)")
    open(OUT, "a").write("\n".join(L_) + "\n")
    sys.exit(0)
f0, fcoh = OQ.tier_glu_nax6n_1cq, None
res = []
for coh in (True, False):
    ref2.CFG["nax_coh"] = coh
    for ks, nsg_b in ((4, 2), (2, 2), (1, 2), (4, 1), (4, 4)):
        def f_(*a, _ks=ks, _nb=nsg_b, **kw):
            kw.setdefault("ks", _ks)
            kw.setdefault("nsg_b", _nb)
            return f0(*a, **kw)
        OQ.tier_glu_nax6n_1cq = f_
        try:
            u32 = run(B32[:NB])
            u16 = run(B16[:NB]) if (ks, nsg_b) == (4, 2) else float("nan")
        except Exception as e:
            say(f"  coh {int(coh)} ks {ks} nsg_b {nsg_b}: FAILED {repr(e)[:160]}")
            continue
        finally:
            OQ.tier_glu_nax6n_1cq = f0
        g32 = float(np.mean(GB32[:NB]))
        bw = g32 / (u32 * 1e-6)
        say(f"  coh {int(coh)} ks {ks} nsg_b {nsg_b}: 32 rows {u32:7.1f} us/call ({bw:5.1f} GB/s, x{280 / bw:.2f} the 280 GB/s "
            f"floor)" + (f"; 16 rows {u16:7.1f} us/call ({float(np.mean(GB16[:NB])) / (u16 * 1e-6):5.1f} GB/s)"
                         if u16 == u16 else ""))
        res.append(dict(coh=coh, ks=ks, nsg_b=nsg_b, us32=u32, us16=u16, gb32=g32))
say(f"done; peak {mx.get_peak_memory() / 1e9:.1f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "a").write("\n".join(L_) + "\n")
json.dump(res, open(OUT.replace(".txt", ".json"), "w"), indent=1)
