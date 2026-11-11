"""Grouped expert kernel microbench (NewIdeas, the 350 road, 2026-10-22): wide verify passes should read each distinct
expert ONCE. One real 125B MoE layer (v3 pools: records3 MSB-3 parts of all 512 experts + LSB-1 sidecars, in RAM),
per-layer routed-expert time at S = 4 / 8 / 16 / 32 rows (top-10 each):
  3c       today's fused 3/4-bit GLU (tierglu.tier_glu, plan 4433333333 with the LSB sidecar for ranks 0-1, fused combine):
           one full weight read per (row, rank) pair, S x 10 reads
  3c_3     the same kernel on plan 3333333333 (MSB-3 only): the access-pattern baseline for the grouped arm
  mlx      MLX gather_qmm unsorted (BigMoE's 3b path for MSB-3 ranks): one gather_qmv per (row, rank)
  grouped  MLX gather_qmm with SORTED rhs indices (mlx_lm's sorted SwitchGLU): pairs sorted by expert, each distinct
           expert's weights read once per row tile; the argsort / inverse permutation are inside the timed call
  3c_grp   the SAME 3c kernels expert-major (wide/grouped3c.py: one threadgroup per output tile x distinct expert loops over its
           pairs; bit-identical to 3c), pair grouping on the GPU inside the timed call
  mr       the 3c arithmetic expert-major, each code group unpacked ONCE for up to MB_RM (8) rows (wide/mr3c.py); meant
           to be bit-identical to 3c (checked at every S)
  3c_distinct  the 3c kernel once per distinct expert (one row each): every expert read once at 3c's efficiency, a proxy
           for an expert-major 3c kernel (without the other rows' arithmetic)
Routing: MoEPrefetch's 125B decode traces (exact model, greedy, router top-20 per token in rank order); a pass = S
consecutive tokens. natural = each row's own top-10 (U = the union). budget = TrainLab's batch-aware set: the U experts
with the largest sum over rows of w(rank)^2 (w(rank) = exp(-0.25 rank): the traces keep ranks, not probabilities), each
row then uses its top-10 within the set (padded at weight 0 if fewer than 10 of its top-20 are in it).
Floor: distinct experts x 2,150,400 B (MSB-3) at the read bandwidth measured on the same pool.
Timing: N calls chained by a data dependence (y <- y + 0 * moe(y)), one eval, minus the same chain without the MoE.
usage: microbench.py [layer=20] [trace=code] [N=30]   (GPU, ~2 GB; quiet lock)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/MoonShots/defer/v3d"))      # tierglu.py: the 3c kernels BigMoE v3 runs
import tierglu as TG
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grouped3c as GC                                                    # the expert-major form of the same 3c kernels
import mr3c as MR                                                         # dequant-once multi-row form (bit-identical intent)
ARMS = os.environ.get("MB_ARMS", "3c,3c_3,mlx,grouped,3c_grp,mr,3c_distinct").split(",")
RM = int(os.environ.get("MB_RM", "8"))
RA_MR, RB_MR = int(os.environ.get("MB_RA", "2")), int(os.environ.get("MB_RB", "2"))
SS = [int(v) for v in os.environ.get("MB_S", "4,8,16,32").split(",")]

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TRACE = sys.argv[2] if len(sys.argv) > 2 else "code"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
PROJ = TG.PROJ
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}      # (out, in)

# ---- pools from the records3 bytes (one layer, every expert resident: slot = expert id)
t0 = time.time()
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb = {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
del raw
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
MSB_B = sum(v.nbytes for v in msb["gate_proj"].values()) * 3 // E
print(f"layer {LAYER}: {E} experts loaded in {time.time() - t0:.1f}s; MSB-3 {MSB_B} B / expert, pools "
      f"{sum(v.nbytes for d in (msb, lsb) for pr in d for v in d[pr].values()) / 1e9:.2f} GB", flush=True)

# ---- routing sets from the trace
tr = np.load(os.path.expanduser(f"~/MoEPrefetch/data/traces125/{TRACE}_400_pf.npz"))[f"L{LAYER}"].astype(np.int32)   # [400, 20]
WR = np.exp(-0.25 * np.arange(20))


def passes(S, U=None):
    """per pass: ids [S, 10] (rank order), weights [S, 10] (normalised w(rank), 0 = padding), distinct experts"""
    out = []
    for t in range(0, tr.shape[0] - S + 1, S):
        top = tr[t:t + S]
        if U is None:
            ids = top[:, :10]; w = np.tile(WR[:10], (S, 1))
        else:
            sc = {}
            for r in range(S):
                for k, e in enumerate(top[r]):
                    sc[e] = sc.get(e, 0.0) + WR[k] ** 2
            A = set(sorted(sc, key=lambda e: -sc[e])[:U])
            ids = np.zeros((S, 10), np.int32); w = np.zeros((S, 10))
            Ab = sorted(A, key=lambda e: -sc[e])
            for r in range(S):
                use = [(k, e) for k, e in enumerate(top[r]) if e in A][:10]
                for j, (k, e) in enumerate(use):
                    ids[r, j], w[r, j] = e, WR[k]
                pad = [e for e in Ab if e not in set(ids[r, :len(use)])]
                for j in range(len(use), 10):
                    ids[r, j], w[r, j] = pad[j - len(use)], 0.0                     # padding: weight 0
        w = w / w.sum(-1, keepdims=True)
        out.append((ids, w, len(np.unique(ids))))                               # every id is read (padding too)
    return out


# ---- the arms: x [S, 2560] bf16 -> y [S, 2560]
def arm_3c(x, ids, w, plan):
    return TG.tier_glu(x, msb, lsb, ids, ids[:, :plan.count("4")] if "4" in plan else None, plan=plan, scores=w)


def _glu_gather(xe, idx, sorted_):
    g = mx.gather_qmm(xe, msb["gate_proj"]["w3"], msb["gate_proj"]["s3"], msb["gate_proj"]["b3"], rhs_indices=idx,
                      transpose=True, group_size=64, bits=3, sorted_indices=sorted_)
    u = mx.gather_qmm(xe, msb["up_proj"]["w3"], msb["up_proj"]["s3"], msb["up_proj"]["b3"], rhs_indices=idx,
                      transpose=True, group_size=64, bits=3, sorted_indices=sorted_)
    return mx.gather_qmm(nn.silu(g) * u, msb["down_proj"]["w3"], msb["down_proj"]["s3"], msb["down_proj"]["b3"],
                         rhs_indices=idx, transpose=True, group_size=64, bits=3, sorted_indices=sorted_)


def arm_mlx(x, ids, w):
    y = _glu_gather(mx.expand_dims(x, (-2, -3)), ids, False)                      # [S, 10, 1, D]
    return (y.squeeze(-2) * w[..., None]).sum(axis=-2)


def arm_grouped(x, ids, w):
    S, K = ids.shape
    flat = ids.reshape(-1)
    order = mx.argsort(flat)
    inv = mx.argsort(order)
    y = _glu_gather(mx.expand_dims(x, -2)[order // K], flat[order], True)        # [S*K, 1, D], rows grouped by expert
    return (y.squeeze(-2)[inv].reshape(S, K, -1) * w[..., None]).sum(axis=-2)


def arm_3c_grp(x, ids, w):
    """the 3c kernels expert-major (wide/grouped3c.py): == arm 3c bit for bit, each distinct expert read once; the pair
    grouping (csr kernel) is inside the timed call"""
    return GC.tier_glu_grouped(x, msb, lsb, ids, ids[:, :2], w, plan="4433333333")


def arm_mr(x, ids, w):
    """the 3c arithmetic expert-major with each code group unpacked ONCE for up to RM rows (wide/mr3c.py)"""
    return MR.tier_glu_mr(x, msb, lsb, ids, ids[:, :2], w, plan="4433333333", rm=RM, ra=RA_MR, rb=RB_MR)


def arm_3c_distinct(x, ids, w):
    """the 3c kernel once per DISTINCT expert (one row each, plan '3'): reading every expert once at 3c's efficiency, a proxy
    for an expert-major 3c kernel (the other rows' arithmetic is not included)"""
    u = mx.array(np.unique(np.array(ids)).astype(np.int32))
    xu = mx.broadcast_to(x[:1], (u.size, x.shape[-1]))
    y = TG.tier_glu(xu, msb, lsb, u[:, None], None, plan="3", scores=mx.ones((u.size, 1), mx.bfloat16))
    return mx.broadcast_to(y.sum(axis=0, keepdims=True), x.shape)


def timed(f, x):
    """ms per call: N calls chained through x, one eval, minus the bare chain"""
    def chain(g):
        y = x
        for _ in range(N):
            y = y + g(y) * 0
        return y
    for _ in range(2):
        mx.eval(chain(f))
    best = []
    for _ in range(3):
        a = time.perf_counter(); mx.eval(chain(f)); t = time.perf_counter() - a
        b = time.perf_counter(); mx.eval(chain(lambda y: y)); t0 = time.perf_counter() - b
        best.append((t - t0) / N * 1e3)
    return float(np.median(best))


# ---- read bandwidth on the same pool (a reduction over all MSB-3 weight words of gate_proj)
W = msb["gate_proj"]["w3"]
mx.eval(W.sum())
a = time.perf_counter()
for _ in range(10):
    mx.eval((W.view(mx.uint32).sum(axis=(1, 2))))
BW = W.nbytes * 10 / (time.perf_counter() - a) / 1e9
print(f"read bandwidth (reduction over {W.nbytes / 1e6:.0f} MB): {BW:.0f} GB/s", flush=True)

rng = np.random.default_rng(0)
res = {"layer": LAYER, "trace": TRACE, "bw_GBs": BW, "msb_bytes": MSB_B, "rows": []}
check = None
for S in SS:
    for U in ([None, 16] if S == 8 else [None, 32] if S == 16 else [None, 48, 64] if S == 32 else [None]):
        P = passes(S, U)
        pick = [P[i] for i in rng.choice(len(P), min(6, len(P)), replace=False)]      # a few passes per cell
        x = mx.random.normal((S, 2560)).astype(mx.bfloat16) * 0.05
        row = {"S": S, "U": U or "natural", "distinct": float(np.mean([p[2] for p in P]))}
        for name, fn in (("3c", lambda x, i, w: arm_3c(x, i, w, "4433333333")), ("3c_3", lambda x, i, w: arm_3c(x, i, w, "3333333333")),
                         ("mlx", arm_mlx), ("grouped", arm_grouped), ("3c_grp", arm_3c_grp), ("mr", arm_mr),
                         ("3c_distinct", arm_3c_distinct)):
            if name not in ARMS:
                row[name] = float("nan"); continue
            ts = []
            for ids, w, _ in pick:
                ii, ww = mx.array(ids), mx.array(w, dtype=mx.bfloat16)
                ts.append(timed(lambda y, ii=ii, ww=ww, fn=fn: fn(y, ii, ww), x))
            row[name] = float(np.mean(ts))
        if "mr" in ARMS and U is None:                                           # bit-identity of mr vs 3c at every S
            ids, w, _ = pick[0]
            ii, ww = mx.array(ids), mx.array(w, dtype=mx.bfloat16)
            c4, cm = arm_3c(x, ii, ww, "4433333333"), arm_mr(x, ii, ww)
            ok = bool(mx.all(c4 == cm).item())
            d = float(mx.abs(c4.astype(mx.float32) - cm.astype(mx.float32)).max())
            print(f"check mr S={S}: == 3c bit for bit: {ok} (max |diff| {d:.3g})", flush=True)
            res.setdefault("mr_bitwise", {})[str(S)] = ok
        if check is None and "grouped" in ARMS:                                   # same math? (3c_3 vs mlx vs grouped)
            ids, w, _ = pick[0]
            ii, ww = mx.array(ids), mx.array(w, dtype=mx.bfloat16)
            a3, am, ag = (f(x, ii, ww).astype(mx.float32) for f in (lambda x, i, w: arm_3c(x, i, w, "3333333333"), arm_mlx, arm_grouped))
            check = (float(mx.abs(a3 - am).max()), float(mx.abs(a3 - ag).max()), float(mx.abs(a3).max()))
            c4, cg = arm_3c(x, ii, ww, "4433333333"), arm_3c_grp(x, ii, ww)
            bit = bool(mx.all(c4 == cg).item())
            print(f"check S={S}: max |3c_3 - mlx| {check[0]:.3g}, |3c_3 - grouped| {check[1]:.3g} (|y| max {check[2]:.3g}); "
                  f"3c_grp == 3c bit for bit: {bit}", flush=True)
            res["grp_bitwise"] = bit
        row["floor"] = row["distinct"] * MSB_B / BW / 1e6                         # ms
        row["floor_pairs"] = S * 10 * MSB_B / BW / 1e6
        res["rows"].append(row)
        print(f"S={S:2d} {str(row['U']):>7s}: distinct {row['distinct']:5.1f} | 3c {row['3c']:.3f} ms, 3c_3 {row['3c_3']:.3f}, "
              f"mlx {row['mlx']:.3f}, grouped {row['grouped']:.3f}, 3c_grp {row['3c_grp']:.3f}, mr {row['mr']:.3f}, 3c_distinct {row['3c_distinct']:.3f} | floor distinct {row['floor']:.3f} ms, floor S x 10 "
              f"{row['floor_pairs']:.3f} ms | 3c / mr {row['3c'] / row['mr']:.2f}", flush=True)
os.makedirs(os.path.expanduser("~/MoonShots/data/wide"), exist_ok=True)
json.dump(res, open(os.path.expanduser(f"~/MoonShots/data/wide/microbench_L{LAYER}_{TRACE}{os.environ.get('MB_TAG', '')}.json"), "w"), indent=1)
