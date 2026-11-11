"""3a / 3b / 3c speed at decode and verify shapes (rows M = 1..8, top-10, 125B expert shapes), measured the way a forward
pass runs it (glu_kernels_bench.py style): 48 MoE layers per command buffer over NLP distinct synthetic layer pools
(far beyond the SLC), each layer's input = the previous layer's combined output. ms per layer:
  stock4   all 10 ranks 4-bit, SwitchGLU's chain (gather_qmm up, gate, swiglu, down) + combine: today's decode
  all3     all 10 ranks through gather_qmm(bits=3) on the MSB pool (the 3a kernels alone)
  3b       BigMoE's tier_call (bigmoe/tier.py: gathers + rebuild4 + gather_qmm bits=4 for ranks 1-2, bits=3 for 3-10)
  3b-rs    3b with planes_engine/rebuild.py (one rebuild dispatch for the 3 projections of all 4-bit pairs)
  3c       planes_engine/tierglu.py (two dispatches)
GB/s = unique expert bytes the layer reads (4-bit 2.765 MB; MSB part 2.150 MB; + LSB part 0.768 MB for 4-bit ranks).
usage: bench_tierglu.py [plan]   (GPU ~3 GB; quiet lock)"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU
import bigmoe.tier as T
from tierglu import tier_glu, PROJ
from rebuild import rebuild_slots

PLAN = sys.argv[1] if len(sys.argv) > 1 else "4433333333"
K, N, G = 2560, 640, 64
C_M, C_L, C_4, NLP, LAYERS, REPS = 96, 48, 96, 6, 48, 5
SH = {"gate_proj": (N, K), "up_proj": (N, K), "down_proj": (K, N)}
act = SwiGLU()
rng = np.random.default_rng(0)


def rbits(shape):
    return mx.random.randint(0, 2 ** 31, shape, dtype=mx.int32).astype(mx.uint32) * 2 + mx.random.randint(0, 2, shape, dtype=mx.int32).astype(mx.uint32)


def rscale(shape, s):
    return (mx.random.normal(shape) * s).astype(mx.bfloat16)


POOLS = []
for _ in range(NLP):
    msb = {p: {"w3": rbits((C_M, o, k * 3 // 32)), "s3": rscale((C_M, o, k // G), 2e-3), "b3": rscale((C_M, o, k // G), 1e-2)}
           for p, (o, k) in SH.items()}
    lsb = {p: {"l1": rbits((C_L, o, k // 32)), "b": rscale((C_L, o, k // G), 1e-2)} for p, (o, k) in SH.items()}
    q4 = {p: (rbits((C_4, o, k // 8)), rscale((C_4, o, k // G), 1e-3), rscale((C_4, o, k // G), 1e-2)) for p, (o, k) in SH.items()}
    mx.eval(msb, lsb, q4)
    POOLS.append((msb, lsb, q4))
segs = T.segments(PLAN)
R4 = PLAN.count("4")


def combine(y, w):                                         # the block's (y * scores[..., None]).sum(axis=-2)
    return (y * w[..., None]).sum(axis=-2)


def stock4(x, li, sm, sl, w):
    q4 = POOLS[li][2]
    xe = mx.expand_dims(x, (-2, -3))
    pj = lambda h, p: mx.gather_qmm(h, *q4[p], rhs_indices=sm, transpose=True, group_size=G, bits=4)
    h = act(pj(xe, "up_proj"), pj(xe, "gate_proj"))
    return combine(pj(h, "down_proj").squeeze(-2), w)


def all3(x, li, sm, sl, w):
    msb = POOLS[li][0]
    xe = mx.expand_dims(x, (-2, -3))
    pj = lambda h, p: mx.gather_qmm(h, msb[p]["w3"], msb[p]["s3"], msb[p]["b3"], rhs_indices=sm, transpose=True,
                                    group_size=G, bits=3)
    h = act(pj(xe, "up_proj"), pj(xe, "gate_proj"))
    return combine(pj(h, "down_proj").squeeze(-2), w)


def tb(x, li, sm, sl, w, rs=False):
    msb, lsb, _ = POOLS[li]
    xe = mx.expand_dims(x, (-2, -3))
    outs = []
    for a, b, n in segs:
        if n == 4 and rs:
            sub = rebuild_slots(msb, lsb, sm[..., a:b].reshape(-1), sl[..., a:b].reshape(-1))
            idx = mx.arange(sm[..., a:b].size, dtype=mx.uint32).reshape(sm[..., a:b].shape)
            pj = lambda h, p: mx.gather_qmm(h, *sub[p], rhs_indices=idx, transpose=True, group_size=G, bits=4)
            h = act(pj(xe, "up_proj"), pj(xe, "gate_proj"))
            outs.append(pj(h, "down_proj").squeeze(-2))
            continue
        sl_ab = (sm[..., a:b], sl[..., a:b]) if n == 4 else sm[..., a:b]
        up = T._proj(xe, (msb, lsb), "up_proj", sl_ab, n)
        gt = T._proj(xe, (msb, lsb), "gate_proj", sl_ab, n)
        outs.append(T._proj(act(up, gt), (msb, lsb), "down_proj", sl_ab, n).squeeze(-2))
    return combine(mx.concatenate(outs, axis=-2), w)


def tc(x, li, sm, sl, w):
    msb, lsb, _ = POOLS[li]
    return combine(tier_glu(x, msb, lsb, sm, sl, plan=PLAN, ld=LD_A, **GEO), w)


LD_A = int(os.environ.get("TG_LD", "0"))                    # kernel A load variant
GEO = dict(ra=int(os.environ.get("TG_RA", "4")), rb=int(os.environ.get("TG_RB", "4")), nsg_b=int(os.environ.get("TG_NSG", "2")))
LY = int(os.environ.get("TG_LY", "8"))                     # MLX's combine order (combine_probe.py)


def tcc(x, li, sm, sl, w):                                  # 3c with the combine fused into dispatch B
    msb, lsb, _ = POOLS[li]
    return tier_glu(x, msb, lsb, sm, sl, plan=PLAN, scores=w, ly=LY, ld=LD_A, ra=GEO["ra"])


VARIANTS = {"stock4": stock4, "all3": all3, "3b": tb, "3b-rs": lambda *a: tb(*a, rs=True), "3c": tc, "3c+comb": tcc}
if os.environ.get("TG_ONLY"):
    VARIANTS = {k: VARIANTS[k] for k in os.environ["TG_ONLY"].split(",")}


def run(fn, M):
    routes = []
    for li in range(LAYERS):
        sm = mx.array(np.stack([rng.choice(C_M, 10, replace=False) for _ in range(M)]).astype(np.int32))
        sl = mx.array(np.stack([rng.choice(C_L, R4, replace=False) for _ in range(M)]).astype(np.int32))
        w = mx.softmax(mx.array(rng.standard_normal((M, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        routes.append((sm, sl, w))
    mx.eval(routes)
    um = np.mean([len(np.unique(np.array(r[0]))) for r in routes])     # distinct MSB slots per layer
    ul = np.mean([len(np.unique(np.array(r[1]))) for r in routes])     # distinct LSB slots per layer
    best = 1e9
    for rep in range(REPS + 1):
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        t = time.perf_counter()
        for li in range(LAYERS):
            sm, sl, w = routes[li]
            x = fn(x, li % NLP, sm, sl, w)
        mx.eval(x)
        dt = time.perf_counter() - t
        if rep:
            best = min(best, dt)
    return best / LAYERS * 1e3, um, ul


B4, BM, BL = 2.765e6, 2.150e6, 0.768e6
for M in (1, 2, 4, 8):
    parts = []
    for name, fn in VARIANTS.items():
        ms, um, ul = run(fn, M)
        byt = um * B4 if name == "stock4" else um * BM + (0 if name == "all3" else ul * BL)
        parts.append(f"{name} {ms:.3f} ms ({byt / ms / 1e6:.0f} GB/s)")
    print(f"M={M}: " + " | ".join(parts), flush=True)
