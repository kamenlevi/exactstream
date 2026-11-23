"""D3C, 3c's zero-DRAM diagnostic (Leap, Mon 2026-11-02 21:2x; orchestrator 21:2x: "a faster bitwise replica of 3c";
pre-registered in ~/Leap/TABLE_350.md 'D3C'): what bounds REF2-T's expert kernel at 16 rows? 3c (MoonShots'
planes_engine tierglu.tier_glu, the engine's kernel, imported read-only) on random 3-bit / 4-bit pools (k62e_diag's
setup: 640 x 2560 experts, plan 4433333333, M 16 rows x 10 ranks = 160 pairs), with the fused combine (scores):
  real  U distinct experts per layer (random top-10 per row from the first U slots; U 70 is a 16-row layer's union)
  one   every pair reads slot 0 (MSB and LSB): the same pairs, tiers and arithmetic, ~3 MB of codes that stay in cache,
        so no DRAM traffic and short load latencies
  all   160 distinct experts (every pair its own): the most DRAM
Kernel A alone (gate / up + SwiGLU) and A + B (down + combine), ms per call, interleaved rounds, median.
Lines: 'one' >= 30 % faster than 'real' at U 70 (A + B) -> 3c is latency / memory-bound: a bitwise prefetching replica is
worth building. < 10 % -> ALU / issue-bound: drop the replica road (only fewer ops per weight would help). Between:
partial, the replica's ceiling is that gap.
Usage: [BigMoE venv] d3c_diag.py -> ~/Leap/data/d3c_diag.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))
import tierglu as TG

OUT = os.path.expanduser("~/Leap/data/d3c_diag.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
H, I, CM, CL, NR, PLAN, M = 2560, 640, 192, 64, 10, "4433333333", 16
say(f"D3C 3c zero-DRAM diagnostic {time.strftime('%a %H:%M')} ({os.environ.get('D3C_COND', 'shared GPU, rough')}); "
    f"tierglu from {TG.__file__}")
mx.random.seed(0)
def ru32(*shape): return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01): return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k): return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64)}
def lsb_part(o, k): return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
rng = np.random.default_rng(1)
side_of = rng.permutation(CM) % CL
x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
sc = mx.softmax(mx.random.normal((M, NR)), axis=-1).astype(mx.bfloat16)
mx.eval(x, sc)

def routing(kind, U=70):
    if kind == "one":
        ms_np = np.zeros((M, NR), np.int32)
        ls_np = np.zeros((M, 2), np.int32)
    elif kind == "all":
        ms_np = rng.permutation(CM)[:M * NR].reshape(M, NR).astype(np.int32)
        ls_np = side_of[ms_np[:, :2]].astype(np.int32)
    else:
        ms_np = np.stack([rng.choice(U, NR, replace=False) for _ in range(M)]).astype(np.int32)
        ls_np = side_of[ms_np[:, :2]].astype(np.int32)
    return mx.array(ms_np), mx.array(ls_np), len(np.unique(ms_np))

def kernel_a(ms_, ls_):
    """tier_glu's first dispatch alone (gate / up + SwiGLU -> h), the same call tier_glu makes"""
    R = NR
    m4, m2, r4 = TG.plan_masks(PLAN)
    g, u = msb["gate_proj"], msb["up_proj"]
    bb = {p: msb[p]["b3"] for p in TG.PROJ}
    L = lsb
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    (h,) = TG._ka()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                            L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"],
                            ms_.reshape(-1), ls_.reshape(-1)],
                    template=tmpl + [("K_C", H), ("N_C", I), ("RA_C", 4), ("LD_C", 0)],
                    grid=(I // 4 * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, I)],
                    output_dtypes=[x.dtype])
    return h

ARMS = {}
for kind, U in (("real", 70), ("real", 24), ("one", 0), ("all", 0)):
    ms_, ls_, nd = routing(kind, U)
    tag = f"{kind}{U if kind == 'real' else ''} ({nd} distinct)"
    ARMS[tag + " A"] = lambda ms_=ms_, ls_=ls_: kernel_a(ms_, ls_)
    ARMS[tag + " A+B"] = lambda ms_=ms_, ls_=ls_: TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc)
T = {a: [] for a in ARMS}
order = list(ARMS)
for f in ARMS.values():
    for _ in range(3):
        mx.eval(*[f() for _ in range(8)])
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
for r in range(25):
    while os.path.exists(PAUSE):
        time.sleep(5)
    for a in order[r % len(order):] + order[:r % len(order)]:
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(*[ARMS[a]() for _ in range(8)])
        mx.synchronize()
        T[a].append(1e3 * (time.perf_counter() - t0) / 8)
med = {a: float(np.median(v)) for a, v in T.items()}
for a in order:
    say(f"  {a}: {med[a]:.3f} ms per call [p10 {np.percentile(T[a], 10):.3f}, p90 {np.percentile(T[a], 90):.3f}]")
real = [a for a in order if a.startswith("real70")]
one = [a for a in order if a.startswith("one")]
for part in ("A", "A+B"):
    r_ = med[next(a for a in real if a.endswith(" " + part))]
    o_ = med[next(a for a in one if a.endswith(" " + part))]
    say(f"  {part}: zero-DRAM 'one' is {100 * (1 - o_ / r_):.1f} % faster than 'real' U 70 ({o_:.3f} vs {r_:.3f} ms)")
r_ = med[next(a for a in real if a.endswith(" A+B"))]
o_ = med[next(a for a in one if a.endswith(" A+B"))]
gain = 1 - o_ / r_
v = ("latency / memory-bound: build the bitwise prefetching replica" if gain >= 0.30 else
     "ALU / issue-bound: drop the replica road" if gain < 0.10 else "partial: the replica's ceiling is this gap")
say(f"D3C verdict (A+B; >= 30 % build, < 10 % drop): {100 * gain:.1f} % -> {v}")
