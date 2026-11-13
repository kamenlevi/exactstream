"""K-60-T (Leap, Sun 2026-11-01 06:2x; NewIdeas 06:1x routing, REF2c bundle; shared GPU, rough): K-60 for the 45 GB default's
experts kernel. With REF2-T (tiers 4433333333) the routed experts run MoonShots' fused 3c kernel (tierglu.tier_glu: MSB-3
base pool for every rank, LSB-1 sidecar for ranks 0-1), not REF2's 4-bit qmmg, whose K-60 headroom was 150 vs ~210 GB/s.
Synthetic pools at the 125B's expert shapes (gate / up [640, 2560], down [2560, 640]; base 128 slots, sidecar 64), a
check's call: 16 rows x 10 ranks drawn from 100 base slots, ranks 0-1 from 48 sidecar slots. Per call: tier_glu ms
(median of 30) -> GB/s over the touched slots' bytes; the read ceiling over the same bytes (K-60's uint4 streaming
kernel). Uses the frozen tierglu copy in /Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn.
Usage: [BigMoE venv] k60t_tier_bw.py -> ~/Leap/data/k60t_tier_bw.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
from tierglu import tier_glu

D_ = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))
H, I, CM, CL, R, NR, USED, USED4 = 2560, 640, 128, 64, 16, 10, 100, 48
mx.random.seed(0)

def ru32(*shape):
    return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01):
    return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k):              # MSB-3 codes (3 bits / weight as 32-bit words), bf16 scales and biases per group of 64
    return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64)}
def lsb_part(o, k):              # LSB-1 bits (1 bit / weight) and the 4-bit tier's biases
    return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
x = (mx.random.normal((R, H)) * 0.5).astype(mx.bfloat16)
rng = np.random.default_rng(1)
ms_np = np.stack([rng.choice(USED, NR, replace=False) for _ in range(R)]).astype(np.int32)
ls_np = np.stack([rng.choice(USED4, 2, replace=False) for _ in range(R)]).astype(np.int32)
ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
um, ul = np.unique(ms_np), np.unique(ls_np)
per_m = sum(a.nbytes for p in msb.values() for a in p.values()) / CM
per_l = sum(a.nbytes for p in lsb.values() for a in p.values()) / CL
B = len(um) * per_m + len(ul) * per_l
say(f"K-60-T: MLX {mx.__version__}; tier plan 4433333333; base slot {per_m / 1e6:.2f} MB, sidecar slot {per_l / 1e6:.2f} MB; a "
    f"check's call: {R} rows x {NR} ranks, {len(um)} distinct base + {len(ul)} sidecar slots = {B / 1e6:.0f} MB per layer call")

def timeit(f, n=30):
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return med(ts)

t_a = timeit(lambda: tier_glu(x, msb, lsb, ms_, ls_, plan="4433333333"))
say(f"  a. tier_glu (3c): {1e3 * t_a:.3f} ms per layer call = {B / t_a / 1e9:.0f} GB/s (x 48 layers = {48e3 * t_a:.1f} ms per check)")

_RD = r"""
    uint tid = thread_position_in_grid.x, nth = threads_per_grid.x;
    const device uint4* s4 = reinterpret_cast<const device uint4*>(src);
    uint4 acc = 0;
    for (uint i = tid; i < n4[0]; i += nth) acc ^= s4[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = 1u;    // keep the loads live
"""
_rd = mx.fast.metal_kernel(name="k60t_read", input_names=["src", "n4"], output_names=["out"], source=_RD)
parts = [a[mx.array(um)].reshape(-1).view(mx.uint32) for p in msb.values() for a in p.values()]
parts += [a[mx.array(ul)].reshape(-1).view(mx.uint32) for p in lsb.values() for a in p.values()]
flat = mx.concatenate(parts)
flat = flat[: flat.size // 4 * 4]
mx.eval(flat)
n4 = mx.array([flat.size // 4], dtype=mx.uint32)
for thr in (256 * 1024, 1024 * 1024):
    t_c = timeit(lambda: _rd(inputs=[flat, n4], output_shapes=[(1,)], output_dtypes=[mx.uint32], grid=(thr, 1, 1),
                             threadgroup=(256, 1, 1), init_value=0)[0])
    say(f"  c. read ceiling ({thr // 1024}K threads, uint4): {1e3 * t_c:.3f} ms for {flat.nbytes / 1e6:.0f} MB = "
        f"{flat.nbytes / t_c / 1e9:.0f} GB/s")
say("K-60-T done")
open(os.path.join(D_, "k60t_tier_bw.txt"), "w").write("\n".join(lines) + "\n")
