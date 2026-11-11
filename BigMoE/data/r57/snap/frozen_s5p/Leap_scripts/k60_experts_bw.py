"""K-60 EXPBW (Leap, Fri 2026-10-30 19:40; FAST INTEGRATION: a proof-of-concept on a shared GPU, rough numbers): how far is
REF2's experts kernel (qmmg25.qmmg via experts_split, cfg_a (4, 1) / cfg_b (1, 1)) from this Mac's read ceiling at the
check's real shapes? R-50c: ~100 distinct experts x 2.76 MB per layer, ~13.4 GB per 16-row check at ~173 GB/s (77.7 ms);
the earlier floor used 280 GB/s (~48 ms).
Synthetic 4-bit g64 experts (gate / up [640, 2560], down [2560, 640]) in a pool of 128 slots; 16 rows x top-10 ids
drawn from 100 slots (as a check); timed per call (median of 30 after warm-up):
  a. experts_split (REF2's path: gate, up, SwiGLU, down) -> GB/s over the touched experts' bytes
  c. the read ceiling: a Metal kernel that streams the same touched experts' bytes once (uint4 loads, a tiny reduction)
Usage: [BigMoE venv] k60_experts_bw.py -> ~/Leap/data/k60_experts_bw.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from qmmg25 import qmmg
import ref2

D_ = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))
H, I, POOL, USED, R, KTOP = 2560, 640, 128, 100, 16, 10
mx.random.seed(0)

def qpool(n, k):
    w = (mx.random.normal((POOL, n, k)) * 0.02).astype(mx.bfloat16)
    q = [mx.quantize(w[e], group_size=64, bits=4) for e in range(POOL)]
    return tuple(mx.stack([t[i] for t in q]) for i in range(3))
gate, up, down = qpool(I, H), qpool(I, H), qpool(H, I)
mx.eval(gate, up, down)
nbytes = lambda t: sum(a.nbytes for a in t)
per_exp = (nbytes(gate) + nbytes(up) + nbytes(down)) / POOL
x = (mx.random.normal((R, H)) * 0.5).astype(mx.bfloat16)
rng = np.random.default_rng(1)
ids = np.stack([rng.choice(USED, KTOP, replace=False) for _ in range(R)]).astype(np.int32)
touched = len(np.unique(ids))
B = touched * per_exp
say(f"K-60: MLX {mx.__version__}; pool {POOL} experts x {per_exp / 1e6:.2f} MB; a check's call: {R} rows x top-{KTOP}, "
    f"{touched} distinct experts = {B / 1e6:.0f} MB per layer call")

def timeit(f, n=30):
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return med(ts)

ids_m = mx.array(ids)
t_a = timeit(lambda: ref2.experts_split(x, gate, up, down, ids_m, ref2.CFG["experts_cfg_a"], ref2.CFG["experts_cfg_b"]))
say(f"  a. experts_split (REF2's path): {1e3 * t_a:.3f} ms per layer call = {B / t_a / 1e9:.0f} GB/s "
    f"(x 48 layers = {48e3 * t_a:.1f} ms per check)")

# the ceiling: stream the touched experts' bytes once (the slots REF2 reads), one uint4 per thread per step
_RD = r"""
    uint tid = thread_position_in_grid.x, nth = threads_per_grid.x;
    const device uint4* s4 = reinterpret_cast<const device uint4*>(src);
    uint4 acc = 0;
    for (uint i = tid; i < n4[0]; i += nth) acc ^= s4[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = 1u;    // keep the loads live
"""
_rd = mx.fast.metal_kernel(name="k60_read", input_names=["src", "n4"], output_names=["out"], source=_RD)
sl = np.unique(ids)
flat = mx.concatenate([mx.concatenate([a[mx.array(sl)].reshape(-1).view(mx.uint32) for a in t]) for t in (gate, up, down)])
flat = flat[: flat.size // 4 * 4]
mx.eval(flat)
n4 = mx.array([flat.size // 4], dtype=mx.uint32)
for thr in (256 * 1024, 1024 * 1024):
    f = lambda: _rd(inputs=[flat, n4], output_shapes=[(1,)], output_dtypes=[mx.uint32], grid=(thr, 1, 1),
                    threadgroup=(256, 1, 1), init_value=0)[0]
    t_c = timeit(f)
    say(f"  c. read ceiling ({thr // 1024}K threads, uint4): {1e3 * t_c:.3f} ms for {flat.nbytes / 1e6:.0f} MB = "
        f"{flat.nbytes / t_c / 1e9:.0f} GB/s")
say(f"K-60 done")
open(os.path.join(D_, "k60_experts_bw.txt"), "w").write("\n".join(lines) + "\n")
