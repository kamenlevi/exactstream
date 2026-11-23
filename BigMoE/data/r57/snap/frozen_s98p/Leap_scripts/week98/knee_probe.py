"""KNEE (WEEK98 pillar 3; Leap, Sun 15 Nov 2026 01:4x; the orchestrator 01:2x): is there a GPU last-level-cache knee that
would let a drafter chain whose weights stay cache-resident step much faster than DRAM?
Pre-registered (before any run): the knee = the largest weight set W whose re-read bandwidth is >= 3x the ~290 GB/s DRAM
line (>= 870 GB/s).
  - A knee at W >= 4 MB -> the drafter owners get a "cache-resident fused chain" budget (MB per step) as a second route
    beside block drafting, with the measured per-step cost of a dependent chain at that W;
  - no knee (no W >= 4 MB at >= 870 GB/s) -> chains stay closed.
Two measurements per W in {2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256} MB:
  (1) REREAD: one dispatch, 2 threadgroups per GPU core (40 x 1024 threads), each re-reading its own 1/40 of W for P
      passes (uint4 loads, a dependent XOR so every pass is read). Effective GB/s = P x W / time. The access pattern of a
      chain step whose weights every core re-reads each step (an upper bound: no step barrier).
  (2) CHAIN: 32 dependent bf16 GEMVs x <- W x (W square, N = K, the same W every step, as a drafter step re-reads its
      weights) built as one MLX graph and evaluated once (no host round trip): ms per step and GB/s, i.e. MLX's
      per-dispatch form of a cache-resident chain.
usage: [venv] knee_probe.py (MEM < 2 GB)"""
import os, sys, time, json
import numpy as np
import mlx.core as mx


def say(s):
    print(s, flush=True)


SRC = r"""
    const device uint4* b4 = (const device uint4*)buf;
    const uint tg = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint n4 = N4 / NTG;
    const uint base = tg * n4;
    uint acc = tid;
    for (uint p = 0; p < PASSES; p++) {
        for (uint i = tid; i < n4; i += TPT) {
            const uint4 v = b4[base + i];
            acc = (acc ^ (v.x + v.y + v.z + v.w)) + p;
        }
    }
    out[tg * TPT + tid] = acc;
"""
NTG, TPT = 40, 1024
_K = {}


def kern():
    if "k" not in _K:
        _K["k"] = mx.fast.metal_kernel(name="leap_knee_reread", input_names=["buf"], output_names=["out"], source=SRC)
    return _K["k"]


def reread(buf, passes):
    n4 = buf.size // 4
    (o,) = kern()(inputs=[buf], template=[("N4", n4), ("NTG", NTG), ("TPT", TPT), ("PASSES", passes)],
                  grid=(NTG * TPT, 1, 1), threadgroup=(TPT, 1, 1), output_shapes=[(NTG * TPT,)],
                  output_dtypes=[mx.uint32])
    return o


def tmed(f, n=7, warm=2):
    for _ in range(warm):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


try:
    _arch = mx.metal.device_info().get("architecture", "?")
except Exception:
    _arch = "?"
say(f"KNEE {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; device {_arch}")
SIZES = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256)
res = []
for W in SIZES:
    nbytes = W * 2 ** 20
    n_u32 = nbytes // 4 // (NTG * 4) * (NTG * 4)
    buf = mx.random.randint(0, 2 ** 31, (n_u32,)).astype(mx.uint32)
    mx.eval(buf)
    passes = max(2, int(3 * 2 ** 30 / (n_u32 * 4)))         # ~3 GiB of reads per measurement
    t1 = tmed(lambda: reread(buf, passes))
    bw1 = passes * n_u32 * 4 / t1 / 1e9
    N = max(64, int(round(np.sqrt(nbytes / 2) / 64)) * 64)
    Wm = (mx.random.normal((N, N)) * (1.0 / np.sqrt(N))).astype(mx.bfloat16)
    x0 = mx.random.normal((1, N)).astype(mx.bfloat16)
    mx.eval(Wm, x0)

    def chain():
        x = x0
        for _ in range(32):
            x = x @ Wm.T
        return x
    t2 = tmed(chain)
    ms_step = 1e3 * t2 / 32
    bw2 = Wm.nbytes / (t2 / 32) / 1e9
    res.append(dict(W_MB=W, reread_GBs=bw1, passes=passes, chain_N=N, chain_W_MB=Wm.nbytes / 2 ** 20, chain_ms_step=ms_step,
                    chain_GBs=bw2))
    say(f"W {W:4d} MB | reread {bw1:7.1f} GB/s ({passes} passes, {1e3 * t1:.1f} ms) | chain of 32 GEMVs at N {N} "
        f"({Wm.nbytes / 2 ** 20:.1f} MB): {ms_step:.4f} ms per step = {bw2:6.1f} GB/s")
    del buf, Wm

dram = float(np.median([r["reread_GBs"] for r in res if r["W_MB"] >= 128]))
knee = [r for r in res if r["reread_GBs"] >= 3 * 290 and r["W_MB"] >= 4]
say("")
say(f"DRAM line (re-read at W >= 128 MB): {dram:.0f} GB/s")
if knee:
    k = max(knee, key=lambda r: r["W_MB"])
    say(f"KNEE verdict: re-read >= 870 GB/s up to W {k['W_MB']} MB ({k['reread_GBs']:.0f} GB/s) -> give the drafter owners a "
        f"cache-resident fused-chain budget of <= {k['W_MB']} MB per step; MLX's per-dispatch chain at that size: "
        f"{k['chain_ms_step']:.4f} ms per step")
else:
    best = max(res, key=lambda r: r["reread_GBs"] if r["W_MB"] >= 4 else 0)
    say(f"KNEE verdict: no W >= 4 MB re-reads at >= 870 GB/s (best {best['reread_GBs']:.0f} GB/s at {best['W_MB']} MB) "
        f"-> chains stay closed")
json.dump(res, open(os.path.expanduser("~/Leap/data/week98/screens/knee_probe.json"), "w"), indent=1)
