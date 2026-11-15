"""L3 HC SLC STAGING microbenches (NewIdeas; PRE-REGISTERED Sun 15 Nov 16:3x, before any run; CHECK55_LEVERS.md lever 3).
Runs only with Leap's OK, on a quiet ticket after ~17:37 and before 22:30. One process, <= 4 GB (~1.3 GB of weights).
Leap's kernels are imported read-only: ~/Leap/scripts qmm16.qmm16c (REF2-T's call: ks 1, nt 1) and leapk.s4_gemm (R-09b
S4, the HC read's merged down + inject GEMM, MLX's NAX split-K in partitions of 4,096). Weights are random, at the 125B's
shapes (kern23 SHAPES; the HC read's merged GEMM is N 324 = 320 mix + 4 inject, K 10240, bf16, 6.64 MB).

(a) COLD / CACHED gap per qmm16c shape (16 rows):
    - per-call time over R distinct weight copies (cold: every call reads DRAM) vs one copy R times (cached);
    - shapes: o_proj+out_proj (N 2560, K 6144), in_proj_z (6144, 2560), in_proj_qkv (10240, 2560);
    - LINE (a): gap = 1 - cached / cold. A shape with gap < 15 % can't gain from staging, so it is dropped from lever 3.
(b) HC TOUCH on S4. Per iteration i, over a dependent chain of R iterations like layers:
    - COLD:    stub(i) -> S4(copy i)
    - TOUCHED: [stub(i) || touch(copy i)] -> S4(copy i)   (touch = a max-reduce of copy i, independent of the stub; S4's
               input is x * (1 + 0 * touch), bitwise x, so S4 starts after the touch)
    - WARM:    stub(i) -> S4(copy 0)  (the same weights every call: cached)
    - STUB:    stub(i) alone
    The stub is the latency-bound window after the experts: the shared expert at 16 rows (qmm16c gate|up N 1280, K 2560
    -> silu * up -> down N 2560, K 640) + 2 elementwise ops.
    - S4 per arm = (arm - STUB) / R. The TOUCHED figure includes any cost of the touch that is not hidden.
    - Bits: S4(copy i) is bitwise equal between COLD and TOUCHED on every i.
    - LINE (b): PASS if S4_touched <= S4_warm + 6.3 us (D3-0's 25 - 18.7: within 6.3 us of fully cached, robust to this
      harness's per-call overhead) AND S4_touched <= S4_cold - 0.7 x (S4_cold - S4_warm) (>= 70 % of the cached gain
      recovered net of the touch) AND bitwise.
      -> ceiling -1.9 ms per 16-row engine check (96 reads x ~20 us); next, a slice A/B (A16, 12 rotated rounds).
      KILL otherwise.
    - CONTEXT: S4_cold / S4_warm vs D3-0's 38.9 / 18.7 us. Outside +-25 % means a different regime, so it is reported,
      not used to judge.
usage: BigMoE venv python -I -B l3_slc_bench.py > l3_slc_bench.txt"""
import os, sys, time, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx
from qmm16 import qmm16c
import leapk

BF = mx.bfloat16
OUT = os.path.expanduser("~/NewIdeas/research/week98/l3_slc_bench.json")
RUNS, WARM = 7, 2
mx.random.seed(1115)

def q4(N, K):
    """random 4-bit weights in qmm16c's layout: w [N, K/8] uint32, scales / biases [N, K/64] bf16"""
    w = mx.random.randint(0, 2 ** 31 - 1, (N, K // 8)).astype(mx.uint32)
    s = (mx.random.uniform(0.001, 0.01, (N, K // 64))).astype(BF)
    b = (mx.random.uniform(-0.05, 0.05, (N, K // 64))).astype(BF)
    mx.eval(w, s, b); return w, s, b

def med_wall(build):
    """median wall seconds of mx.eval(build()) over RUNS after WARM, with a synchronize around each"""
    ts = []
    for k in range(WARM + RUNS):
        outs = build(); mx.synchronize(); t0 = time.perf_counter(); mx.eval(outs); mx.synchronize()
        if k >= WARM: ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

res = {}
# ---------------------------------------------------------------- (a) cold / cached gap on qmm16c shapes
R_A = 30
x16 = mx.random.normal((16, 6144)).astype(BF); mx.eval(x16)
for name, N, K in (("o_proj+out_proj", 2560, 6144), ("in_proj_z", 6144, 2560), ("in_proj_qkv", 10240, 2560)):
    Ws = [q4(N, K) for _ in range(R_A)]
    xa = x16[:, :K]
    def cold():
        y = xa; outs = []
        for W in Ws:
            o = qmm16c(y if y.shape[1] == K else xa, *W)
            outs.append(o); y = xa * (1 + 0 * o[0, 0])           # a dependent chain, bitwise xa
        return outs
    def cached():
        y = xa; outs = []
        for _ in range(R_A):
            o = qmm16c(y, *Ws[0]); outs.append(o); y = xa * (1 + 0 * o[0, 0])
        return outs
    tc, tw = med_wall(cold) / R_A, med_wall(cached) / R_A
    mb = (N * K / 2 + 2 * 2 * N * K / 64) / 1e6
    gap = 1 - tw / tc
    res[name] = {"MB": round(mb, 2), "cold_us": round(tc * 1e6, 1), "cached_us": round(tw * 1e6, 1), "gap": round(gap, 3),
                 "cold_GBps": round(mb / 1e3 / tc, 1), "verdict": "staging can help" if gap >= 0.15 else "DROP (gap < 15 %)"}
    print(f"(a) {name:16s} {mb:6.2f} MB | cold {tc * 1e6:7.1f} us ({mb / 1e3 / tc:5.1f} GB/s) | cached {tw * 1e6:7.1f} us | "
          f"gap {100 * gap:5.1f} % -> {res[name]['verdict']}", flush=True)
    del Ws

# ---------------------------------------------------------------- (b) HC touch on S4
R_B, N4, K4 = 40, 324, 10240
W4 = [mx.random.normal((N4, K4)).astype(BF) * 0.02 for _ in range(R_B)]; mx.eval(W4)
a16 = mx.random.normal((1, 16, K4)).astype(BF); mx.eval(a16)
sgu, sdn = q4(1280, 2560), q4(2560, 640)
xs = mx.random.normal((16, 2560)).astype(BF); mx.eval(xs)

def stub(x):
    gu = qmm16c(x, *sgu); g, u = gu[:, :640], gu[:, 640:]
    h = (g * mx.sigmoid(g)) * u
    d = qmm16c(h, *sdn)
    return x + d * 0.5

def chain(arm):
    x, a = xs, a16; outs = []
    for i in range(R_B):
        x = stub(x)
        if arm == "STUB":
            outs.append(x); continue
        Wi = W4[0] if arm == "WARM" else W4[i]
        if arm == "TOUCHED":
            t = mx.max(Wi)                                      # touch: reads copy i, independent of the stub
            ai = a * (1 + 0 * t)                                # bitwise a; S4 waits for the touch
        else:
            ai = a
        ai = ai * (1 + 0 * x[0, 0])                             # S4 also waits for the stub (the real order)
        y = leapk.s4_gemm(ai, Wi)
        outs.append(y); x = x * (1 + 0 * y[0, 0, 0])            # the next layer waits for this read
    return outs

T = {arm: med_wall(lambda arm=arm: chain(arm)) for arm in ("STUB", "COLD", "TOUCHED", "WARM")}
s4 = {arm: (T[arm] - T["STUB"]) / R_B * 1e6 for arm in ("COLD", "TOUCHED", "WARM")}
# bits: COLD vs TOUCHED outputs on every copy
oc, ot = chain("COLD"), chain("TOUCHED"); mx.eval(oc, ot)
same = all(np.array_equal(np.asarray(c.astype(mx.float32)).view(np.uint32), np.asarray(t_.astype(mx.float32)).view(np.uint32)) for c, t_ in zip(oc, ot))
gain_ok = s4["TOUCHED"] <= s4["COLD"] - 0.7 * (s4["COLD"] - s4["WARM"])
verdict = "PASS" if (s4["TOUCHED"] <= s4["WARM"] + 6.3 and gain_ok and same) else "KILL"
ctx = all(abs(s4[k] / v - 1) <= 0.25 for k, v in (("COLD", 38.9), ("WARM", 18.7)))
res["S4"] = {"stub_us": round(T["STUB"] / R_B * 1e6, 1), **{f"s4_{k.lower()}_us": round(v, 1) for k, v in s4.items()},
             "bitwise_cold_vs_touched": same, "context_vs_D3_0": ctx, "verdict": verdict}
print(f"(b) stub {T['STUB'] / R_B * 1e6:.1f} us per iteration | S4 cold {s4['COLD']:.1f} | touched {s4['TOUCHED']:.1f} | "
      f"warm {s4['WARM']:.1f} us | bitwise cold == touched: {same} | D3-0 context {'ok' if ctx else 'DIFFERENT'} -> LINE (b): {verdict}",
      flush=True)
print(f"MLX peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
