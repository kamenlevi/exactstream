"""L3 HC SLC STAGING microbenches, v2 (NewIdeas; PRE-REGISTERED Sun 15 Nov 17:4x, before any run; CHECK55_LEVERS.md lever 3).
v2 replaces v1 (16:3x, never run) after Leap's notes (17:3x):
- reproduce D3-0 first, with its own method;
- MLX puts a barrier before any kernel that reads an earlier output, and everything after a barrier waits for
  everything before it. So the touch can hide behind ONE kernel per barrier interval, and S4 must depend on that
  kernel's output;
- the stub is a real latency-bound kernel;
- the eviction step includes o_proj / out_proj's weights.
Runs only with Leap's OK (given 17:3x), on a quiet ticket after MoonShots' TREE-E2E (~19:30) and before 22:30.
Same stream. One process, <= 4 GB (weights ~1.3 GB). Leap's kernels are imported read-only (~/Leap/scripts:
qmm16.qmm16c with REF2-T's ks 1, nt 1; leapk.s4_partials + the _ACCUM path via leapk.s4_gemm; D3-0's cached-probe source
edit, recomputed here, ~/Leap/scripts/hc_bw_probe.py). Weights are random at the 125B's shapes.

STEP 0, reproduce D3-0 (context gate): leapk.s4_partials over 48 distinct 6.6 MB copies (N 324, K 10240) vs D3-0's
  S4-cached kernel (every tile's B rows -> tile 0's: the bytes-free time), median of 15 rotated rounds.
  - It must land within +-25 % of 38.9 / 18.7 us. Otherwise the regime differs: report it and do not judge (b).
STEP 1 (a), cold / cached gap on qmm16c shapes:
  - 16 rows; v2.1 (after the 17:2x functional smoke, which showed per-eval overhead swamping a dependent chain at
    small R) uses D3-0's independent-call method: cold = R distinct copies, cached = one copy R times;
  - o_proj+out_proj (2560 x 6144), in_proj_z (6144 x 2560), in_proj_qkv (10240 x 2560).
  - LINE (a): gap = 1 - cached / cold < 15 % -> that shape can't gain from staging.
STEP 2, the window: WIN = a real 16-row latency-bound kernel, MLX's SDPA at the 125B's attention shape (q 24 heads x 256,
  k / v 2 heads x 256, 1,024 cached keys), in a dependent chain.
  - TOUCH = a read-all kernel over S4 copy i's 6.64 MB. It lists WIN's input as an input, so it lands in WIN's barrier
    interval.
  - LINE (window, Leap's +5 us): (chain[WIN || TOUCH] - chain[WIN]) / R <= 5 us.
STEP 3 (b), the full sequence per iteration (R 40, dependent like layers):
  - EVICT (qmm16c on a cold in_proj_qkv-size copy, 13 MB, the experts' stand-in)
    -> WIN [|| TOUCH in TOUCHED]
    -> OPROJ (qmm16c on a cold o_proj copy, 7.9 MB)
    -> S4 (copy i; its input depends on OPROJ)
    -> the next iteration depends on S4.
  - Arms:
    - NO_S4 (S4 left out);
    - COLD;
    - TOUCHED (touch beside WIN, before OPROJ);
    - LATE (touch beside a 16-row elementwise HC-write stand-in placed after OPROJ);
    - BYTES-FREE (D3-0's cached kernel in S4's place).
  - S4_eff(arm) = (arm - NO_S4) / R.
  - Bits: S4's outputs are bitwise equal in COLD, TOUCHED and LATE.
  - LINE (b): PASS if TOUCHED or LATE has S4_eff <= S4_eff(BYTES-FREE) + 6.3 us AND recovers >= 70 % of
    (S4_eff(COLD) - S4_eff(BYTES-FREE)), with bits equal
    -> next, a slice A/B (A16, 12 rotated rounds) at the arm's placement.
    KILL otherwise.
usage: BigMoE venv python -I -B l3_slc_bench.py [--smoke]   (writes l3_slc_bench.json / prints the read-out)"""
import os, sys, time, json
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import mlx.core as mx
from qmm16 import qmm16c
import leapk as LK

SMOKE = "--smoke" in sys.argv
BF = mx.bfloat16
OUT = os.path.expanduser(f"~/NewIdeas/research/week98/l3_slc_bench{'_smoke' if SMOKE else ''}.json")
RUNS, WARM = (1, 1) if SMOKE else (15, 2)
R = 3 if SMOKE else 40
mx.random.seed(1115)
res = {"smoke": SMOKE}

def q4(N, K):
    w = mx.random.randint(0, 2 ** 31 - 1, (N, K // 8)).astype(mx.uint32)
    s = mx.random.uniform(0.001, 0.01, (N, K // 64)).astype(BF)
    b = mx.random.uniform(-0.05, 0.05, (N, K // 64)).astype(BF)
    mx.eval(w, s, b); return w, s, b

def timed(arms):
    """median wall seconds per arm over RUNS rounds (arm order rotated per round), each mx.eval between synchronizes"""
    T = {k: [] for k in arms}
    keys = list(arms)
    for r in range(WARM + RUNS):
        order = keys if r % 2 == 0 else keys[::-1]
        for k in order:
            outs = arms[k](); mx.synchronize(); t0 = time.perf_counter(); mx.eval(outs); mx.synchronize()
            if r >= WARM: T[k].append(time.perf_counter() - t0)
    return {k: float(np.median(v)) for k, v in T.items()}

# ---------------------------------------------------------------- STEP 0: reproduce D3-0
N4, K4, NL = 324, 10240, (6 if SMOKE else 48)
W4 = [(mx.random.normal((N4, K4)) * 0.02).astype(BF) for _ in range(max(NL, R))]
x4 = (mx.random.normal((1, 16, K4)) * 0.5).astype(BF)
mx.eval(W4, x4)
SRC_C = LK._S4_SRC.replace("const device T* Bp = w + size_t(c_col) * GK + k0;", "const device T* Bp = w + k0;")
assert SRC_C != LK._S4_SRC, "D3-0's source edit no longer applies"
PS = 4096; NP, NT = (K4 + PS - 1) // PS, (N4 + 31) // 32
kc = mx.fast.metal_kernel(name="nis_s4_cached", input_names=["x", "w"], output_names=["cs"], source=SRC_C,
                          header=LK._nax_header() + "\n" + LK.consts(GN=N4, GK=K4, GPS=PS, GNP=NP, GNT=NT, GPF=4))
def s4c_partials(a, w):
    (cs,) = kc(inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
               output_shapes=[(NP, 16, N4)], output_dtypes=[mx.float32])
    return cs
T0 = timed({"S4": lambda: [LK.s4_partials(x4, w)[0] for w in W4[:NL]], "S4-cached": lambda: [s4c_partials(x4, w) for w in W4[:NL]]})
s4_us, s4c_us = T0["S4"] / NL * 1e6, T0["S4-cached"] / NL * 1e6
ctx = abs(s4_us / 38.9 - 1) <= 0.25 and abs(s4c_us / 18.7 - 1) <= 0.25
res["step0"] = {"S4_us": round(s4_us, 1), "S4_cached_us": round(s4c_us, 1), "context_ok": ctx}
print(f"STEP 0 (D3-0 reproduce): S4 {s4_us:.1f} us, S4-cached {s4c_us:.1f} us (D3-0 38.9 / 18.7) -> context {'OK' if ctx else 'DIFFERENT'}", flush=True)

# ---------------------------------------------------------------- STEP 1 (a): qmm16c cold / cached gap, dependent chain
x16 = mx.random.normal((16, 6144)).astype(BF); mx.eval(x16)
res["step1"] = {}
for name, N, K in (("o_proj+out_proj", 2560, 6144), ("in_proj_z", 6144, 2560), ("in_proj_qkv", 10240, 2560)):
    Ws = [q4(N, K) for _ in range(R)]
    xa = x16[:, :K]
    def chain(cached, Ws=Ws, xa=xa):                               # v2.1: D3-0's independent-call method (throughput)
        return [qmm16c(xa, *(Ws[0] if cached else Ws[i])) for i in range(R)]
    T1 = timed({"cold": lambda: chain(False), "cached": lambda: chain(True)})
    tc, tw = T1["cold"] / R, T1["cached"] / R
    mb = (N * K / 2 + 2 * 2 * N * K / 64) / 1e6; gap = 1 - tw / tc
    res["step1"][name] = {"MB": round(mb, 2), "cold_us": round(tc * 1e6, 1), "cached_us": round(tw * 1e6, 1), "gap": round(gap, 3),
                          "verdict": "staging can help" if gap >= 0.15 else "DROP (gap < 15 %)"}
    print(f"STEP 1 (a) {name:16s} {mb:6.2f} MB | cold {tc * 1e6:7.1f} us | cached {tw * 1e6:7.1f} us | gap {100 * gap:5.1f} % "
          f"-> {res['step1'][name]['verdict']}", flush=True)
    del Ws

# ---------------------------------------------------------------- STEP 2: the window and the touch
L = 1024
q = mx.random.normal((1, 24, 16, 256)).astype(BF); kk = mx.random.normal((1, 2, L, 256)).astype(BF)
vv = mx.random.normal((1, 2, L, 256)).astype(BF); mx.eval(q, kk, vv)
TOUCH_SRC = """
    const device uint4* w4 = (const device uint4*)w;
    uint acc = 0;
    for (uint i = thread_position_in_grid.x; i < NW; i += threads_per_grid.x) { uint4 v = w4[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (thread_index_in_threadgroup == 0) out[threadgroup_position_in_grid.x] = acc + (uint)(dep[0] != dep[0]);
"""
NW = N4 * K4 * 2 // 16
touch_k = mx.fast.metal_kernel(name="nis_touch", input_names=["w", "dep"], output_names=["out"], source=TOUCH_SRC,
                               header=f"#define NW {NW}u\n")
def touch(w, dep):
    (o,) = touch_k(inputs=[w, dep], grid=(256 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(256,)],
                   output_dtypes=[mx.uint32])
    return o
def win(qi):
    return mx.fast.scaled_dot_product_attention(qi, kk, vv, scale=256 ** -0.5)
def win_chain(with_touch):
    qi = q; outs = []
    for i in range(R):
        if with_touch: outs.append(touch(W4[i], qi))
        o = win(qi); outs.append(o); qi = q * (1 + 0 * o[0, 0, 0, 0])
    return outs
T2 = timed({"WIN": lambda: win_chain(False), "WIN+TOUCH": lambda: win_chain(True)})
win_us = T2["WIN"] / R * 1e6; extra_us = (T2["WIN+TOUCH"] - T2["WIN"]) / R * 1e6
res["step2"] = {"win_us": round(win_us, 1), "touch_extra_us": round(extra_us, 1), "line_5us": extra_us <= 5.0}
print(f"STEP 2 window: WIN {win_us:.1f} us | + TOUCH (6.64 MB) +{extra_us:.1f} us per window -> Leap's +5 us line: "
      f"{'MET' if extra_us <= 5.0 else 'NOT MET'}", flush=True)

# ---------------------------------------------------------------- STEP 3 (b): the full sequence
EV = [q4(10240, 2560) for _ in range(R)]; OP = [q4(2560, 6144) for _ in range(R)]
xe, xo = x16[:, :2560], x16[:, :6144]
def seq(arm):
    qi = q; a = x4; outs = []
    for i in range(R):
        e = qmm16c(xe * (1 + 0 * qi[0, 0, 0, 0]), *EV[i])                     # EVICT: experts' stand-in
        qi2 = q * (1 + 0 * e[0, 0])
        if arm == "TOUCHED": outs.append(touch(W4[i], qi2))
        o = win(qi2)                                                          # WIN
        p = qmm16c(xo * (1 + 0 * o[0, 0, 0, 0]), *OP[i])                    # OPROJ
        if arm == "LATE":
            hw = xo * (1 + 0 * p[0, 0])                                       # HC-write stand-in (16-row elementwise)
            outs.append(touch(W4[i], hw)); p = hw[:, :1] * 0 + p[:, :1]
        ai = a * (1 + 0 * p[0, 0])                                            # S4 waits for OPROJ (and the touch)
        if arm == "NO_S4":
            y = ai[0, :1, :1]
        elif arm == "BYTES_FREE":
            y = s4c_partials(ai, W4[i])
        else:
            y = LK.s4_partials(ai, W4[i])[0]
        outs.append(y); qi = q * (1 + 0 * y.reshape(-1)[0].astype(BF))     # keep the chain bf16 (S4's partials are fp32)
    return outs
T3 = timed({arm: (lambda arm=arm: seq(arm)) for arm in ("NO_S4", "COLD", "TOUCHED", "LATE", "BYTES_FREE")})
eff = {arm: (T3[arm] - T3["NO_S4"]) / R * 1e6 for arm in ("COLD", "TOUCHED", "LATE", "BYTES_FREE")}
def s4outs(arm):
    o = seq(arm); mx.eval(o)
    return [np.asarray(t) for t in o if t.dtype == mx.float32 and t.ndim == 3]
oc, ot, ol = s4outs("COLD"), s4outs("TOUCHED"), s4outs("LATE")
same = all(np.array_equal(a_.view(np.uint32), b_.view(np.uint32)) for a_, b_ in zip(oc, ot)) and \
       all(np.array_equal(a_.view(np.uint32), b_.view(np.uint32)) for a_, b_ in zip(oc, ol))
gain = eff["COLD"] - eff["BYTES_FREE"]
def ok(arm): return eff[arm] <= eff["BYTES_FREE"] + 6.3 and eff[arm] <= eff["COLD"] - 0.7 * gain
verdict = "PASS" if (same and ctx and (ok("TOUCHED") or ok("LATE"))) else ("NO JUDGEMENT (context)" if not ctx else "KILL")
res["step3"] = {**{f"{k.lower()}_us": round(v, 1) for k, v in eff.items()}, "bitwise": same,
                "touched_ok": ok("TOUCHED"), "late_ok": ok("LATE"), "verdict": verdict}
print(f"STEP 3 (b) S4 in the sequence: cold {eff['COLD']:.1f} | touched {eff['TOUCHED']:.1f} | late {eff['LATE']:.1f} | "
      f"bytes-free {eff['BYTES_FREE']:.1f} us | bitwise {same} -> LINE (b): {verdict}", flush=True)
print(f"MLX peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
