"""R-09b build (Leap, pre-registered Tue 2026-10-27 19:41 EET in ~/Leap/TABLE_350.md 'R-09b build'; setup from
r09b_screen.py): the merged 16-row down + inject projection (M 16, N 324, K 10,240) with MLX's own summation order but
its loads prefetched.
MLX 0.32.2 serves this shape with its NAX split-K kernel (matmul.cpp: gemv_wide declines at 4 passes; use_nax; K >= 3 x
max(M, N)): 64 x 64 tiles, 2 x 2 simdgroups, BK 256, K partitions of 4,096 (4,096 + 4,096 + 2,048), float partials,
then gemm_splitk_accum (out = 0; out += each partition in order; cast). Only 18 threadgroups (36 working simdgroups:
the M = 16 rows fill half of each tile). Each simdgroup walks its partition in 32-wide k steps, loading A and B
straight from device memory and multiplying with the fixed 16 x 32 x 16 NAX op (gemm_nax.h gemm_loop: no unroll, a
volatile compiler barrier per step), so every step waits on memory.
Kernels here (mx.fast.metal_kernel; MLX's nax.h inlined from the installed package, read only):
  S<pf>  one simdgroup per 32-column tile and partition (33 simdgroups), the same NAX op sequence per output, loads
         issued pf - 1 steps ahead (pf 1 = MLX's own loop), then the same accumulation kernel
  F<pf>  one simdgroup per 32-column tile walks all three partitions, adds them in MLX's order (0 + p0 + p1 + p2) in
         registers and casts: no accumulation kernel (11 simdgroups)
Checks: every kernel's 96 projections (the model's merged weights) bitwise = MLX's (a @ w.T); the X5 chain with the
kernel as its merged GEMM bitwise = X5 for all 96 reads. Timing: arms interleaved, 96-read chains, median of 20 after 3.
Lines (QUEUE R-09b): PASS >= 15 ms saved per check vs E AND bitwise. KILL < 11 ms. A replica (S1) that is not bitwise
= MLX, or a compile failure, is BLOCKED (not a verdict on the idea).
Usage: [BigMoE venv] r09c_screen.py -> ~/Leap/data/r09c_screen.txt"""
import json, os, re, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mtplx.models import qwen4_exp as Q
from mtplx.kernels import hc_verify_read as HV
from mtplx.attention_context import attention_phase

D = os.path.expanduser("~/Leap/data")
M = os.path.expanduser("~/BigMoE/models/bare")
ROWS, HC, W, LR = 16, 4, 2560, 320
HCD, BF = HC * W, mx.bfloat16
N, K = LR + HC, HCD                                      # the merged projection: 324 x 10,240
PS = 4096                                                # MLX's NAX split-K partition size for this shape
NP, NT = (K + PS - 1) // PS, (N + 31) // 32
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))
def done(code):
    open(os.path.join(D, "r09c_screen.txt"), "w").write("\n".join(lines) + "\n")
    raise SystemExit(code)

# ---------------------------------------------------------------- r09b_screen.py's setup (96 mixers, tables, kernels)
cfg = json.load(open(os.path.join(M, "config.json")))
EPS = float(cfg.get("text_config", cfg)["rms_norm_eps"])
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
SUBS = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight")
names = [f"language_model.model.layers.{i}.{p}_hyper_connection" for i in range(48) for p in ("attn", "mlp")]
byshard = {}
for n in names:
    for s in SUBS:
        byshard.setdefault(idx[f"{n}.{s}"], []).append(f"{n}.{s}")
Wt = {}
for f, ks in byshard.items():
    w = mx.load(os.path.join(M, f))
    for k in ks:
        Wt[k] = w[k].astype(BF)
    del w
mods = []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt[f"{n}.{s}"]) for s in SUBS])
    mods.append(g)
WCAT = [mx.concatenate([g.input_mix_weight_down.weight, g.block_inject_weight.weight], axis=0) for g in mods]
mx.eval([g.parameters() for g in mods], WCAT)
vals = mx.view(mx.arange(1 << 16, dtype=mx.uint32).astype(mx.uint16), BF)
TAB = mx.concatenate([nn.silu(vals / HC), Q._verify_inject(vals, HC), mx.sigmoid(vals)])
mx.eval(TAB)
NORM = HV._norm_kernel(HC, W, LR, EPS, True)
GATE1 = mx.fast.metal_kernel(
    name="r09_gate1", input_names=["both", "tab"], output_names=["mix", "inject"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i < NDN) {
        const uint r = i / LR, c = i % LR;
        mix[i] = tab[as_type<ushort>(both[r * (LR + HC) + c])];
    } else if (i < NDN + NIJ) {
        const uint j = i - NDN, r = j / HC, c = j % HC;
        inject[j] = tab[65536 + as_type<ushort>(both[r * (LR + HC) + LR + c])];
    }
""")
TAIL = mx.fast.metal_kernel(
    name="r09_tail", input_names=["up", "normed", "tab"], output_names=["mixed"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= NOUT) return;
    const uint r = i / D, c = i % D;
    T total = T(0);
    for (int s = 0; s < HC; ++s) {
        const size_t k = (size_t)r * HC * D + (size_t)s * D + c;
        const T gate = tab[2 * 65536 + as_type<ushort>(up[k])];
        const T product = gate * normed[k];
        const T lane_total = product + T(0);
        total = (s == 0) ? lane_total : T(lane_total + total);
    }
    mixed[i] = total * T(0.25f);
""")
TC = [("NDN", ROWS * LR), ("NIJ", ROWS * HC), ("LR", LR), ("HC", HC)]
say(f"R-09b build: {len(mods)} mixers loaded; merged projection M {ROWS} x N {N} x K {K}; MLX's split: {NP} partitions "
    f"of {PS}, here {NT} column tiles of 32")

# ---------------------------------------------------------------- the kernels
INC = Path(mx.__file__).resolve().parent / "include"
def inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (INC / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)
HDR = inline("mlx/backend/metal/kernels/steel/gemm/nax.h", set())

WALK = """
    using namespace mlx::steel;
    const bool full_n = sgp_sn == 32;
    NAXTile<T, 1, 2> As[GPF];
    NAXTile<T, 2, 2> Bs[GPF];
    auto ld = [&](thread NAXTile<T, 1, 2>& a_, thread NAXTile<T, 2, 2>& b_, int s) {
        a_.load(Ap + s * 32, GK);
        if (full_n) {
            b_.load(Bp + s * 32, GK);
        } else {
            b_.load_safe(Bp + s * 32, GK, short2(32, sgp_sn));
        }
    };
    Dt.clear();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < GPF - 1; j++) {
        if (j < iters) ld(As[j], Bs[j], j);
    }
    for (int s0 = 0; s0 < iters; s0 += GPF) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < GPF; j++) {
            const int s = s0 + j;
            if (s < iters) {
                const int nx = s + GPF - 1;
                if (nx < iters) ld(As[(j + GPF - 1) % GPF], Bs[(j + GPF - 1) % GPF], nx);
                tile_matmad_nax(Dt, As[j], metal::bool_constant<false>{}, Bs[j], metal::bool_constant<true>{});
            }
        }
    }
"""
SPLIT_SRC = """
    using namespace mlx::steel;
    const int tg = int(threadgroup_position_in_grid.x);
    const int tile = tg % GNT, part = tg / GNT;
    const int c_col = tile * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
    const int iters = (k1 - k0) / 32;
    const device T* Ap = x + k0;
    const device T* Bp = w + size_t(c_col) * GK + k0;
    NAXTile<float, 1, 2> Dt;
""" + WALK + """
    device float* Cp = cs + size_t(part) * 16 * GN + c_col;
    if (full_n) {
        Dt.store(Cp, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
    }
"""
FUSED_SRC = """
    using namespace mlx::steel;
    const int tile = int(threadgroup_position_in_grid.x);
    const int c_col = tile * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    NAXTile<float, 1, 2> St;
    St.clear();                                   // MLX's accumulation: out = 0; out += each partition in order
    for (int part = 0; part < GNP; part++) {
        const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
        const int iters = (k1 - k0) / 32;
        const device T* Ap = x + k0;
        const device T* Bp = w + size_t(c_col) * GK + k0;
        NAXTile<float, 1, 2> Dt;
""" + WALK + """
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < NAXTile<float, 1, 2>::kElemsPerTile; e++) {
            St.elems()[e] += Dt.elems()[e];
        }
    }
    if (sgp_sn == 32) {
        St.store(y + c_col, GN);
    } else {
        St.store_safe(y + c_col, GN, short2(sgp_sn, 16));
    }
"""
ACCUM = mx.fast.metal_kernel(
    name="r09c_accum", input_names=["cs"], output_names=["y"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= 16 * N) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * 16 * N + i];
    }
    y[i] = static_cast<T>(out);
""")
def consts(pf):   # integer constants in the header, as MTPLX's kernels do (int template args + a header did not build)
    return HDR + "".join(f"\nconstant constexpr int {k} = {v};" for k, v in
                         (("GN", N), ("GK", K), ("GPS", PS), ("GNP", NP), ("GNT", NT), ("GPF", pf))) + "\n"
KERN = {}
def kern(kind, pf):
    if (kind, pf) not in KERN:
        KERN[(kind, pf)] = mx.fast.metal_kernel(
            name=f"r09c_{kind}{pf}", input_names=["x", "w"], output_names=["cs" if kind == "S" else "y"],
            source=SPLIT_SRC if kind == "S" else FUSED_SRC, header=consts(pf))
    return KERN[(kind, pf)]

def gemm(kind, pf):
    def f(a, w):                                         # a: [1, 16, K] bf16, w: [N, K] bf16 -> [1, 16, N] bf16
        if kind == "S":
            (cs,) = kern("S", pf)(inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
                                  output_shapes=[(NP, 16, N)], output_dtypes=[mx.float32])
            (y,) = ACCUM(inputs=[cs], template=[("T", BF), ("N", N), ("NP", NP)], grid=(16 * N, 1, 1),
                         threadgroup=(256, 1, 1), output_shapes=[(1, 16, N)], output_dtypes=[BF])
        else:
            (y,) = kern("F", pf)(inputs=[a, w], template=[("T", BF)], grid=(32 * NT, 1, 1), threadgroup=(32, 1, 1),
                                 output_shapes=[(1, 16, N)], output_dtypes=[BF])
        return y
    return f
mlx_gemm = lambda a, w: a @ w.T
ARMS = {"S1": gemm("S", 1), "S2": gemm("S", 2), "S4": gemm("S", 4), "S8": gemm("S", 8), "F4": gemm("F", 4),
        "F8": gemm("F", 8)}

eq = lambda a, b: bool(mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())
mx.random.seed(9)
A0 = mx.random.normal((1, ROWS, K)).astype(BF)
mx.eval(A0)
ref = [mlx_gemm(A0, w) for w in WCAT]
mx.eval(ref)
bit = {}
for name, f in ARMS.items():
    try:
        out = [f(A0, w) for w in WCAT]
        mx.eval(out)
    except Exception as exc:                             # e.g. no Metal 4 / MetalPerformancePrimitives for custom kernels
        say(f"  {name}: kernel failed: {str(exc)[:600]}")
        say("R-09b build verdict: BLOCKED (the NAX kernel does not build as a custom kernel here)")
        done(2)
    bit[name] = sum(eq(o, r) for o, r in zip(out, ref))
say("  projections bitwise = MLX's (of 96): " + ", ".join(f"{k} {v}" for k, v in bit.items()))
if bit["S1"] != 96:
    say("R-09b build verdict: BLOCKED (the replica S1 is not MLX's sums: MLX may not run the NAX split-K here)")
    done(2)

# ---------------------------------------------------------------- the X5 chain with each GEMM
mx.random.seed(9)
H0 = mx.random.normal((ROWS, HCD)).astype(BF)
B0 = mx.random.normal((ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)

def read(g, wcat, hyper, bo, inj, gm):
    written, normed = NORM(inputs=[hyper, bo, inj, g.hc_norm.weight], template=[("T", BF)],
                           grid=(ROWS * HC * 256, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(ROWS, HCD), (ROWS, HCD)], output_dtypes=[BF, BF])
    both = gm(normed.reshape(1, ROWS, HCD), wcat)
    mix, inject = GATE1(inputs=[both, TAB], template=TC, grid=(ROWS * (LR + HC), 1, 1), threadgroup=(256, 1, 1),
                        output_shapes=[(ROWS, LR), (ROWS, HC)], output_dtypes=[BF, BF])
    up = mix.reshape(1, ROWS, LR) @ g.input_mix_weight_up.weight.T
    (mixed,) = TAIL(inputs=[up, normed, TAB], template=[("T", BF), ("D", W), ("HC", HC), ("NOUT", ROWS * W)],
                    grid=(ROWS * W, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, W)], output_dtypes=[BF])
    return mixed, written, inject

def chain(gm):
    outs, hyper, bo, inj = [], H0, B0, I0
    for i, (g, wc) in enumerate(zip(mods, WCAT)):
        mixed, hyper, inj = read(g, wc, hyper, bo, inj, gm)
        outs.append((mixed, hyper, inj)); bo = mixed
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs

def chain_E():
    outs, hyper, pend = [], H0[None], (B0[None], I0[None])
    with attention_phase("decode_verify"):
        for i, g in enumerate(mods):
            mixed, written, inj = g(hyper, pend)
            outs.append((mixed[0], written[0], inj[0]))
            hyper, pend = written, (mixed, inj)
            if i % 8 == 7:
                mx.async_eval(outs[-1])
    return outs

x5 = chain(mlx_gemm); mx.eval(x5)
e = chain_E(); mx.eval(e)
say(f"  X5 bitwise = E (96 reads): {all(eq(x, y) for r, o in zip(x5, e) for x, y in zip(r, o))}")
cbit = {}
for name, f in ARMS.items():
    o = chain(f); mx.eval(o)
    cbit[name] = all(eq(x, y) for r, s in zip(x5, o) for x, y in zip(r, s))
say("  chains bitwise = X5 (all 96 reads): " + ", ".join(f"{k} {v}" for k, v in cbit.items()))

VARS = ["E", "X5"] + list(ARMS)
T = {v: [] for v in VARS}
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
for rep in range(23):
    while os.path.exists(PAUSE):
        time.sleep(5)
    for v in VARS:
        fn = chain_E if v == "E" else (lambda: chain(mlx_gemm)) if v == "X5" else (lambda f=ARMS[v]: chain(f))
        mx.synchronize(); t0 = time.perf_counter()
        o = fn(); mx.eval(o); mx.synchronize()
        T[v].append(1e3 * (time.perf_counter() - t0))
m = {v: med(T[v][3:]) for v in VARS}
say("  per 96 reads (ms, median of 20): " + ", ".join(f"{v} {m[v]:.2f}" for v in VARS))
say("  saved vs E: " + ", ".join(f"{v} {m['E'] - m[v]:.2f}" for v in VARS[1:]))
ok = [v for v in ARMS if cbit[v] and bit[v] == 96]
best = min(ok, key=lambda v: m[v]) if ok else None
saved = m["E"] - m[best] if best else 0.0
v = ("PASS" if saved >= 15 else "KILL" if saved < 11 else "BETWEEN") if best else "KILL (no bitwise arm)"
say(f"R-09b build verdict (PASS >= 15 ms saved per check vs E AND bitwise; KILL < 11): {v} (best bitwise arm {best}: "
    f"{saved:.2f} ms saved; X5 alone {m['E'] - m['X5']:.2f})")
done(0)
