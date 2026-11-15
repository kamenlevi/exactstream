"""R-09b step 1 (Leap, Mon 2026-10-26, pre-registered 22:33 EET in TABLE_350 'R-09b'; the setup is r09_screen.py's): where
the exact 5-kernel read X5 spends its 14.7 ms, and which GEMM MLX runs for the merged 16-row projection. Decides
whether a same-K-order GEMM with more threadgroups (R-09b) can reach its lines: its ceiling = the merged GEMM's in-chain
cost minus its bytes floor (6.6 MB per read at ~250 GB/s = ~2.5 ms per 96 reads).
  Part 1, bits: the merged projection (M 16, N 324, K 10,240) vs the same rows inside a 32-row GEMM, inside a batch of
  two 16-row GEMMs, and as two 8-row calls; the up projection (K 320, N 10,240) the same way.
  Part 2, knockouts: X5 with one stage at a time replaced by a constant of its shape (its work and its wait removed);
  the time each removal saves = that stage's in-chain cost.
Usage: [BigMoE venv] r09b_screen.py -> ~/Leap/data/r09b_screen.txt

The R-09 setup this copies: R-09 HCF (NewIdeas' rapid screen, GO Mon 2026-10-26 19:12; Leap, pre-registered 20:51 EET in ~/Leap/TABLE_350.md
'R-09'): the hyper-connection read at 16 verify rows, fused. CA run 3: HC reads / writes = 31-34 ms of the 143-150 ms
16-row check; at 16 rows the engine runs the eager chain (~14 kernels per read: write multiply + add, RMS norm, weight
multiply, down / inject / up matmuls, divide, silu, inject gate, sigmoid, product, stream sum, scale). MTPLX's 3-kernel
read serves 2-8 rows only, and matches the compiled chain (its matmuls are MLX's wide gemv, not the 16-row GEMM).
Arms, 96 reads (48 layers x attention + MLP mixers, the model's own weights, each read's pending write = the previous
read's mixed output and inject) in one graph at 16 rows:
  E    the engine's eager chain (GatedResidual.__call__ outside a compiled body, decode_verify phase)
  X    exact by construction, 6 kernels: MTPLX's norm kernel (pending write + grouped RMS norm in MLX's arithmetic),
       MLX's own down / inject / up matmuls (the same calls as E, so the same sums), one gate kernel (silu(/4) and the
       inject gate read from tables of the stock functions at every bf16 input) and one tail kernel (sigmoid table,
       product, stream sum from +0 in stream order, the mean's 1/4: MTPLX's up-kernel tail)
  X5   X with the down and inject matmuls as one (weights concatenated): 5 kernels
  F8   MTPLX's 3-kernel read as two 8-row calls (not expected bitwise: wide gemv sums)
  F16  the same at 16 rows in one call (if it builds)
Bitwise: every arm's 96 (mixed, written, inject) against E's from the same inputs (bf16 bit patterns).
Timing: the 96-read graph built + evaluated, one sync; arms interleaved, median of 20 after 3.
Usage: [BigMoE venv] r09_screen.py -> ~/Leap/data/r09_screen.txt"""
import json, os, time
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
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

cfg = json.load(open(os.path.join(M, "config.json")))
EPS = float(cfg.get("text_config", cfg)["rms_norm_eps"])
idx = json.load(open(os.path.join(M, "model.safetensors.index.json")))["weight_map"]
SUBS = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight")
names = [f"language_model.model.layers.{i}.{p}_hyper_connection" for i in range(48) for p in ("attn", "mlp")]
byshard = {}
for n in names:
    for s in SUBS:
        byshard.setdefault(idx[f"{n}.{s}"], []).append(f"{n}.{s}")
Wt, dts = {}, set()
for f, ks in byshard.items():
    w = mx.load(os.path.join(M, f))
    for k in ks:
        dts.add(str(w[k].dtype)); Wt[k] = w[k].astype(BF)
    del w
mods = []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt[f"{n}.{s}"]) for s in SUBS])
    mods.append(g)
WCAT = [mx.concatenate([g.input_mix_weight_down.weight, g.block_inject_weight.weight], axis=0) for g in mods]
mx.eval([g.parameters() for g in mods], WCAT)
say(f"R-09: {len(mods)} mixers loaded (stored as {sorted(dts)}, run in bf16), eps {EPS}")

# the stock elementwise functions at every bf16 input
vals = mx.view(mx.arange(1 << 16, dtype=mx.uint32).astype(mx.uint16), BF)
TAB = mx.concatenate([nn.silu(vals / HC), Q._verify_inject(vals, HC), mx.sigmoid(vals)])
mx.eval(TAB)

NORM = HV._norm_kernel(HC, W, LR, EPS, True)
GATE2 = mx.fast.metal_kernel(
    name="r09_gate2", input_names=["dn", "ij", "tab"], output_names=["mix", "inject"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i < NDN) {
        mix[i] = tab[as_type<ushort>(dn[i])];
    } else if (i < NDN + NIJ) {
        const uint j = i - NDN;
        inject[j] = tab[65536 + as_type<ushort>(ij[j])];
    }
""")
GATE1 = mx.fast.metal_kernel(   # down and inject logits in one [rows, LR + HC] matmul output
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

def read_X(g, wcat, hyper, bo, inj):
    written, normed = NORM(inputs=[hyper, bo, inj, g.hc_norm.weight], template=[("T", BF)],
                           grid=(ROWS * HC * 256, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(ROWS, HCD), (ROWS, HCD)], output_dtypes=[BF, BF])
    n3 = normed.reshape(1, ROWS, HCD)
    if wcat is None:
        mix, inject = GATE2(inputs=[n3 @ g.input_mix_weight_down.weight.T, n3 @ g.block_inject_weight.weight.T, TAB],
                            template=TC, grid=(ROWS * (LR + HC), 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(ROWS, LR), (ROWS, HC)], output_dtypes=[BF, BF])
    else:
        mix, inject = GATE1(inputs=[n3 @ wcat.T, TAB], template=TC, grid=(ROWS * (LR + HC), 1, 1),
                            threadgroup=(256, 1, 1), output_shapes=[(ROWS, LR), (ROWS, HC)], output_dtypes=[BF, BF])
    up = mix.reshape(1, ROWS, LR) @ g.input_mix_weight_up.weight.T
    (mixed,) = TAIL(inputs=[up, normed, TAB], template=[("T", BF), ("D", W), ("HC", HC), ("NOUT", ROWS * W)],
                    grid=(ROWS * W, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, W)], output_dtypes=[BF])
    return mixed, written, inject

mx.random.seed(9)
H0 = mx.random.normal((ROWS, HCD)).astype(BF)
B0 = mx.random.normal((ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)

# Every chain hands the GPU its work every 8 reads (the engine: async_eval every 4 layers), so graph building overlaps.
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

def chain_X(cat):
    outs, hyper, bo, inj = [], H0, B0, I0
    for i, (g, wc) in enumerate(zip(mods, WCAT)):
        mixed, hyper, inj = read_X(g, wc if cat else None, hyper, bo, inj)
        outs.append((mixed, hyper, inj)); bo = mixed
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs


eq = lambda a, b: bool(mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())
for name, w in (("merged down + inject (K 10,240, N 324)", WCAT[0]), ("up (K 320, N 10,240)", mods[0].input_mix_weight_up.weight)):
    K_ = w.shape[1]
    a = mx.random.normal((1, 16, K_)).astype(BF)
    b = mx.random.normal((1, 16, K_)).astype(BF)
    y16 = a @ w.T
    y32 = mx.concatenate([a, b], axis=1) @ w.T
    yb2 = mx.concatenate([a, b], axis=0) @ w.T
    y8 = mx.concatenate([a[:, :8] @ w.T, a[:, 8:] @ w.T], axis=1)
    mx.eval(y16, y32, yb2, y8)
    say(f"  {name} at 16 rows, bitwise equal to: the same rows of a 32-row GEMM {eq(y16, y32[:, :16])}; "
        f"of a batch of two {eq(y16, yb2[:1])}; two 8-row calls {eq(y16, y8)}")

KO = set()
CONST = {}
def ko(stage, make):
    if stage in KO:
        if stage not in CONST:
            CONST[stage] = make(); mx.eval(CONST[stage])
        return CONST[stage]
    return make()

def read_K(g, wcat, hyper, bo, inj):
    written, normed = ko("norm", lambda: tuple(NORM(inputs=[hyper, bo, inj, g.hc_norm.weight], template=[("T", BF)],
        grid=(ROWS * HC * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, HCD), (ROWS, HCD)],
        output_dtypes=[BF, BF])))
    both = ko("gemm1", lambda: normed.reshape(1, ROWS, HCD) @ wcat.T)
    mix, inject = ko("gate", lambda: tuple(GATE1(inputs=[both, TAB], template=TC, grid=(ROWS * (LR + HC), 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(ROWS, LR), (ROWS, HC)], output_dtypes=[BF, BF])))
    up = ko("gemm2", lambda: mix.reshape(1, ROWS, LR) @ g.input_mix_weight_up.weight.T)
    (mixed,) = ko("tail", lambda: tuple(TAIL(inputs=[up, normed, TAB], template=[("T", BF), ("D", W), ("HC", HC),
        ("NOUT", ROWS * W)], grid=(ROWS * W, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, W)], output_dtypes=[BF])))
    return mixed, written, inject

def chain_K():
    outs, hyper, bo, inj = [], H0, B0, I0
    for i, (g, wc) in enumerate(zip(mods, WCAT)):
        mixed, hyper, inj = read_K(g, wc, hyper, bo, inj)
        outs.append((mixed, hyper, inj)); bo = mixed
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs

ref = chain_X(True); out = chain_K(); mx.eval(ref, out)
say(f"  X5 rebuilt with knockout hooks, nothing knocked out: bitwise = r09's X5 {all(eq(x, y) for r, o in zip(ref, out) for x, y in zip(r, o))}")
VARS = ["X5", "-norm", "-gemm1", "-gate", "-gemm2", "-tail", "E"]
T = {v: [] for v in VARS}
for rep in range(23):
    for v in VARS:
        KO.clear(); CONST.clear()
        if v.startswith("-"):
            KO.add(v[1:])
            chain_K()                      # builds the constant (outside the timing)
            mx.synchronize()
        fn = chain_E if v == "E" else chain_K
        mx.synchronize(); t0 = time.perf_counter()
        o = fn(); mx.eval(o); mx.synchronize()
        T[v].append(1e3 * (time.perf_counter() - t0))
x5 = med(T["X5"][3:])
say(f"  E {med(T['E'][3:]):.2f} ms, X5 {x5:.2f} ms per 96 reads; in-chain cost of each stage (X5 minus X5 without it):")
cost = {v[1:]: x5 - med(T[v][3:]) for v in VARS if v.startswith("-")}
say("    " + ", ".join(f"{k} {c:.2f}" for k, c in cost.items()) + f" ms (sum {sum(cost.values()):.2f})")
ceil = cost["gemm1"] - 2.5
say(f"R-09b step 1: merged GEMM in-chain {cost['gemm1']:.2f} ms; a same-order GEMM at its bytes floor saves <= {ceil:.2f} ms "
    f"more, so R-09b's best = {9.67 + ceil:.2f} ms saved vs E (lines: PASS >= 15, KILL < 11): "
    + ("build it" if 9.67 + ceil >= 15 else "KILL (cannot reach 11)" if 9.67 + ceil < 11 else "BETWEEN at best"))
open(os.path.join(D, "r09b_screen.txt"), "w").write("\n".join(lines) + "\n")
