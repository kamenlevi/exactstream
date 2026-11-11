"""R-09 HCF (NewIdeas' rapid screen, GO Mon 2026-10-26 19:12; Leap, pre-registered 20:51 EET in ~/Leap/TABLE_350.md
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

SIG = HV.sigmoid_table(BF)
def chain_F(rows):
    """MTPLX's verify read on row blocks of `rows` (independent chains: every op is per row). Timed without joining
    the blocks after each read (the engine would: ~2 small copies per read more)"""
    parts = [[] for _ in range(0, ROWS, rows)]
    state = [(H0[r0:r0 + rows], B0[r0:r0 + rows], I0[r0:r0 + rows]) for r0 in range(0, ROWS, rows)]
    for i, g in enumerate(mods):
        for b, (hyper, bo, inj) in enumerate(state):
            mixed, hyper, inj = HV.read_rows(hyper, g.hc_norm.weight, g.input_mix_weight_down.weight,
                                             g.input_mix_weight_up.weight, g.block_inject_weight.weight, bo, inj,
                                             hc=HC, eps=EPS, sigmoid=SIG)
            parts[b].append((mixed, hyper, inj)); state[b] = (hyper, mixed, inj)
        if i % 8 == 7:
            mx.async_eval([p[-1] for p in parts])
    return parts

def joined(parts):
    return [tuple(mx.concatenate([p[i][k] for p in parts]) for k in range(3)) for i in range(len(mods))]

ARMS = {"E": chain_E, "X": lambda: chain_X(False), "X5": lambda: chain_X(True),
        "F8": lambda: chain_F(8), "F16": lambda: chain_F(16)}
REF = chain_E(); mx.eval(REF)
def bits(o):
    return [mx.view(a.reshape(-1), mx.uint16) for a in o]
for name in list(ARMS):
    if name == "E":
        continue
    try:
        out = ARMS[name](); out = joined(out) if name.startswith("F") else out; mx.eval(out)
    except Exception as e:                                 # an arm that does not build is reported, not fatal
        say(f"  {name}: does not run ({type(e).__name__}: {str(e)[:160]})"); del ARMS[name]; continue
    first, nd = None, 0
    for i, (a, b) in enumerate(zip(REF, out)):
        d = sum(int(mx.sum(x != y).item()) for x, y in zip(bits(a), bits(b)))
        if d and first is None:
            first = i
        nd += d
    say(f"  {name}: " + ("BITWISE = E over all 96 reads (mixed, written, inject)" if first is None else
                         f"differs from E: first at read {first}, {nd} values over the chain"))
    ARMS[name] = (ARMS[name], first is None)
ARMS["E"] = (chain_E, True)

T, TB = {n: [] for n in ARMS}, {n: [] for n in ARMS}
for rep in range(23):
    for name, (fn, _) in ARMS.items():
        mx.synchronize(); t0 = time.perf_counter()
        out = fn(); t1 = time.perf_counter()
        mx.eval(out); mx.synchronize()
        T[name].append(1e3 * (time.perf_counter() - t0)); TB[name].append(1e3 * (t1 - t0))
e = med(T["E"][3:])
for name, (fn, exact) in ARMS.items():
    t = med(T[name][3:])
    say(f"  {name:3s}: {t:6.2f} ms per 96 reads (graph build {med(TB[name][3:]):5.2f}); saves {e - t:5.2f} ms per check "
        f"vs E; {'bitwise' if exact else 'NOT bitwise'}")
ex = [(med(T[n][3:]), n) for n, (_, x) in ARMS.items() if x and n != "E"]
best = min(ex) if ex else None
saved = e - best[0] if best else 0.0
v = ("PASS" if saved >= 15 else "KILL" if saved < 8 else "BETWEEN") if best else "KILL (no exact arm)"
fast = [(med(T[n][3:]), n) for n, (_, x) in ARMS.items() if not x]
say(f"R-09 verdict (PASS >= 15 ms saved per 16-row check and bitwise = the eager chain; KILL < 8 ms): {v}"
    + (f" (best exact: {best[1]}, saves {saved:.2f} ms)" if best else "")
    + (f"; fastest non-bitwise (a reference change): {min(fast)[1]} saves {e - min(fast)[0]:.2f} ms" if fast else ""))
open(os.path.join(D, "r09_screen.txt"), "w").write("\n".join(lines) + "\n")
