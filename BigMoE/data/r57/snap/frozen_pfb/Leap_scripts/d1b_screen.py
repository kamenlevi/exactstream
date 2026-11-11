"""D1-B (Leap, Mon 2026-11-02 20:5x EET; pre-registered in ~/Leap/TABLE_350.md 'D1-B'; Fable mission 26, ~/FableGod/HC_READ.md
design D1, kernel B; orchestrator 20:4x, NewIdeas' queue item 1): the 16-row HC read's partition sum + gate (R-82) in the
prologue of a bitwise replica of MLX's K-320 bf16 up GEMM, and leapk's tail in its epilogue: 5 -> 3 kernels per read.
MLX 0.32.2 serves up = mix [16, 320] @ W_up^T with steel_gemm_fused_nax (matmul.cpp: gemv_wide declines at 16 rows, no NAX
split-K at K 320): 64 x 128 tiles on 's' devices, 2 x 4 simdgroups of 32 x 32, BK 256 + a K tail of 64. Every output is the
same sequence of ten 32-wide k steps, each two 16 x 32 x 16 NAX ops (gemm_nax.h gemm_loop, nax.h tile_matmad_nax); the 16
rows fill the first 16-row fragment of the first simdgroup row, whose ops are independent of the zero-filled rows 16-31.
Kernel B: one threadgroup per 32 x CPT channels; simdgroup (sub, s) computes channels ch0 .. ch0 + 31 of stream s, i.e. the
32-aligned column run s x 2560 + ch0 of up, which MLX also gives one simdgroup, so every output keeps its place in its
16 x 32 op (no column permutation of W_up, no weight copy; Fable's interleave is not needed):
  1. W_up loads for the first GPF - 1 k steps go out first (they do not depend on the prologue);
  2. prologue = R-82's _ACCUM_GATE: mix = tab[bf16(0 + p0 + p1 + p2)] for the 16 x 320 entries into threadgroup memory
     (every threadgroup; the 3 x 16 x 324 fp32 partials stay in L2), inject by threadgroup 0; V4 = float4 loads;
  3. the up GEMM: MLX's op sequence, A from threadgroup memory, W_up loads GPF - 1 steps ahead (S4's prefetch);
  4. epilogue: up cast to bf16 (MLX's store) into threadgroup memory, where the four streams of a channel meet, then
     leapk's tail per (row, channel): gate = tab[sigmoid](up), bf16 product with normed (loaded before the GEMM), the
     streams summed from +0 in order, x 1/4.
Checks: (1) the replica's up (debug variant) == MLX's mix @ W_up^T for all 96 mixers, with mixed / inject; (2) 96-read
chains (real HC weights of all 48 layers, attention and MLP reads, the pending write on, as R-09c): every read's mixed /
written / inject == the S4 chain (leapk.hc_read; R-82 off and on); (3) kernels per read (R-65's census); (4) ms per
96-read chain, arms interleaved, median of 40 after 3 (shared GPU: rough).
Lines: bits 96 / 96 else BLOCKED (not a verdict); PASS >= 1.9 ms saved per 96 reads vs S4 + R-82 (Fable's bound for the
B half: 9.9 -> 8.0); KILL < 0.8 ms; BETWEEN otherwise.
Usage: [BigMoE venv] d1b_screen.py -> ~/Leap/data/d1b_screen.txt"""
import collections, json, os, re, sys, tempfile, time
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import leapk as LK
from mtplx.models import qwen4_exp as Q

OUT = os.path.expanduser("~/Leap/data/d1b_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
ROWS, HC, W, LR = 16, 4, 2560, 320
HCD, BF = HC * W, mx.bfloat16
N, NP = LR + HC, 3
M = os.path.expanduser("~/BigMoE/models/bare")
say(f"D1-B screen {time.strftime('%a %H:%M')}")

# ---------------------------------------------------------------- the 96 mixers (R-09c's setup, without the spare weights)
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
G, WCAT = [], []
for n in names:
    g = Q.GatedResidual(SimpleNamespace(hc_count=HC, hidden_size=W, rms_norm_eps=EPS, hc_lowrank=LR))
    g.load_weights([(s, Wt.pop(f"{n}.{s}")) for s in SUBS])
    WCAT.append(mx.concatenate([g.input_mix_weight_down.weight, g.block_inject_weight.weight], axis=0))
    mx.eval(WCAT[-1], g.parameters())
    g.input_mix_weight_down.weight = mx.zeros((LR, 1), BF)    # hc_read reads only its row count; the merged copy is WCAT
    g.block_inject_weight.weight = mx.zeros((HC, 1), BF)
    G.append(g)
mx.eval([g.parameters() for g in G])
EPS_HC = float(G[0].hc_norm.eps)
KH = LK._hc_kernels(HC, W, LR, EPS_HC)
say(f"{len(G)} mixers; up GEMM M {ROWS} x N {HCD} x K {LR}; hc_norm eps {EPS_HC}")

# ---------------------------------------------------------------- kernel B
B_SRC = """
    using namespace mlx::steel;
    threadgroup T Am[16 * LDA];
    threadgroup T Gs[CPT * HC * 16 * 32];
    const int tg = int(threadgroup_position_in_grid.x);
    const uint lid = thread_index_in_threadgroup;
    const short sg = short(simdgroup_index_in_threadgroup);
    const short st = sg % HC, sub = sg / HC;
    const int ch0 = (tg * CPT + sub) * 32;
    const device T* Bp = wu + size_t(st * D + ch0) * LR;
    NAXTile<T, 1, 2> As[GPF];
    NAXTile<T, 2, 2> Bs[GPF];
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < GPF - 1; j++) {
        Bs[j].load(Bp + j * 32, LR);
    }
    T nv[NO][HC];
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < NO; q++) {
        const uint o = lid + q * NTH;
        const uint r = o / (32 * CPT), cc = o % (32 * CPT);
        STEEL_PRAGMA_UNROLL
        for (int s = 0; s < HC; s++) {
            nv[q][s] = normed[(size_t)r * HC * D + (size_t)s * D + uint(tg) * 32 * CPT + cc];
        }
    }
//PROLOGUE
    if (tg == 0 && lid < 16 * HC) {
        const uint r = lid / HC, j = lid % HC;
        float out = 0;
        for (int p = 0; p < NP; p++) {
            out += cs[size_t(p) * 16 * N + r * N + LR + j];
        }
        const T yv = static_cast<T>(out);
        inject[r * HC + j] = tab[65536 + as_type<ushort>(yv)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    NAXTile<float, 1, 2> Dt;
    Dt.clear();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < GPF - 1; j++) {
        As[j].template load<T, LDA, 1>(Am + j * 32);
    }
    STEEL_PRAGMA_UNROLL
    for (int s0 = 0; s0 < ITERS; s0 += GPF) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < GPF; j++) {
            const int s = s0 + j;
            if (s < ITERS) {
                const int nx = s + GPF - 1;
                if (nx < ITERS) {
                    As[(j + GPF - 1) % GPF].template load<T, LDA, 1>(Am + nx * 32);
                    Bs[(j + GPF - 1) % GPF].load(Bp + nx * 32, LR);
                }
                tile_matmad_nax(Dt, As[j], metal::bool_constant<false>{}, Bs[j], metal::bool_constant<true>{});
            }
        }
    }
    Dt.template store<T, 32, 1>(Gs + (sub * HC + st) * 16 * 32);
//DBG
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < NO; q++) {
        const uint o = lid + q * NTH;
        const uint r = o / (32 * CPT), cc = o % (32 * CPT);
        const uint sb = cc / 32, cl = cc % 32;
        T total = T(0);
        for (int s = 0; s < HC; ++s) {
            const T gate = tab[2 * 65536 + as_type<ushort>(Gs[((sb * HC + s) * 16 + r) * 32 + cl])];
            const T product = gate * nv[q][s];
            const T lane_total = product + T(0);
            total = (s == 0) ? lane_total : T(lane_total + total);
        }
        mixed[r * D + uint(tg) * 32 * CPT + cc] = total * T(0.25f);
    }
"""
PRO_S = """
    for (uint i = lid; i < 16 * LR; i += NTH) {
        const uint r = i / LR, c = i % LR;
        float out = 0;
        for (int p = 0; p < NP; p++) {
            out += cs[size_t(p) * 16 * N + r * N + c];
        }
        const T yv = static_cast<T>(out);
        Am[r * LDA + c] = tab[as_type<ushort>(yv)];
    }
"""
PRO_V4 = """
    const device float4* cs4 = (const device float4*)cs;
    for (uint i = lid; i < 16 * (LR / 4); i += NTH) {
        const uint r = i / (LR / 4), c = (i % (LR / 4)) * 4;
        float4 out = 0;
        for (int p = 0; p < NP; p++) {
            out += cs4[(size_t(p) * 16 * N + r * N + c) / 4];
        }
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < 4; e++) {
            const T yv = static_cast<T>(out[e]);
            Am[r * LDA + c + e] = tab[as_type<ushort>(yv)];
        }
    }
"""
KB = {}
def kern_b(cpt, gpf, v4=False, dbg=False):
    key = (cpt, gpf, v4, dbg)
    if key not in KB:
        src = B_SRC.replace("//PROLOGUE", PRO_V4 if v4 else PRO_S).replace(
            "//DBG", "    Dt.store(upo + size_t(st * D + ch0), HC * D);" if dbg else "")
        hdr = LK._nax_header() + "\n" + LK.consts(LR=LR, HC=HC, D=W, N=N, NP=NP, LDA=LR + 8, CPT=cpt, GPF=gpf,
                                                    NTH=128 * cpt, ITERS=LR // 32, NO=4)
        KB[key] = mx.fast.metal_kernel(name=f"d1b_{cpt}_{gpf}_{int(v4)}_{int(dbg)}",
                                       input_names=["cs", "tab", "wu", "normed"],
                                       output_names=["mixed", "inject"] + (["upo"] if dbg else []), source=src, header=hdr)
    k = KB[key]
    def f(cs, tab, wu, normed):
        return k(inputs=[cs, tab, wu, normed], template=[("T", BF)], grid=(128 * cpt * (W // 32 // cpt), 1, 1),
                 threadgroup=(128 * cpt, 1, 1), output_shapes=[(ROWS, W), (ROWS, HC)] + ([(ROWS, HCD)] if dbg else []),
                 output_dtypes=[BF, BF] + ([BF] if dbg else []))
    return f

def read_b(cpt, gpf, v4):
    kb = kern_b(cpt, gpf, v4)
    def f(i, hyper, bo, inj):
        g = G[i]
        written, normed = KH["norm_p"](inputs=[hyper.reshape(ROWS, HCD), bo.reshape(ROWS, W), inj.reshape(ROWS, HC),
                                               g.hc_norm.weight], template=[("T", BF)], grid=(ROWS * HC * 256, 1, 1),
                                       threadgroup=(256, 1, 1), output_shapes=[(ROWS, HCD), (ROWS, HCD)],
                                       output_dtypes=[BF, BF])
        cs, _ = LK.s4_partials(normed.reshape(1, ROWS, HCD), WCAT[i])
        mixed, inject = kb(cs, KH["tab"], g.input_mix_weight_up.weight, normed)
        return mixed.reshape(1, ROWS, W), written.reshape(1, ROWS, HCD), inject.reshape(1, ROWS, HC)
    return f

def read_s4(fuse):
    return lambda i, hyper, bo, inj: LK.hc_read(G[i], WCAT[i], hyper, (bo, inj), rows=ROWS, fuse_gate=fuse)

ARMS = {"S4": read_s4(False), "S4g": read_s4(True)}
SPEC = os.environ.get("D1B_ARMS", "B1p4,B2p4,B4p4,B2p1,B2p2,B2p4v")   # follow-ups name other B{cpt}p{gpf}[v] arms
for a in SPEC.split(","):
    m = re.fullmatch(r"B(\d)p(\d)(v?)", a)
    ARMS[a] = read_b(int(m.group(1)), int(m.group(2)), m.group(3) == "v")
BARMS = [a for a in ARMS if a.startswith("B")]

eq = lambda a, b: bool(a.shape == b.shape and mx.all(mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())

# ---------------------------------------------------------------- 1. the replica alone vs MLX's up GEMM (+ mixed / inject)
mx.random.seed(9)
H0 = mx.random.normal((1, ROWS, HCD)).astype(BF)
B0 = mx.random.normal((1, ROWS, W)).astype(BF)
I0 = (2 * mx.sigmoid(mx.random.normal((1, ROWS, HC)))).astype(BF)
mx.eval(H0, B0, I0)
rep = collections.Counter()
for i, g in enumerate(G):
    (normed,) = KH["norm_n"](inputs=[H0.reshape(ROWS, HCD)] * 3 + [g.hc_norm.weight], template=[("T", BF)],
                             grid=(ROWS * HC * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, HCD)],
                             output_dtypes=[BF])
    cs, _ = LK.s4_partials(normed.reshape(1, ROWS, HCD), WCAT[i])
    mix, inj = LK._ACCUM_GATE(inputs=[cs, KH["tab"]], template=[("T", BF), ("N", N), ("NP", NP), ("ROWS", ROWS),
                              ("LR", LR), ("HC", HC)], grid=(16 * N, 1, 1), threadgroup=(256, 1, 1),
                              output_shapes=[(ROWS, LR), (ROWS, HC)], output_dtypes=[BF, BF])
    up = mix.reshape(1, ROWS, LR) @ g.input_mix_weight_up.weight.T
    (mixed,) = KH["tail"](inputs=[up, normed, KH["tab"]], template=[("T", BF), ("D", W), ("HC", HC), ("NOUT", ROWS * W)],
                          grid=(ROWS * W, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(ROWS, W)], output_dtypes=[BF])
    for v4 in (False, True):
        mb, ib, ub = kern_b(2, 4, v4, dbg=True)(cs, KH["tab"], g.input_mix_weight_up.weight, normed)
        mx.eval(mb, ib, ub, up, mixed, inj)
        tag = "v4" if v4 else "scalar"
        rep[f"{tag} up"] += eq(ub, up.reshape(ROWS, HCD))
        rep[f"{tag} mixed"] += eq(mb, mixed)
        rep[f"{tag} inject"] += eq(ib, inj)
    if i == 0 and not eq(ub, up.reshape(ROWS, HCD)):
        d = (np.array(ub.view(mx.uint16)) != np.array(up.reshape(ROWS, HCD).view(mx.uint16)))
        say(f"   mixer 0: up differs in {d.mean() * 100:.3f} % of values; first at {np.argwhere(d)[:4].tolist()}")
say("1. replica (debug variant B2p4) == MLX's up GEMM / tail / inject, of 96 mixers: " + ", ".join(f"{k} {v}" for k, v in rep.items()))

# ---------------------------------------------------------------- 2. 96-read chains, bitwise vs the S4 chain
def chain(arm):
    outs, hyper, bo, inj = [], H0, B0, I0
    for i in range(len(G)):
        mixed, written, inject = arm(i, hyper, bo, inj)
        outs.append((mixed, written, inject))
        hyper, bo, inj = written, mixed, inject
        if i % 8 == 7:
            mx.async_eval(outs[-1])
    return outs
ref = chain(ARMS["S4"])
mx.eval(ref)
cbit = {}
for a in ARMS:
    if a == "S4":
        continue
    o = chain(ARMS[a])
    mx.eval(o)
    cbit[a] = sum(all(eq(x, y) for x, y in zip(r_, o_)) for r_, o_ in zip(ref, o))
say("2. chains: reads bitwise == the S4 chain (mixed, written, inject), of 96: " + ", ".join(f"{k} {v}" for k, v in cbit.items()))

# ---------------------------------------------------------------- 3. kernels per read
VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided", "Flatten",
         "Unflatten", "StopGradient", "Depends", "Split"}
def census(*arrs):
    cnt = collections.Counter()
    fd, p = tempfile.mkstemp(suffix=".dot")
    os.close(fd)
    try:
        mx.export_to_dot(p, *arrs)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m and m.group(1) not in VIEWS:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return sum(cnt.values())
say("3. kernels per read: " + ", ".join(f"{a} {census(*ARMS[a](3, H0, B0, I0))}" for a in ("S4", "S4g", BARMS[0])))

# ---------------------------------------------------------------- 4. ms per 96-read chain
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
order = list(ARMS)
T = {a: [] for a in order}
for r in range(43):
    while os.path.exists(PAUSE):
        time.sleep(5)
    for a in order[r % len(order):] + order[:r % len(order)]:
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(chain(ARMS[a]))
        mx.synchronize()
        T[a].append(1e3 * (time.perf_counter() - t0))
med = {a: float(np.median(T[a][3:])) for a in order}
say("4. ms per 96-read chain (median of 40, shared GPU, rough): " + ", ".join(
    f"{a} {med[a]:.2f} [{np.percentile(T[a][3:], 10):.2f}-{np.percentile(T[a][3:], 90):.2f}]" for a in order))
say("   saved vs S4 + R-82 (S4g): " + ", ".join(f"{a} {med['S4g'] - med[a]:+.2f}" for a in order if a != "S4g"))
ok = [a for a in BARMS if cbit[a] == len(G)]
if not ok:
    say("D1-B verdict: BLOCKED (no kernel-B arm is bitwise on all 96 reads)")
else:
    best = min(ok, key=lambda a: med[a])
    sv = med["S4g"] - med[best]
    v = "PASS" if sv >= 1.9 else "KILL" if sv < 0.8 else "BETWEEN"
    say(f"D1-B verdict (PASS >= 1.9 ms saved per 96 reads vs S4 + R-82 AND bitwise; KILL < 0.8): {v} (best bitwise arm "
        f"{best}: {sv:.2f} ms saved vs S4g, {med['S4'] - med[best]:.2f} vs S4; shared GPU, rough)")
