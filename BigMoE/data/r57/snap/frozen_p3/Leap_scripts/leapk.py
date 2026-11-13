"""Leap's bit-identical 16-row kernels (Tue 2026-10-27), shared by the screens (from r09c_screen.py and
r13d_screen.py, unchanged in arithmetic):
  hc_read(g, wcat, hyper, pending)  the engine's 16-row hyper-connection read with R-09b's S4 GEMM (MLX's NAX split-K
                                    order on 33 one-simdgroup tiles, loads 3 steps ahead): fused norm (+ pending
                                    write), merged down + inject GEMM, gate, MLX's up GEMM, tail. Bitwise = the
                                    engine's eager chain (R-09 / R-09b).
  splitk_d2(x, W) / routes_splitk   R-13d's D2: MLX's own split-K qmm_t_impl at BM 16 + mx.sum over the partitions
                                    (MLX's own reduction). Bitwise = mx.quantized_matmul where MLX takes split-K.
Integer constants live in each kernel's header (int template arguments with a header did not build)."""
import re
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

BF = mx.bfloat16
INC = Path(mx.__file__).resolve().parent / "include"

def inline(rel, seen=None):
    """an installed MLX header with its mlx/ includes inlined once each"""
    seen = set() if seen is None else seen
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

def consts(**kv):
    return "".join(f"constant constexpr int {k} = {v};\n" for k, v in kv.items())

# ---------------------------------------------------------------- S4: the merged HC GEMM with MLX's NAX split-K sums
_NAX = None
def _nax_header():
    global _NAX
    if _NAX is None:
        _NAX = inline("mlx/backend/metal/kernels/steel/gemm/nax.h")
    return _NAX

_S4_SRC = """
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
    device float* Cp = cs + size_t(part) * 16 * GN + c_col;
    if (full_n) {
        Dt.store(Cp, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
    }
"""
_ACCUM = mx.fast.metal_kernel(
    name="leapk_accum", input_names=["cs"], output_names=["y"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= 16 * N) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * 16 * N + i];
    }
    y[i] = static_cast<T>(out);
""")
_S4K = {}
def s4_gemm(a, w, pf=4):
    """[1, 16, K] x [N, K]^T -> [1, 16, N] bf16 with MLX's NAX split-K sums (partitions of 4,096)"""
    N, K = w.shape
    PS = 4096
    NP, NT = (K + PS - 1) // PS, (N + 31) // 32
    key = (N, K, pf)
    if key not in _S4K:
        _S4K[key] = mx.fast.metal_kernel(
            name=f"leapk_s4_{N}_{K}_{pf}", input_names=["x", "w"], output_names=["cs"], source=_S4_SRC,
            header=_nax_header() + "\n" + consts(GN=N, GK=K, GPS=PS, GNP=NP, GNT=NT, GPF=pf))
    (cs,) = _S4K[key](inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
                      output_shapes=[(NP, 16, N)], output_dtypes=[mx.float32])
    (y,) = _ACCUM(inputs=[cs], template=[("T", BF), ("N", N), ("NP", NP)], grid=(16 * N, 1, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(1, 16, N)], output_dtypes=[BF])
    return y

def s4_partials(a, w, pf=4):
    """s4_gemm's first kernel alone: the split-K partials cs [NP, 16, N] (fp32) and NP"""
    N, K = w.shape
    PS = 4096
    NP, NT = (K + PS - 1) // PS, (N + 31) // 32
    key = (N, K, pf)
    if key not in _S4K:
        _S4K[key] = mx.fast.metal_kernel(
            name=f"leapk_s4_{N}_{K}_{pf}", input_names=["x", "w"], output_names=["cs"], source=_S4_SRC,
            header=_nax_header() + "\n" + consts(GN=N, GK=K, GPS=PS, GNP=NP, GNT=NT, GPF=pf))
    (cs,) = _S4K[key](inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
                      output_shapes=[(NP, 16, N)], output_dtypes=[mx.float32])
    return cs, NP

# R-82 (exact-safe): _ACCUM + the HC read's gate1 in one kernel: the same fp32 partition sum in the same order, rounded to
# bf16, then gate1's table lookups on those bits (mix for the first LR columns, inject for the last HC)
_ACCUM_GATE = mx.fast.metal_kernel(
    name="leapk_accum_gate", input_names=["cs", "tab"], output_names=["mix", "inject"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= 16 * N) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * 16 * N + i];
    }
    const T yv = static_cast<T>(out);
    const uint r = i / N, c = i % N;
    if (r >= ROWS) return;
    if (c < LR) mix[r * LR + c] = tab[as_type<ushort>(yv)];
    else inject[r * HC + (c - LR)] = tab[65536 + as_type<ushort>(yv)];
""")

# D1-B (exact-safe, Mon 2026-11-02; d1b_screen.py's B4p2v): _ACCUM_GATE in the prologue of a bitwise replica of MLX's
# K = LR bf16 up GEMM (steel_gemm_fused_nax: ten 32-wide k steps of two 16 x 32 x 16 NAX ops per output), the tail in its
# epilogue. Simdgroup (sub, s) computes the 32-aligned column run s * D + ch0 of up, the run MLX gives one simdgroup, so
# every output keeps its place in its op; the four streams of a channel meet in threadgroup memory.
_KB_SRC = """
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
_KBK = {}
def kernel_b_ok(hc, width, lowrank, n_part, cpt=4):
    """the shapes kernel B is written for: 4 * cpt simdgroups of 32 channels, K = lowrank in 32-wide steps, float4 rows"""
    return (hc * cpt <= 32 and 16 % hc == 0 and width % (32 * cpt) == 0 and lowrank % 32 == 0 and (lowrank + hc) % 4 == 0
            and n_part >= 1 and 2 * (16 * (lowrank + 8) + cpt * hc * 512) <= 32768)

def hc_kernel_b(cs, tab, wu, normed, hc, width, lowrank, cpt=4, gpf=2):
    """D1-B: cs [NP, 16, lowrank + hc] fp32 S4 partials, tab (gate tables), wu [hc * width, lowrank] bf16, normed
    [16, hc * width] bf16 -> (mixed [16, width], inject [16, hc]), bitwise == _ACCUM_GATE + MLX's up matmul + tail"""
    NP = int(cs.shape[0])
    key = (hc, width, lowrank, NP, cpt, gpf)
    if key not in _KBK:
        _KBK[key] = mx.fast.metal_kernel(
            name=f"leapk_hcb_{hc}_{width}_{lowrank}_{NP}_{cpt}_{gpf}", input_names=["cs", "tab", "wu", "normed"],
            output_names=["mixed", "inject"], source=_KB_SRC,
            header=_nax_header() + "\n" + consts(LR=lowrank, HC=hc, D=width, N=lowrank + hc, NP=NP, LDA=lowrank + 8,
                                                 CPT=cpt, GPF=gpf, NTH=32 * hc * cpt, ITERS=lowrank // 32,
                                                 NO=(16 * 32 * cpt) // (32 * hc * cpt)))
    return _KBK[key](inputs=[cs, tab, wu, normed], template=[("T", BF)],
                     grid=(32 * hc * cpt * (width // (32 * cpt)), 1, 1), threadgroup=(32 * hc * cpt, 1, 1),
                     output_shapes=[(16, width), (16, hc)], output_dtypes=[BF, BF])

# ---------------------------------------------------------------- the HC read around it (R-09's X5 pieces)
_HC = {}
def _hc_kernels(hc, width, lowrank, eps):
    key = (hc, width, lowrank, eps)
    if key not in _HC:
        from mtplx.kernels import hc_verify_read as HV
        from mtplx.models import qwen4_exp as Q
        vals = mx.view(mx.arange(1 << 16, dtype=mx.uint32).astype(mx.uint16), BF)
        tab = mx.concatenate([nn.silu(vals / hc), Q._verify_inject(vals, hc), mx.sigmoid(vals)])
        mx.eval(tab)
        gate1 = mx.fast.metal_kernel(
            name=f"leapk_gate1_{hc}_{lowrank}", input_names=["both", "tab"], output_names=["mix", "inject"],
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
        tail = mx.fast.metal_kernel(
            name=f"leapk_tail_{hc}_{width}", input_names=["up", "normed", "tab"], output_names=["mixed"],
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
        _HC[key] = dict(tab=tab, gate1=gate1, tail=tail, norm_p=HV._norm_kernel(hc, width, lowrank, eps, True),
                        norm_n=HV._norm_kernel(hc, width, lowrank, eps, False))
    return _HC[key]

def hc_read(g, wcat, hyper, pending=None, rows=16, fuse_gate=False, kernel_b=False):
    """GatedResidual's 16-row read in the decode_verify phase, as R-09b's S4 arm: returns (mixed, written, inject)
    shaped like the engine's ([*lead, D], hyper's shape, [*lead, HC]). fuse_gate (R-82, exact-safe): the S4 partition
    sum and gate1 as one kernel (bitwise the same mix / inject). kernel_b (D1-B, exact-safe): the partition sum + gate,
    the up GEMM and the tail as one kernel (bitwise the same mixed / inject); it takes precedence over fuse_gate"""
    hc, width = g.hc_count, g.hidden_size
    lowrank = int(g.input_mix_weight_down.weight.shape[0])
    k = _hc_kernels(hc, width, lowrank, float(g.hc_norm.eps))
    lead = hyper.shape[:-1]
    hcd = hc * width
    x = hyper.reshape(rows, hcd)
    if pending is not None:
        bo, inj = pending[0].reshape(rows, width), pending[1].reshape(rows, hc)
        written, normed = k["norm_p"](inputs=[x, bo, inj, g.hc_norm.weight], template=[("T", BF)],
                                      grid=(rows * hc * 256, 1, 1), threadgroup=(256, 1, 1),
                                      output_shapes=[(rows, hcd), (rows, hcd)], output_dtypes=[BF, BF])
        written = written.reshape(hyper.shape)
    else:
        (normed,) = k["norm_n"](inputs=[x, x, x, g.hc_norm.weight], template=[("T", BF)],
                                grid=(rows * hc * 256, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(rows, hcd)], output_dtypes=[BF])
        written = hyper
    if kernel_b and rows == 16:
        cs, NP = s4_partials(normed.reshape(1, rows, hcd), wcat)
        if int(wcat.shape[0]) == lowrank + hc and kernel_b_ok(hc, width, lowrank, NP):
            mixed, inject = hc_kernel_b(cs, k["tab"], g.input_mix_weight_up.weight, normed, hc, width, lowrank)
            return mixed.reshape(*lead, width), written, inject.reshape(*lead, hc)
    if fuse_gate:
        cs, NP = s4_partials(normed.reshape(1, rows, hcd), wcat)
        N = int(wcat.shape[0])
        assert N == lowrank + hc, (N, lowrank, hc)
        mix, inject = _ACCUM_GATE(inputs=[cs, k["tab"]], template=[("T", BF), ("N", N), ("NP", NP), ("ROWS", rows),
                                  ("LR", lowrank), ("HC", hc)], grid=(16 * N, 1, 1), threadgroup=(256, 1, 1),
                                  output_shapes=[(rows, lowrank), (rows, hc)], output_dtypes=[BF, BF])
    else:
        both = s4_gemm(normed.reshape(1, rows, hcd), wcat)
        tc = [("NDN", rows * lowrank), ("NIJ", rows * hc), ("LR", lowrank), ("HC", hc)]
        mix, inject = k["gate1"](inputs=[both, k["tab"]], template=tc, grid=(rows * (lowrank + hc), 1, 1),
                                 threadgroup=(256, 1, 1), output_shapes=[(rows, lowrank), (rows, hc)],
                                 output_dtypes=[BF, BF])
    up = mix.reshape(1, rows, lowrank) @ g.input_mix_weight_up.weight.T
    (mixed,) = k["tail"](inputs=[up, normed, k["tab"]], template=[("T", BF), ("D", width), ("HC", hc),
                         ("NOUT", rows * width)], grid=(rows * width, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(rows, width)], output_dtypes=[BF])
    return mixed.reshape(*lead, width), written, inject.reshape(*lead, hc)

# ---------------------------------------------------------------- D2: MLX's split-K qmm at 16-row tiles + mx.sum
_QH = None
def _q_header():
    global _QH
    if _QH is None:
        QH = (INC / "mlx/backend/metal/kernels/quantized.h").read_text().splitlines()
        i0 = next(i for i, l in enumerate(QH) if l.startswith("#define MLX_MTL_CONST"))
        i_impl = next(i for i, l in enumerate(QH) if l.startswith("METAL_FUNC void qmm_t_impl("))
        depth, i1, seen = 0, None, False
        for i in range(i_impl, len(QH)):
            depth += QH[i].count("{") - QH[i].count("}")
            seen |= "{" in QH[i]
            if seen and depth == 0:
                i1 = i + 1
                break
        lib = [l for l in QH[i0:i1] if "function_constant" not in l]
        _QH = inline("mlx/backend/metal/kernels/steel/gemm/gemm.h") + "\nusing namespace metal;\n" + "\n".join(lib) + "\n"
    return _QH

_D2_SRC = """
    constexpr int BK_padded = (32 + 16 / sizeof(T));
    threadgroup T Xs[GBM * BK_padded];
    threadgroup T Ws[32 * BK_padded];
    const uint3 tid = threadgroup_position_in_grid;
    const int k_start = tid.z * GKP;
    constexpr int pack_factor = get_pack_factor<4, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<4>();
    auto wl = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor;
    qmm_t_impl<T, 64, 4, true, GBM, 32, 32>((const device uint32_t*)wl, scales + k_start / 64, biases + k_start / 64,
        x + k_start, y + tid.z * (GM * GN), Xs, Ws, GK, GN, GM, GKP, tid, thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
_D2K = {}
def split_k_of(M, N, K):
    """MLX 0.32.2 qmm_splitk's partition count"""
    split = max(1, 512 // (((N + 31) // 32) * ((M + 31) // 32)))
    split = min(split, K // 64)
    while split > 1 and K % (split * 64) != 0:
        split -= 1
    return split

def routes_splitk(M, N, K):
    """would MLX 0.32.2 (gen 17, non-'d') serve a 4-bit / group-64 x @ w.T of M rows with split-K?"""
    limit = 33 if (K <= 2048 and N <= 2048) else 25 if (K <= 4096 and N <= 4096) else 13
    return M >= limit and split_k_of(M, N, K) > 1 and N % 32 == 0 and K % 64 == 0

def splitk_d2(x, W, bm=16):
    """x [M, K] bf16, W = (w, scales, biases) 4-bit / group 64 -> [M, N]: MLX's split-K sums, BM-row tiles"""
    w, s, b = W
    M, K = x.shape
    N = w.shape[0]
    split = split_k_of(M, N, K)
    key = (N, K, M, split, bm)
    if key not in _D2K:
        _D2K[key] = mx.fast.metal_kernel(
            name=f"leapk_d2_{N}_{K}_{M}_{split}_{bm}", input_names=["w", "scales", "biases", "x"], output_names=["y"],
            header=_q_header() + consts(GBM=bm, GN=N, GK=K, GM=M, GKP=K // split), source=_D2_SRC)
    (part,) = _D2K[key](inputs=[w, s, b, x], template=[("T", BF)],
                        grid=(32 * ((N + 31) // 32), 2 * ((M + bm - 1) // bm), 2 * split), threadgroup=(32, 2, 2),
                        output_shapes=[(split, M, N)], output_dtypes=[BF])
    return mx.sum(part, axis=0)
