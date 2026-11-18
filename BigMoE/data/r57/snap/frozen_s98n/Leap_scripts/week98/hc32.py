"""hc32.py (Leap, Sun 15 Nov 2026 01:3x; WEEK98 pillar 2, build step 4 of the 17-32-row path): the HC read at 17-32 rows
in ONE pass over its weights, bitwise per row with the 16-row read.
The 16-row read (leapk.hc_read with CFG hc_gate, R-82; kernel B, D1-B, is bitwise the same chain) runs MLX's NAX split-K
GEMM on [16, HC x D] x wcat^T in 4,096-wide K partitions (s4_partials: one simdgroup per 32-column tile per partition, a
16 x 32 NAX accumulator) and sums the partitions in a fixed order (_ACCUM_GATE). Here:
  - S4x2: the same kernel with a SECOND 16-row accumulator sharing every B (weight) tile load: rows 0-15 and 16-31 each get
    exactly the 16-row kernel's sequence of NAX ops on the same weight tiles (rows of an MMA do not mix), and the weights
    are read once instead of twice;
  - ACCUM_GATE32: _ACCUM_GATE with a 32-row stride, the same per-element partition order;
  - the up GEMM at 32 rows is MLX's own call (== 2 x 16 per row for this shape, D32 / C2 'HC mix up'), the norm and tail
    kernels are row-generic.
Rows 17-31 are zero-padded to 32 (rows are independent). install(v32, ref2) after v32.install(): GatedResidual's hook
becomes _gr32, which hands every call it does not take back to v32's 2 x 16 split. Taken only on ref2's leapk path:
bf16 mixers with a block_inject (the _WCAT concat), CFG hc_gate on, no HC8, no pending carry."""
import mlx.core as mx
import leapk as LK

R, HI = 16, 32
BF = mx.bfloat16
CNT = dict(hc32=0, hc32_declined=0)
_S = {}

_S4X2_SRC = """
    using namespace mlx::steel;
    const int tg = int(threadgroup_position_in_grid.x);
    const int tile = tg % GNT, part = tg / GNT;
    const int c_col = tile * 32;
    const short sgp_sn = short(min(32, GN - c_col));
    const int k0 = part * GPS, k1 = min(k0 + GPS, GK);
    const int iters = (k1 - k0) / 32;
    const device T* Ap = x + k0;
    const device T* Ap2 = x + size_t(16) * GK + k0;
    const device T* Bp = w + size_t(c_col) * GK + k0;
    NAXTile<float, 1, 2> Dt;
    NAXTile<float, 1, 2> Dt2;
    const bool full_n = sgp_sn == 32;
    NAXTile<T, 1, 2> As[GPF];
    NAXTile<T, 1, 2> As2[GPF];
    NAXTile<T, 2, 2> Bs[GPF];
    auto ld = [&](thread NAXTile<T, 1, 2>& a_, thread NAXTile<T, 1, 2>& a2_, thread NAXTile<T, 2, 2>& b_, int s) {
        a_.load(Ap + s * 32, GK);
        a2_.load(Ap2 + s * 32, GK);
        if (full_n) {
            b_.load(Bp + s * 32, GK);
        } else {
            b_.load_safe(Bp + s * 32, GK, short2(32, sgp_sn));
        }
    };
    Dt.clear();
    Dt2.clear();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < GPF - 1; j++) {
        if (j < iters) ld(As[j], As2[j], Bs[j], j);
    }
    for (int s0 = 0; s0 < iters; s0 += GPF) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < GPF; j++) {
            const int s = s0 + j;
            if (s < iters) {
                const int nx = s + GPF - 1;
                if (nx < iters) ld(As[(j + GPF - 1) % GPF], As2[(j + GPF - 1) % GPF], Bs[(j + GPF - 1) % GPF], nx);
                tile_matmad_nax(Dt, As[j], metal::bool_constant<false>{}, Bs[j], metal::bool_constant<true>{});
                tile_matmad_nax(Dt2, As2[j], metal::bool_constant<false>{}, Bs[j], metal::bool_constant<true>{});
            }
        }
    }
    device float* Cp = cs + size_t(part) * 32 * GN + c_col;
    device float* Cp2 = Cp + size_t(16) * GN;
    if (full_n) {
        Dt.store(Cp, GN);
        Dt2.store(Cp2, GN);
    } else {
        Dt.store_safe(Cp, GN, short2(sgp_sn, 16));
        Dt2.store_safe(Cp2, GN, short2(sgp_sn, 16));
    }
"""
_K = {}


def s4x2_partials(a, w, pf=4):
    """[1, 32, K] x [N, K]^T -> split-K partials cs [NP, 32, N] (fp32): leapk.s4_partials per 16-row half, weights read once"""
    N, K = w.shape
    PS = 4096
    NP, NT = (K + PS - 1) // PS, (N + 31) // 32
    key = ("s4x2", N, K, pf)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name=f"leap_hc32_s4x2_{N}_{K}_{pf}", input_names=["x", "w"], output_names=["cs"], source=_S4X2_SRC,
            header=LK._nax_header() + "\n" + LK.consts(GN=N, GK=K, GPS=PS, GNP=NP, GNT=NT, GPF=pf))
    (cs,) = _K[key](inputs=[a, w], template=[("T", BF)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
                    output_shapes=[(NP, 32, N)], output_dtypes=[mx.float32])
    return cs, NP


_ACCUM_GATE32 = mx.fast.metal_kernel(
    name="leap_hc32_accum_gate", input_names=["cs", "tab"], output_names=["mix", "inject"],
    source="""
    const uint i = thread_position_in_grid.x;
    if (i >= 32 * N) return;
    float out = 0;
    for (int p = 0; p < NP; p++) {
        out += cs[size_t(p) * 32 * N + i];
    }
    const T yv = static_cast<T>(out);
    const uint r = i / N, c = i % N;
    if (r >= ROWS) return;
    if (c < LR) mix[r * LR + c] = tab[as_type<ushort>(yv)];
    else inject[r * HC + (c - LR)] = tab[65536 + as_type<ushort>(yv)];
""")


def hc_read32(g, wcat, hyper, pending=None):
    """leapk.hc_read's fuse_gate path at 32 rows (hyper [1, 32, HC x D]); returns (mixed, written, inject)"""
    rows = HI
    hc, width = g.hc_count, g.hidden_size
    lowrank = int(g.input_mix_weight_down.weight.shape[0])
    k = LK._hc_kernels(hc, width, lowrank, float(g.hc_norm.eps))
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
    cs, NP = s4x2_partials(normed.reshape(1, rows, hcd), wcat)
    N = int(wcat.shape[0])
    assert N == lowrank + hc, (N, lowrank, hc)
    mix, inject = _ACCUM_GATE32(inputs=[cs, k["tab"]], template=[("T", BF), ("N", N), ("NP", NP), ("ROWS", rows),
                                ("LR", lowrank), ("HC", hc)], grid=(32 * N, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(rows, lowrank), (rows, hc)], output_dtypes=[BF, BF])
    up = mix.reshape(1, rows, lowrank) @ g.input_mix_weight_up.weight.T
    (mixed,) = k["tail"](inputs=[up, normed, k["tab"]], template=[("T", BF), ("D", width), ("HC", hc),
                         ("NOUT", rows * width)], grid=(rows * width, 1, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(rows, width)], output_dtypes=[BF])
    return mixed.reshape(*lead, width), written, inject.reshape(*lead, hc)


def _pad32(x2):
    m = x2.shape[0]
    return x2 if m == HI else mx.concatenate([x2, mx.zeros((HI - m,) + x2.shape[1:], x2.dtype)], axis=0)


def _gr32(self, hyper_input, pending=None):
    V, RF = _S["v32"], _S["ref2"]
    rows = V._rows(hyper_input)
    if not V._live(rows):
        return V._gr(self, hyper_input, pending)
    ok = (RF.CFG["hc_gate"] and not RF.CFG["hc8_fast"] and not RF.CFG["hc_pend_mlp"] and "block_inject_weight" in self
          and hyper_input.dtype == BF and (pending is None or all(t.dtype == BF for t in pending))
          and RF._hc8_params(self) is None)
    if not ok:
        CNT["hc32_declined"] += 1
        return V._gr(self, hyper_input, pending)
    lead = hyper_input.shape[:-1]
    h32 = _pad32(hyper_input.reshape(rows, -1)).reshape(1, HI, -1)
    p32 = None if pending is None else tuple(_pad32(p.reshape(rows, -1)).reshape(1, HI, -1) for p in pending)
    if id(self) not in RF._WCAT:
        RF._WCAT[id(self)] = mx.concatenate([self.input_mix_weight_down.weight, self.block_inject_weight.weight], axis=0)
    mixed, written, inject = hc_read32(self, RF._WCAT[id(self)], h32, p32)
    CNT["hc32"] += 1
    RF.COUNTS["hc_read"] += 1
    return (mixed[0, :rows].reshape(*lead, -1), written[0, :rows].reshape(hyper_input.shape),
            inject[0, :rows].reshape(*lead, -1))


def install(v32, ref2):
    """after v32.install(ref2): GatedResidual's hook becomes _gr32 (which falls back to v32._gr)"""
    import mtplx.models.qwen4_exp as Q
    if not v32._W:
        raise RuntimeError("hc32: v32.install(ref2) first")
    _S.update(v32=v32, ref2=ref2)
    Q.GatedResidual.__call__ = _gr32
