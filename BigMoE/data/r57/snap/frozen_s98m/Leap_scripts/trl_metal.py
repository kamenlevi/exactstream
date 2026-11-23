"""TRL Metal encoder (Leap, Wed 2026-11-04 20:2x; the orchestrator 20:0x, TRELLIS FULL-STORE prep): the exact-MSE bitshift-
trellis Viterbi of Fit12's C encoder (~/Fit12/scripts/fit_trellis_enc.c, built as fit_trellis.dylib; = Breakthrough's
trellis_enc.c + a worker count) on the GPU, bit for bit.
  Trellis: L = 12 bit state, k = 2 / 3 new bits per value, T = 256 values per 16 x 16 tile, free initial and final state,
  reconstruction lut[state] (Fit12's LUT3 = QTIP 3INST / its RMS). State s_t = (s_{t-1} << k | q_t) mod 4096.
  The C code's arithmetic, as compiled (otool, fit_trellis.dylib: no fadd at all, 4 fmadd + 5 fmla):
    t = 0:  cost[s] = d * d,               d = x_0 - lut[s]          (fsub, fmul)
    t > 0:  best[j] = min_h cost[h NJ + j]  (h ascending, strict <: the lowest h wins a tie), NJ = 4096 >> k
            cost[j K + q] = fma(d, d, best[j]), d = x_t - lut[j K + q]   (fsub, fmadd / fmla)
    end:    the lowest state among the minimal final costs; traceback through the stored h.
  This kernel does the same operations in the same order (explicit fma; Apple GPU fp32 sub / mul / fma are IEEE RNE;
  |LUT3| >= 2e-4, so no subnormal can arise). LUT3 has 3,726 distinct values in 4,096 states (ties are real): the tie
  rules are copied, so the STATES match too, not only the values.
Kernel: one threadgroup per tile, NJ threads (1,024 for k 2, 512 for k 3); costs in threadgroup memory (16 KB); each
step's k decision bits packed per simdgroup by simd_or into k words (device scratch: 64 KB per tile at k 2, 48 KB at k 3);
the traceback copies them back into threadgroup memory in chunks and walks them in thread 0.
API: viterbi(x [n, 256] float32, k) -> states [n, 256] uint16 (mx); recon(x, k) -> LUT3[states] (np.float64, = fit_b3.vit's
output for lut = LUT3).
usage: [BigMoE venv] trl_metal.py test [n] -> ~/Leap/data/trl_metal_test.txt (GPU, MEM < 1 GB; share lock)
       [BigMoE venv] trl_metal.py bench    -> ~/Leap/data/trl_metal_bench.txt"""
import ctypes, os, sys, time
import numpy as np
import mlx.core as mx

L, T = 12, 256
_s = np.arange(1 << L, dtype=np.uint64)
_x3 = (np.uint64(89226354) * _s + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
_m3 = ((_x3 & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)).astype(np.uint32)
_c3 = (_m3 & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32) + \
    (_m3 >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
LUT3 = np.ascontiguousarray(_c3 / np.sqrt((_c3 * _c3).mean()), np.float32)   # = fit_b3.LUT3 (same numpy expression)
RMS3 = float(np.sqrt((_c3 * _c3).mean()))                                    # inst3(s) / RMS3 = LUT3[s] (decode)

_SRC = r"""
    constexpr int S = 1 << 12, KK = 1 << KB, NJ = S >> KB, NG = NJ / 32, TT = 256;
    constexpr int CH = S / (NG * KB);                    // decision rows per traceback chunk (threadgroup copy)
    threadgroup float cost[S];
    threadgroup float xs[TT];
    threadgroup float rv[32];
    threadgroup uint ri[32];
    const uint tile = threadgroup_position_in_grid.x;
    const uint j = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint g = simdgroup_index_in_threadgroup;
    for (uint i = j; i < TT; i += NJ) xs[i] = x[tile * TT + i];
    float lv[KK], nc[KK];
    for (int q = 0; q < KK; ++q) lv[q] = lut[j * KK + q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const float x0 = xs[0];
        for (int q = 0; q < KK; ++q) { const float d = x0 - lv[q]; cost[j * KK + q] = d * d; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint* dd = dec + (size_t)tile * (TT - 1) * NG * KB;
    for (int t = 1; t < TT; ++t) {
        float b = cost[j];
        uint h = 0;
        for (int hh = 1; hh < KK; ++hh) {
            const float c = cost[hh * NJ + j];
            if (c < b) { b = c; h = hh; }
        }
        for (int p = 0; p < KB; ++p) {
            const uint w = simd_or(((h >> p) & 1u) << lane);
            if (lane == 0) dd[((t - 1) * NG + g) * KB + p] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float xv = xs[t];
        for (int q = 0; q < KK; ++q) { const float d = xv - lv[q]; nc[q] = fma(d, d, b); cost[j * KK + q] = nc[q]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // final state: the lowest state among the minimal costs (the C scan: strict <, s ascending)
    float bv = nc[0];
    uint bs = j * KK;
    for (int q = 1; q < KK; ++q) if (nc[q] < bv) { bv = nc[q]; bs = j * KK + q; }
    for (uint o = 16; o > 0; o >>= 1) {
        const float ov = simd_shuffle_xor(bv, o);
        const uint os = simd_shuffle_xor(bs, o);
        if (ov < bv || (ov == bv && os < bs)) { bv = ov; bs = os; }
    }
    if (lane == 0) { rv[g] = bv; ri[g] = bs; }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (g == 0) {
        bv = lane < (uint)NG ? rv[lane] : FLT_MAX;
        bs = lane < (uint)NG ? ri[lane] : 0xFFFFFFFFu;
        for (uint o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const uint os = simd_shuffle_xor(bs, o);
            if (ov < bv || (ov == bv && os < bs)) { bv = ov; bs = os; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) ri[0] = bs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint s = ri[0];
    device ushort* so = st + (size_t)tile * TT;
    if (j == 0) so[TT - 1] = (ushort)s;
    threadgroup uint* dt = reinterpret_cast<threadgroup uint*>(cost);
    for (int r1 = TT - 1; r1 > 0; ) {
        const int r0 = max(0, r1 - CH);
        const int nw = (r1 - r0) * NG * KB;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = j; i < nw; i += NJ) dt[i] = dd[r0 * NG * KB + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j == 0) {
            for (int r = r1 - 1; r >= r0; --r) {            // decision row r = step t = r + 1
                const uint jj = s >> KB;
                uint h = 0;
                for (int p = 0; p < KB; ++p) h |= ((dt[((r - r0) * NG + (jj >> 5)) * KB + p] >> (jj & 31u)) & 1u) << p;
                s = (h << (12 - KB)) | jj;
                so[r] = (ushort)s;
            }
        }
        r1 = r0;
    }
"""
_K = {}


def _kernel():
    if "v" not in _K:
        _K["v"] = mx.fast.metal_kernel(name="trl_viterbi", input_names=["x", "lut"], output_names=["st", "dec"],
                                       source=_SRC)
    return _K["v"]


_LUT_MX = {}


def viterbi(x, k):
    """x [n, 256] (mx or np, cast to float32) -> states [n, 256] uint16 (mx)"""
    x = mx.array(np.ascontiguousarray(x, np.float32)) if isinstance(x, np.ndarray) else x.astype(mx.float32)
    n = x.shape[0]
    if "l" not in _LUT_MX:
        _LUT_MX["l"] = mx.array(LUT3)
    nj = 4096 >> k
    st, _ = _kernel()(inputs=[x.reshape(n, T), _LUT_MX["l"]], template=[("KB", k)], grid=(n * nj, 1, 1),
                      threadgroup=(nj, 1, 1), output_shapes=[(n, T), (n * (T - 1) * (nj // 32) * k,)],
                      output_dtypes=[mx.uint16, mx.uint32], stream=mx.gpu)   # GPU even under a CPU default device
    return st


def recon(t, k, lut=None):
    """drop-in for fit_b3.vit(t, k, LUT3): t [n, 256] -> LUT3[Viterbi states] as float64"""
    assert lut is None or np.array_equal(np.asarray(lut, np.float32).view(np.uint32), LUT3.view(np.uint32)), "LUT3 only"
    st = viterbi(np.ascontiguousarray(t, np.float32), k)
    mx.eval(st)
    return LUT3[np.array(st).astype(np.int64)].astype(np.float64)


def pack_bits(st, k):
    """states [n, 256] -> the decode bitstream per tile (B10(b)'s format: s_0's 12 bits, then q_1..q_255, k bits each, MSB
    first) as uint32 words [n, ceil((12 + 255 k) / 32)]"""
    st = np.asarray(st).astype(np.uint32)
    n = st.shape[0]
    nb = 12 + (T - 1) * k
    bits = np.zeros((n, nb), np.uint8)
    for b in range(12):
        bits[:, b] = (st[:, 0] >> (11 - b)) & 1
    q = st[:, 1:] & ((1 << k) - 1)
    for b in range(k):
        bits[:, 12 + b::k][:, :T - 1] = (q >> (k - 1 - b)) & 1
    nw = (nb + 31) // 32
    bits = np.concatenate([bits, np.zeros((n, nw * 32 - nb), np.uint8)], 1)
    return np.packbits(bits, axis=1, bitorder="big").view(">u4").astype(np.uint32).reshape(n, nw)


def unpack_states(words, k):
    """inverse of pack_bits: words [n, nw] -> states [n, 256]"""
    words = np.asarray(words, np.uint32)
    bits = np.unpackbits(words.astype(">u4").view(np.uint8), axis=1, bitorder="big").astype(np.uint32)
    st = np.zeros((words.shape[0], T), np.uint32)
    for t in range(T):
        for b in range(12):
            st[:, t] = (st[:, t] << 1) | bits[:, k * t + b]
    return st


def _clib(path):
    lib = ctypes.CDLL(path)
    lib.viterbi_tiles_n.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
    return lib


def c_recon(lib, t, k, nthr=4):
    x = np.ascontiguousarray(t, np.float32)
    out = np.empty_like(x)
    lib.viterbi_tiles_n(x.ctypes.data, out.ctypes.data, x.shape[0], x.shape[1], L, k, LUT3.ctypes.data, nthr)
    return out


def _say(path):
    def say(s=""):
        print(s, flush=True)
        open(path, "a").write(s + "\n")
    return say


def test(n=384):
    say = _say(os.path.expanduser("~/Leap/data/trl_metal_test.txt"))
    say(f"\nTRL Metal Viterbi test {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}; C reference = Fit12's "
        f"fit_trellis.dylib (read only)")
    lib = _clib(os.path.expanduser("~/Fit12/scripts/fit_trellis.dylib"))
    sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
    try:
        import fit_b3 as B
        say(f"LUT3 == fit_b3.LUT3 bitwise: {np.array_equal(B.LUT3.view(np.uint32), LUT3.view(np.uint32))}")
    except Exception as ex:                                         # fit_b3 imports fit_q (records3 reader): report only
        say(f"fit_b3 import failed ({ex!r}); LUT3 compared through the C output only")
    rng = np.random.default_rng(0)
    cases = {
        "gauss": rng.standard_normal((n, T)).astype(np.float32),
        "laplace": rng.laplace(size=(n, T)).astype(np.float32) / np.sqrt(2),
        "scaled 1e-3": (rng.standard_normal((n // 4, T)) * 1e-3).astype(np.float32),
        "zeros": np.zeros((8, T), np.float32),                    # every path ties: the tie rules decide everything
        "lut values": LUT3[rng.integers(0, 4096, (n // 4, T))],   # exact zero-cost paths (and duplicate-value ties)
        "constant": np.repeat(rng.standard_normal((16, 1)).astype(np.float32), T, 1),
    }
    ok = True
    for k in (2, 3):
        for nm, xt in cases.items():
            t0 = time.time()
            st = viterbi(xt, k)
            mx.eval(st)
            tg = time.time() - t0
            stn = np.array(st).astype(np.int64)
            rc = c_recon(lib, xt, k)
            eq = np.array_equal(LUT3[stn].view(np.uint32), rc.view(np.uint32))
            trel = bool(np.all(((stn[:, :-1] << k) & 4095) == (stn[:, 1:] & ~((1 << k) - 1) & 4095)))
            mse = float(((LUT3[stn] - xt) ** 2).mean() / max(float((xt * xt).mean()), 1e-30))
            rt = np.array_equal(unpack_states(pack_bits(stn, k), k), stn)
            ok &= eq and trel and rt
            say(f"k {k} {nm:12s} n {xt.shape[0]:4d}: recon == C bitwise {eq} (tiles differing "
                f"{int((LUT3[stn] != rc).any(1).sum())}); valid trellis paths {trel}; NMSE {mse:.4f}; bitstream round "
                f"trip {rt}; GPU {tg * 1e3:.0f} ms")
    say(f"TRL Metal Viterbi test: {'PASS' if ok else 'FAIL'}")
    return ok


def bench():
    say = _say(os.path.expanduser("~/Leap/data/trl_metal_bench.txt"))
    say(f"\nTRL Metal Viterbi bench {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
    rng = np.random.default_rng(1)
    for k in (2, 3):
        for n in (40, 160, 640, 2560):
            x = mx.array(rng.standard_normal((n, T)).astype(np.float32))
            mx.eval(viterbi(x, k))
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                mx.eval(viterbi(x, k))
                ts.append(time.perf_counter() - t0)
            m = float(np.median(ts))
            say(f"k {k} n {n:5d}: {m * 1e3:.2f} ms per call, {n / m:,.0f} tiles/s, {m / n * 1e6:.2f} us per tile")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        sys.exit(0 if test(int(sys.argv[2]) if len(sys.argv) > 2 else 384) else 1)
    bench()
