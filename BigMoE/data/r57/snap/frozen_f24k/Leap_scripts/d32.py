"""D32 (NewIdeas Sun 23:41; Leap, Mon 2026-10-26, pre-registered in ~/Leap/TABLE_350.md): the dense part of an exact
check at 32 rows vs 16 rows. Block 32 at the oracle's 15.0 tokens per check cuts expert reads per token only ~4 % (par32),
but dense per token ~37 % IF the dense part at 32 rows stays near its 16-row cost.
Every 16-row path of today's ~18 ms dense set is batch-invariant (L1-23 / L1-26 / L1-29). Per shape (the 125B's dense
shapes and per-check counts, MoonShots' data/b8mr/denseinv.txt; random weights, cold), interleaved paired timing of:
  16 rows: qmm16n (native-layout 4-bit, L1-26) / bmm16 (bf16, L1-29) / MLX's own call (forms a, b, c for bf16);
  32 rows: '2x16'  the 16-row path called twice (exact by construction);
           'mg2'   one launch, the two 16-row tiles in adjacent threadgroups (weights read once from DRAM, unpacked twice);
           'mr2'   one simdgroup computes both 16-row tiles of its 32 columns (weights read and unpacked once, two MMAs
                   per step); same per-tile arithmetic as the 16-row kernel, so meant to be bit-identical per row;
           MLX's own 32-row call (valid only if bitwise equal to its 16-row call per row and padding-independent).
Bitwise, on one weight instance per shape: every 32-row path == the 16-row path (same ks) on rows 0-15 / 16-31, and real
rows 17 / 24 / 31 independent of the padding (zeros vs random rows). Invalid paths are reported and not used.
Totals: best exact 16-row set vs best exact 32-row set, ms per check and the ratio R; dense ms per token at block 16
(9.38 tokens per check) vs block 32 (oracle 15.0; L1-20's trained block-32 drafter 10.51).
Usage: [BigMoE venv] d32.py [smoke] -> ~/Leap/data/d32.txt, d32.json (smoke: 2 small shapes, compile + bitwise only)"""
import os, sys, time, json
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
from qmm16 import _HDR
from qmm26 import qmm16n
from bmm30 import bmm16

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")              # quiet research mode (RULES.md): heat pause
QUIET = os.path.exists(os.path.expanduser("~/.local/share/quiet/FAN_CAP"))
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

# ---- kernels: %%{ ... }%% blocks are repeated once per 16-row M tile of the simdgroup (MR), with @ = the tile index.
# Threadgroup x index = (column-tile group) * MG + mg: MG adjacent threadgroups share the same weights (one DRAM read).
_SRC_Q4 = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int mg = threadgroup_position_in_grid.x % MG;
    const int t = (threadgroup_position_in_grid.x / MG) * NT + nt;   // 32-column tile
    const int n0 = t * 32;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    %%{
    auto ct_c@ = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    float acc@[16];
    for (short i = 0; i < 16; i++) acc@[i] = 0.0f;
    const device T* xr0_@ = x + (size_t)((mg * MR + @) * 16 + fm) * K_C + 16 * fq;
    const device T* xr1_@ = xr0_@ + 8 * K_C;
    }%%
    constexpr int KG = K_C / G;
    const int kper = K_C / KS;
    // native weights (qmm26.repack): [t][g][lane][8 words]; scales / biases: [t][g][32]
    const device uint4* wq = (const device uint4*)(wn + (((size_t)t * KG) * 32 + lane) * 8);
    const device T* sc_ = sn + ((size_t)t * KG) * 32 + 4 * fq;
    const device T* bi_ = bn + ((size_t)t * KG) * 32 + 4 * fq;
    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        const int g = k0 / G;
        const uint4 q0 = wq[(size_t)g * 64];
        const uint4 q1 = wq[(size_t)g * 64 + 1];
        const uint wv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        %%{
        for (short i = 0; i < 16; i++) ct_c@[i] = 0.0f;
        float xs0_@ = 0.0f, xs1_@ = 0.0f;
        }%%
        for (short st = 0; st < 4; st++) {
            for (short p = 0; p < 2; p++) {
                const uint word = wv[st * 2 + p];
                for (short j = 0; j < 4; j++) {
                    const uint two = ((word >> (4 * j)) & 0x000F000Fu) | 0x43004300u;
                    ct_b[p * 8 + j] = as_type<T>((ushort)(two & 0xFFFFu));
                    ct_b[p * 8 + 4 + j] = as_type<T>((ushort)(two >> 16));
                }
            }
            %%{
            for (short j = 0; j < 4; j++) {
                const T a0 = xr0_@[k0 + 4 * st + j], a1 = xr1_@[k0 + 4 * st + j];
                ct_a[j] = a0; ct_a[4 + j] = a1;
                xs0_@ += static_cast<float>(a0); xs1_@ += static_cast<float>(a1);
            }
            op.run(ct_a, ct_b, ct_c@);
            }%%
        }
        %%{
        xs0_@ += simd_shuffle_xor(xs0_@, 1); xs0_@ += simd_shuffle_xor(xs0_@, 8);
        xs1_@ += simd_shuffle_xor(xs1_@, 1); xs1_@ += simd_shuffle_xor(xs1_@, 8);
        }%%
        const device T* scg = sc_ + (size_t)g * 32;
        const device T* big = bi_ + (size_t)g * 32;
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const float s_ = static_cast<float>(scg[16 * bk + j]);
                const float b_ = static_cast<float>(big[16 * bk + j]) - 128.0f * s_;
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                %%{
                acc@[i0] = acc@[i0] + (s_ * ct_c@[i0] + b_ * xs0_@);
                acc@[i1] = acc@[i1] + (s_ * ct_c@[i1] + b_ * xs1_@);
                }%%
            }
    }
"""

_SRC_BF = r"""
    constexpr int G = 64;
    const uint lane = thread_index_in_simdgroup;
    const uint kg = simdgroup_index_in_threadgroup / NT;
    const uint nt = simdgroup_index_in_threadgroup % NT;
    const int mg = threadgroup_position_in_grid.x % MG;
    const int n0 = ((threadgroup_position_in_grid.x / MG) * NT + nt) * 32;
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fq = ((qid & 2) | (lane & 1));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
    %%{
    auto ct_c@ = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_a)>,
                                                                metal::remove_addrspace_t<decltype(ct_b)>, float>();
    for (short i = 0; i < 16; i++) ct_c@[i] = 0.0f;
    const device T* xr0_@ = x + (size_t)((mg * MR + @) * 16 + fm) * K_C + 16 * fq;
    const device T* xr1_@ = xr0_@ + 8 * K_C;
    }%%
    const int kper = K_C / KS;
    const device T* wr[4];
    for (short r = 0; r < 4; r++) wr[r] = w + (size_t)(n0 + (r >> 1) * 16 + fm + (r & 1) * 8) * K_C + 16 * fq;
    for (int k0 = kg * kper; k0 < (int)(kg + 1) * kper; k0 += G) {
        for (short st = 0; st < 4; st++) {
            for (short r = 0; r < 4; r++) {
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                const device T* wp = wr[r] + k0 + 4 * st;
                for (short j = 0; j < 4; j++) ct_b[base + j] = wp[j];
            }
            %%{
            for (short j = 0; j < 4; j++) {
                ct_a[j] = xr0_@[k0 + 4 * st + j];
                ct_a[4 + j] = xr1_@[k0 + 4 * st + j];
            }
            op.run(ct_a, ct_b, ct_c@);
            }%%
        }
    }
    %%{
    float acc@[16];
    for (short i = 0; i < 16; i++) acc@[i] = ct_c@[i];
    }%%
"""

# K-split reduction (fixed order: group 0 + 1 + ...; one M tile at a time through the same buffer) and the store
_TAIL = r"""
    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];
    %%{
    if (KS > 1) {
        if (kg > 0) for (short i = 0; i < 16; i++) part[(((kg - 1) * NT + nt) * 16 + i) * 32 + lane] = acc@[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kg == 0)
            for (int gg = 0; gg < KS - 1; gg++)
                for (short i = 0; i < 16; i++) acc@[i] = acc@[i] + part[((gg * NT + nt) * 16 + i) * 32 + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    }%%
    if (kg != 0) return;
    %%{
    for (short i = 0; i < 16; i++) {
        const int m = (mg * MR + @) * 16 + fm + 8 * ((i & 7) >> 2);
        const int n = n0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        y[(size_t)m * N_C + n] = static_cast<T>(acc@[i]);
    }
    }%%
"""


def _gen(src, mr):
    out, rest = [], src
    while "%%{" in rest:
        a, rest = rest.split("%%{", 1)
        blk, rest = rest.split("}%%", 1)
        out.append(a + "".join(blk.replace("@", str(m)) for m in range(mr)))
    return "".join(out) + rest


@lru_cache(None)
def _kern(kind, mr):
    src = _gen((_SRC_Q4 if kind == "q4" else _SRC_BF) + _TAIL, mr)
    names = ["x", "wn", "sn", "bn"] if kind == "q4" else ["x", "w"]
    return mx.fast.metal_kernel(name=f"leap_d32_{kind}_mr{mr}", input_names=names, output_names=["y"], source=src,
                                header=_HDR)


def mm(kind, x, W, N, ks, nt, mr, mg):
    """x [16 * mr * mg, K] (all rows computed; callers pad) -> y [16 * mr * mg, N]"""
    R, K = x.shape
    assert R == 16 * mr * mg and N % (32 * nt) == 0 and K % (64 * ks) == 0
    (y,) = _kern(kind, mr)(inputs=[x, *W], template=[("T", x.dtype), ("K_C", K), ("N_C", N), ("KS", ks), ("NT", nt),
                                                     ("MR", mr), ("MG", mg)],
                          grid=(N // 32 * 32 * ks * mg, 1, 1), threadgroup=(32 * ks * nt, 1, 1),
                          output_shapes=[(R, N)], output_dtypes=[x.dtype])
    return y


def base16(kind, x, W, N, ks, nt):
    return qmm16n(x, *W, N, ks, nt) if kind == "q4" else bmm16(x, W[0], ks, nt)


# ---- helpers
def same(a, b):
    a, b = np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))
    return a.shape == b.shape and bool((a == b).all())


def t_once(fn, inner=5):
    mx.eval(fn())
    ts = []
    for _ in range(inner):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[inner // 2] * 1e3


def ab(fns, rounds=8, seed=0):
    """interleaved shuffled rounds (paired A/B); a round a heat pause overlapped (PAUSE, or a SIGSTOP jump) is redone"""
    rng = np.random.default_rng(seed)
    res = {k: [] for k in fns}
    for _ in range(rounds):
        for k in rng.permutation(list(fns)):
            while True:
                while os.path.exists(PAUSE):
                    time.sleep(0.5)
                v = t_once(fns[k])
                if not os.path.exists(PAUSE) and v < 500:
                    break
            res[k].append(v)
    return {k: np.array(v) for k, v in res.items()}


def cfgs(kind, N, K):
    c = [(1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (4, 2)]
    if kind == "bf16":
        c += [(5, 1), (10, 1), (5, 2), (8, 2)]
    return [(ks, nt) for ks, nt in c if K % (64 * ks) == 0 and N % (32 * nt) == 0]


# (name, kind, N, K, count per check)
SHAPES = [("in_proj_qkv", "q4", 10240, 2560, 36), ("o_proj+out_proj", "q4", 2560, 6144, 48),
          ("lm_head", "q4", 248320, 2560, 1), ("in_proj_z", "q4", 6144, 2560, 36), ("q_proj", "q4", 12288, 2560, 12),
          ("shared gate/up", "q4", 640, 2560, 96), ("shared down", "q4", 2560, 640, 48),
          ("k/v_proj", "q4", 512, 2560, 24), ("in_proj_a/b", "q4", 48, 2560, 72),
          ("shared_expert_gate", "q4", 1, 2560, 48), ("router gate", "q8", 512, 2560, 48),
          ("indexer qk", "q8", 640, 2560, 12), ("HC mix up", "bf16", 10240, 320, 97),
          ("HC mix down", "bf16", 320, 10240, 97), ("ple key", "bf16", 10240, 2560, 1),
          ("ple value", "bf16", 2560, 2560, 1), ("block_inject", "bf16", 4, 10240, 96)]
if SMOKE:
    SHAPES = [("shared down", "q4", 2560, 640, 4), ("HC mix down", "bf16", 320, 10240, 4)]


def rand_mlx_q(N, K, bits, rng):
    """random MLX affine weights (codes uniform; scales ~2e-3, biases ~ -8 scales)"""
    w = mx.array(rng.integers(0, 1 << 32, (N, K * bits // 32), dtype=np.uint32))
    s = rng.uniform(0.001, 0.003, (N, K // 64)).astype(np.float32)
    b = -(2 ** (bits - 1)) * s + rng.normal(0, 1e-3, s.shape).astype(np.float32)
    return (w, mx.array(s).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16))


def rand_native(N, K, rng):
    KG = K // 64
    wn = mx.array(rng.integers(0, 1 << 32, (N // 32 * KG * 32 * 8,), dtype=np.uint32))
    s = rng.uniform(0.001, 0.003, (N // 32, KG, 32)).astype(np.float32)
    b = -8 * s + rng.normal(0, 1e-3, s.shape).astype(np.float32)
    return (wn, mx.array(s).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16))


def pad_ok(f, X32, Xr, ref32):
    """real rows of 32-row calls padded with zeros / other random rows == the all-real call"""
    K = X32.shape[1]
    for M in (17, 24, 31):
        for pad in (mx.zeros((32 - M, K), X32.dtype), Xr[:32 - M]):
            if not same(f(mx.concatenate([X32[:M], pad]))[:M], ref32[:M]):
                return False
    return True


if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(32)
    mx.random.seed(32)
    say(f"D32 ({time.strftime('%a %H:%M')}): dense part of an exact check, 32 rows vs 16, MLX {mx.__version__}, "
        f"random weights at the 125B's shapes x per-check counts (cold); ms per check; 8 interleaved rounds x median of 5"
        + (" [QUIET MODE: fans capped; absolute times are quiet-mode numbers, compare arms]" if QUIET else "")
        + (" [SMOKE]" if SMOKE else ""))
    rows, out = [], {}
    tot = {"floor": 0.0, "t16": 0.0, "t32": 0.0, "x2": 0.0}
    for name, kind, N, K, cnt in SHAPES:
        ts = time.time()
        X32 = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
        Xr = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
        X16a, X16b = mx.contiguous(X32[:16]), mx.contiguous(X32[16:])
        mx.eval(X32, Xr, X16a, X16b)
        arms16, arms32, info, notes = {}, {}, {}, []
        # MLX family
        if kind in ("q4", "q8"):
            bits = 4 if kind == "q4" else 8
            Wm = [rand_mlx_q(N, K, bits, rng) for _ in range(cnt)]
            mx.eval(Wm)
            nbytes = sum(a.nbytes for a in Wm[0])
            forms = {"mlx": lambda x, W, bits=bits: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=bits)}
            fw = {"mlx": Wm}
        else:
            Wb = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(cnt)]
            WT = [mx.contiguous(W.T) for W in Wb]
            mx.eval(Wb, WT)
            nbytes = Wb[0].nbytes
            forms = {"mlx a": lambda x, W: x @ W.T, "mlx b": lambda x, W: (W @ x.T).T, "mlx c": lambda x, W: x @ W}
            fw = {"mlx a": Wb, "mlx b": Wb, "mlx c": WT}
        floor = cnt * nbytes / 280e9 * 1e3
        for fn_, f in forms.items():
            Ws = fw[fn_]
            r16 = mx.concatenate([f(X16a, Ws[0]), f(X16b, Ws[0])])
            r32 = f(X32, Ws[0])
            eq = same(r32, r16)
            pad = pad_ok(lambda x: f(x, Ws[0]), X32, Xr, r32)
            notes.append(f"{fn_} 32-row call {'= 2x16' if eq else '!= 2x16'}, {'pad-indep' if pad else 'NOT pad-indep'}")
            arms16[f"{fn_} 16"] = lambda f=f, Ws=Ws: [f(X16a, W) for W in Ws]
            arms32[f"{fn_} 2x16"] = lambda f=f, Ws=Ws: [(f(X16a, W), f(X16b, W)) for W in Ws]
            if eq and pad:
                arms32[f"{fn_} 32"] = lambda f=f, Ws=Ws: [f(X32, W) for W in Ws]
        # MMA family (fixed 16 x 32 tiles; batch-invariant by construction)
        mk = "q4" if kind == "q4" else ("bf16" if kind == "bf16" else None)
        if mk and N % 32 == 0:
            Wk = [rand_native(N, K, rng) for _ in range(cnt)] if mk == "q4" else [(W,) for W in Wb]
            mx.eval(Wk)
            C = cfgs(mk, N, K)
            # sweep the 16-row kernel
            sw16 = {c: t_once(lambda c=c: [base16(mk, X16a, W, N, *c) for W in Wk]) for c in C}
            c16 = min(sw16, key=sw16.get)
            arms16["mma 16"] = lambda c=c16: [base16(mk, X16a, W, N, *c) for W in Wk]
            arms32["mma 2x16"] = lambda c=c16: [(base16(mk, X16a, W, N, *c), base16(mk, X16b, W, N, *c)) for W in Wk]
            for mr, mg in ((2, 1), (1, 2)):
                tag = "mr2" if mr == 2 else "mg2"
                good, badt, err = {}, {}, []
                for c in C:
                    try:
                        y = mm(mk, X32, Wk[0], N, *c, mr, mg)
                        ref = mx.concatenate([base16(mk, X16a, Wk[0], N, *c), base16(mk, X16b, Wk[0], N, *c)])
                        tt = t_once(lambda c=c: [mm(mk, X32, W, N, *c, mr, mg) for W in Wk])
                        (good if same(y, ref) else badt)[c] = tt
                    except Exception as e:
                        err.append(c); notes.append(f"{tag} {c} error: {str(e).splitlines()[0][:160]}")
                if badt:
                    notes.append(f"{tag} NOT bitwise at {sorted(badt)}")
                    if not good:                                    # timed for information only, never in the totals
                        cx = min(badt, key=badt.get)
                        info[f"mma {tag} {cx} NOT EXACT"] = lambda c=cx, mr=mr, mg=mg: [mm(mk, X32, W, N, *c, mr, mg) for W in Wk]
                if good:
                    cb = min(good, key=good.get)
                    pad = pad_ok(lambda x, c=cb: mm(mk, x, Wk[0], N, *c, mr, mg), X32, Xr,
                                 mm(mk, X32, Wk[0], N, *cb, mr, mg))
                    notes.append(f"{tag} bitwise = 2x16 at {sorted(good)}; best {cb} {'pad-indep' if pad else 'NOT pad-indep'}")
                    if pad:
                        arms32[f"mma {tag}"] = lambda c=cb, mr=mr, mg=mg: [mm(mk, X32, W, N, *c, mr, mg) for W in Wk]
                        if cb[0] != c16[0]:                         # 16-row cost at the 32-row winner's K split
                            c16b = min((c for c in C if c[0] == cb[0]), key=lambda c: sw16[c])
                            arms16[f"mma 16 ks{cb[0]}"] = lambda c=c16b: [base16(mk, X16a, W, N, *c) for W in Wk]
            notes.append(f"mma 16 sweep best {c16} {sw16[c16]:.3f} ms")
        if SMOKE:
            say(f"{name}: " + "; ".join(notes) + f" ({time.time() - ts:.0f} s)")
            continue
        res = ab({**arms16, **arms32, **info})
        med = {k: float(np.median(v)) for k, v in res.items()}
        b16 = min(arms16, key=lambda k: med[k])
        b32 = min(arms32, key=lambda k: med[k])
        r = res[b32] / res[b16]
        twox = [k for k in arms32 if k.endswith("2x16") and k.rsplit(" ", 1)[0] == b16.rsplit(" ", 1)[0]]
        t2x = med[twox[0]] if twox else 2 * med[b16]
        tot["floor"] += floor; tot["t16"] += med[b16]; tot["t32"] += med[b32]; tot["x2"] += t2x
        out[name] = {"N": N, "K": K, "count": cnt, "floor": floor, "arms": med, "best16": b16, "best32": b32,
                     "ratio_median": float(np.median(r)), "ratio_iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
                     "notes": notes}
        say(f"{name:18s} {N:6d}x{K:5d} x{cnt:3d} | floor {floor:5.2f} | 16: {b16} {med[b16]:6.3f} ({floor / med[b16]:4.0%}) | "
            f"32: {b32} {med[b32]:6.3f} ({floor / med[b32]:4.0%}) | x{np.median(r):.2f} (IQR {np.percentile(r, 25):.2f}-"
            f"{np.percentile(r, 75):.2f}) | 2x16 {t2x:.3f} ({time.time() - ts:.0f} s)")
        say("    all arms: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(med.items(), key=lambda kv: kv[1])))
        say("    " + "; ".join(notes))
        del arms16, arms32, info, res
        for v in ("Wm", "Wb", "WT", "Wk", "fw", "forms", "Ws", "f", "r16", "r32", "y", "ref"):
            globals().pop(v, None)
        getattr(mx, "clear_cache", lambda: None)()
    if not SMOKE:
        R = tot["t32"] / tot["t16"]
        say(f"TOTAL dense per check: floor {tot['floor']:.1f} ms | best exact 16 rows {tot['t16']:.2f} ms | best exact 32 rows "
            f"{tot['t32']:.2f} ms | R = {R:.2f} (two 16-row calls: {tot['x2'] / tot['t16']:.2f})")
        for tag, tau32 in (("oracle 15.0", 15.0), ("L1-20 drafter 10.51", 10.51)):
            say(f"  dense ms per token: block 16 at 9.38 tokens {tot['t16'] / 9.38:.2f} | block 32 at {tag} "
                f"{tot['t32'] / tau32:.2f} ({tot['t32'] / tau32 / (tot['t16'] / 9.38) - 1:+.0%})")
        say("  D32 lines (pre-registered): PASS (dense stays near its 16-row cost) R <= 1.25; KILL R >= 1.60; between: alive, priced")
        json.dump({"shapes": out, "total": tot, "R": R}, open(os.path.expanduser("~/Leap/data/d32.json"), "w"), indent=1)
    say(f"done in {time.time() - t_start:.0f} s")
    open(os.path.expanduser(f"~/Leap/data/d32{'_smoke' if SMOKE else ''}.txt"), "w").write("\n".join(lines) + "\n")
