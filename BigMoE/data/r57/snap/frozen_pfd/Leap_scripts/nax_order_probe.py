"""N7-7 order probe (Leap, Mon 2026-11-02 21:5x; Fable mission 27 design H's gate): what exact arithmetic does the matrix
unit's 16 x 32 x 32 op (NAX6's form: A bf16 in registers, B uint4 from threadgroup memory, C fp32 accumulate) perform per
output? The hybrid (1-2-pair groups on the SIMD ALUs) is bitwise to NAX6 only if an ALU sequence reproduces it.
Adversarial operands: A bf16 over a wide exponent range (2^-12 .. 2^12, random signs), B nibbles 0..15, C fp32 over
2^-8 .. 2^12 (so fp64 holds every exact sum: < 53 bits of spread); 1,024 tests x 512 outputs; the GPU result vs CPU candidates (all products are exact in fp32: 8-bit mantissa
x 4-bit integer):
  seq_c      ((C + p0) + p1) + ... + p31 (k ascending), each add rounded to fp32
  seq_then   C + (((p0 + p1) + p2) + ... + p31)
  tree_then  C + pairwise tree over the 32 products (adjacent pairs, 5 levels)
  half_seq   C + seq(p0..p15) + seq(p16..p31) in that order
  quad_tree  sums of 4 adjacent products (exact) combined sequentially, C first
  exact      the exact sum of C and the 32 products, rounded once to fp32
  exact_k16  C + round(exact(p0..15)) + round(exact(p16..31)) (two 16-wide exact dot products, fp32 accumulation)
  exact_k8   the same with four 8-wide exact dot products
Usage: [BigMoE venv] nax_order_probe.py -> ~/Leap/data/nax_order_probe.txt"""
import os, sys, time
from fractions import Fraction
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
import tierglu_nax6 as N6

OUT = os.path.expanduser("~/Leap/data/nax_order_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
SRC = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint t = threadgroup_position_in_grid.x;
    threadgroup uint32_t tb[32 * 8];                                 // 32 columns n x 64 nibbles (k), NAX6's layout
    const device uint32_t* bsrc = bq + (size_t)t * 32 * 8;
    for (int i = 0; i < 8; ++i) tb[lane * 8 + i] = bsrc[lane * 8 + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, RELAX_C,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    const device bfloat* ar0 = a + (size_t)t * 16 * 32 + fm0 * 32;
    const device bfloat* ar1 = a + (size_t)t * 16 * 32 + (fm0 + 8) * 32;
    for (int q = 0; q < 4; ++q) {
        ca[q] = ar0[fn0 + q]; ca[4 + q] = ar1[fn0 + q]; ca[8 + q] = ar0[16 + fn0 + q]; ca[12 + q] = ar1[16 + fn0 + q];
    }
    for (int i = 0; i < 16; ++i) {
        const int n = 16 * (i >> 3) + fn0 + (i & 3), m = fm0 + 8 * ((i >> 2) & 1);
        cg[i] = c[(size_t)t * 512 + m * 32 + n];
    }
    TB tbt((threadgroup uchar*)tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    op.run(ca, tbt, cg);
    for (int i = 0; i < 16; ++i) {
        const int n = 16 * (i >> 3) + fn0 + (i & 3), m = fm0 + 8 * ((i >> 2) & 1);
        d[(size_t)t * 512 + m * 32 + n] = cg[i];
    }
"""
RELAX = os.environ.get("NOP_RELAX", "1") == "1"               # the descriptor's relaxed-precision flag (NAX6: true)
k = mx.fast.metal_kernel(name=f"nax_order_probe_r{int(RELAX)}", input_names=["a", "bq", "c"], output_names=["d"], source=SRC,
                         header=N6._HDR4 + f"#define RELAX_C {'true' if RELAX else 'false'}\n")
NT = int(os.environ.get("NOP_TESTS", "1024"))
rng = np.random.default_rng(7)
say(f"N7-7 matrix-unit order probe {time.strftime('%a %H:%M')} (mode {os.environ.get('NOP_MODE', 'adv')}, relaxed precision {os.environ.get('NOP_RELAX', '1')}); {NT} tests x 16 x 32 outputs, K 32")
# A: bf16 with wide exponents; B: nibbles (row n holds k = 0..63; the op reads k 0..31 of each row)
MODE = os.environ.get("NOP_MODE", "adv")                         # adv | int (integers: every order is exact)
ae = rng.integers(-12, 13, (NT, 16, 32)) if MODE == "adv" else np.full((NT, 16, 32), 7)
am = rng.integers(128, 256, (NT, 16, 32)) if MODE == "adv" else rng.integers(-8, 9, (NT, 16, 32))
asg = rng.choice([-1, 1], (NT, 16, 32))
A32 = (asg * am * np.exp2(ae - 7.0)).astype(np.float32)           # exactly representable in bf16
A16 = (A32.view(np.uint32) >> 16).astype(np.uint16)
assert np.array_equal((A16.astype(np.uint32) << 16).view(np.float32), A32)
Bn = rng.integers(0, 16, (NT, 32, 64)).astype(np.uint32)            # [t][n][k]
words = np.zeros((NT, 32, 8), np.uint32)
for j in range(8):
    for q in range(8):
        words[:, :, j] |= Bn[:, :, 8 * j + q] << (4 * q)           # low nibble = even k
ce = rng.integers(-8, 13, (NT, 16, 32))
C32 = ((rng.choice([-1, 1], (NT, 16, 32)) * rng.random((NT, 16, 32)) * np.exp2(ce)) if MODE == "adv" else
       rng.integers(-1000, 1001, (NT, 16, 32))).astype(np.float32)
(D,) = k(inputs=[mx.array(A16).view(mx.bfloat16), mx.array(words), mx.array(C32)], template=[],
         grid=(NT * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(NT * 512,)], output_dtypes=[mx.float32])
D = np.array(D).reshape(NT, 16, 32)
SAVE = os.environ.get("NOP_SAVE")
if SAVE:
    np.savez(SAVE, A32=A32, Bn=Bn[:, :, :32], C32=C32, D=D)
B = Bn[:, :, :32].transpose(0, 2, 1).astype(np.float32)             # [t][k][n]
P = A32[:, :, :, None] * B[:, None, :, :]                           # [t][m][k][n], exact in fp32
f32 = np.float32
def seq(vals, start=None):
    acc = vals[0].astype(f32) if start is None else (start.astype(f32) + vals[0]).astype(f32)
    for v in vals[1:]:
        acc = (acc + v).astype(f32)
    return acc
pk = [P[:, :, kk, :] for kk in range(32)]
cand = {}
cand["seq_c"] = seq(pk, C32)
cand["seq_then"] = (C32 + seq(pk)).astype(f32)
lvl = pk
while len(lvl) > 1:
    lvl = [(lvl[i] + lvl[i + 1]).astype(f32) for i in range(0, len(lvl), 2)]
cand["tree_then"] = (C32 + lvl[0]).astype(f32)
cand["half_seq"] = ((C32 + seq(pk[:16])).astype(f32) + seq(pk[16:])).astype(f32)
quads = [(((pk[i] + pk[i + 1]).astype(f32) + pk[i + 2]).astype(f32) + pk[i + 3]).astype(f32) for i in range(0, 32, 4)]
cand["quad_seq"] = seq(quads, C32)
P64 = P.astype(np.float64)
def exact_round(parts, base=None):
    """exact: the operand ranges keep every sum within fp64's 53 bits"""
    s = parts.sum(axis=2) if base is None else base + parts.sum(axis=2)
    return s.astype(f32)
cand["exact"] = exact_round(P64, C32.astype(np.float64))
cand["exact_k16"] = ((C32 + exact_round(P64[:, :, :16, :])).astype(f32) + exact_round(P64[:, :, 16:, :])).astype(f32)
e8 = [exact_round(P64[:, :, i:i + 8, :]) for i in range(0, 32, 8)]
cand["exact_k8"] = seq(e8, C32)
tot = D.size
for name, v in cand.items():
    same = int((v.view(np.uint32) == D.view(np.uint32)).sum())
    say(f"  {name:10s}: {same} / {tot} outputs bitwise ({100 * same / tot:.3f} %)")
ex = cand["exact"].astype(np.float64)
rel = np.abs(D.astype(np.float64) - ex) / np.maximum(np.abs(ex), 1e-30)
ulp = np.abs(D.view(np.int32).astype(np.int64) - cand["exact"].view(np.int32).astype(np.int64))
say(f"  vs exact: rel err median {np.median(rel):.2e}, max {rel.max():.2e}; ulp diff: 0 {100 * (ulp == 0).mean():.1f} %, "
    f"1 {100 * (ulp == 1).mean():.1f} %, 2-8 {100 * ((ulp >= 2) & (ulp <= 8)).mean():.1f} %, >8 {100 * (ulp > 8).mean():.1f} %")
dif = D.astype(np.float64) - ex
nz = dif != 0
toward0 = (np.sign(dif) == -np.sign(ex)) & nz
say(f"  nonzero errors {100 * nz.mean():.1f} %: toward zero {100 * toward0.sum() / max(nz.sum(), 1):.1f} % of them")
best = max(cand, key=lambda n: int((cand[n].view(np.uint32) == D.view(np.uint32)).sum()))
say(f"best candidate: {best}")
