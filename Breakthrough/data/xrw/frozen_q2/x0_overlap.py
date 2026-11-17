"""x0_overlap.py (Breakthrough, Tue 17 Nov 2026 09:4x; EXPERT REWRITE screen X0, pre-registered in docs/EXPERT_REWRITE.md
0aee02c before any run): can the M5's matmul2d (MPP, the GPU neural accelerators) run UNDER a weight stream at all?
Four synthetic kernels on kernel A's grid (62 segments x 20 column tiles, 8 simdgroups = 2 projections x 4 K chunks,
10 quant groups per simdgroup) and kernel A's byte pattern (native layout, 6 code words per lane per quant group; 76 MB):
  S1 STREAM  the code loads only (folded into one word)
  S2 MMA     2 x matmul2d(16, 32, 32) per quant group on a threadgroup-resident uint4b tile, x fragments from device,
             today's epilogue (scale x cg + bias x rowsum) - no code loads
  S3 BOTH    S1's loads and S2's MMAs in one loop, no data dependency between them: loads issued before the MMAs,
             folded after them (amended 09:5x after the shared smoke; S3f = the smoke's fold-first form, info)
  S4 CHAIN   today's dependent chain: load -> spread3 decode -> threadgroup -> MMA -> epilogue (tier 3, plain scales)
GPU busy per kernel from libmtlprof (command-buffer GPU intervals), ROUNDS interleaved rounds x REPS launches, median.
Lines (pre-registered): FEASIBLE if S3 <= 1.15 x max(S1, S2); KILL if S3 >= 0.90 x (S1 + S2); else PARTIAL.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x0_overlap.py [--rounds 5] [--reps 20] [--tag T]
  -> data/xrw/x0_overlap[_T].txt / .json   (MEM < 1 GB)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP                                        # before mlx: names pipelines, times command buffers

MP.load(2)
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.environ.get("XRW_OUTD") or os.path.join(HERE, "..", "..", "data", "xrw")
NSEG, NT, KG, KS = 62, 20, 40, 4                            # segments, 32-column tiles (N 640), quant groups (K 2560), K chunks
GPC = KG // KS
K = 64 * KG

HDR = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline uint spread3(uint x) {                              // 8 3-bit codes (24 bits) -> 8 nibbles (Leap's NAX6 helper)
    uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
    a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
    return (a & 0x07070707u) | ((a & 0x38383838u) << 1);
}
"""

# shared prologue: indices, the lane's code row (native layout [seg][proj][tile][KG][32 rows][6 words]), x fragment rows
PRO = r"""
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint pj = sg & 1u, kc = sg >> 1;
    const uint sgi = threadgroup_position_in_grid.y, tile = threadgroup_position_in_grid.x;
    const device uint32_t* wrow = codes + ((((size_t)sgi * 2 + pj) * NT_C + tile) * KG_C) * 192 + lane * 6;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const uint ma0 = (sgi * 5u + (uint)fm0) % 32u, ma1 = (sgi * 5u + (uint)fm0 + 8u) % 32u;
    const device T* ar0 = x + (size_t)ma0 * K_C;
    const device T* ar1 = x + (size_t)ma1 * K_C;
    const device float* as0 = xs + (size_t)ma0 * KG_C;
    const device float* as1 = xs + (size_t)ma1 * KG_C;
    threadgroup uint32_t tb_all[8 * 2 * 32 * 8];
    threadgroup float sb_all[8 * 2 * 64];
    threadgroup uint32_t* tbw = tb_all + sg * 512;
    threadgroup float* sbw = sb_all + sg * 128;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    uint fold = 0;
"""

MMA = r"""
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        for (int hh = 0; hh < 2; ++hh) {
            const int k0 = 64 * g + 32 * hh;
            const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
            const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
            const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
            const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
            for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
            TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
            op.run(ca, tbt, cg);
        }
        {
            const float x0 = as0[g], x1 = as1[g];
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                acc[i] += sb[col] * cg[i] + sb[32 + col] * (((i >> 2) & 1) ? x1 : x0);
            }
        }
"""

LOADFOLD = r"""
        {
            const device uint32_t* c = wrow + 192 * g;
            fold ^= c[0] ^ (c[1] << 1) ^ (c[2] << 2) ^ (c[3] << 3) ^ (c[4] << 4) ^ (c[5] << 5);
        }
"""

LOADISSUE = r"""
        const device uint32_t* cq = wrow + 192 * g;               // issued before the MMAs, consumed after them
        const uint v0 = cq[0], v1 = cq[1], v2 = cq[2], v3 = cq[3], v4 = cq[4], v5 = cq[5];
"""
FOLDLATE = r"""
        fold ^= v0 ^ (v1 << 1) ^ (v2 << 2) ^ (v3 << 3) ^ (v4 << 4) ^ (v5 << 5);
"""

STATIC_TILE = r"""
    for (int i = lane; i < 512; i += 32) tbw[i] = (0x76543210u ^ ((uint)i * 2654435761u)) & 0x77777777u;
    for (int i = lane; i < 128; i += 32) sbw[i] = 0.001f * (float)(i & 63);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint32_t* tb = tbw;
    threadgroup float* sb = sbw;
"""

EPI = r"""
    float s = 0.0f;
    for (int i = 0; i < 16; ++i) s += acc[i];
    out[((size_t)sgi * NT_C + tile) * 256 + thread_index_in_threadgroup] = as_type<uint>(s) ^ fold;
"""

SRC = {
    "S1": PRO + "    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {\n" + LOADFOLD + "    }\n" + EPI,
    "S2": PRO + STATIC_TILE + "    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {\n" + MMA + "    }\n" + EPI,
    "S3": PRO + STATIC_TILE + "    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {\n" + LOADISSUE + MMA + FOLDLATE
          + "    }\n" + EPI,
    "S3f": PRO + STATIC_TILE + "    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {\n" + LOADFOLD + MMA + "    }\n"
           + EPI,
    "S4": PRO + r"""
    for (int i = lane; i < 128; i += 32) sbw[i] = 0.001f * (float)(i & 63);
    for (int g = kc * GPC_C; g < (kc + 1) * GPC_C; ++g) {
        threadgroup uint32_t* tb = tbw + (g & 1) * 256;
        threadgroup float* sb = sbw;
        {
            const device uint32_t* c = wrow + 192 * g;
            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
            uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8);
            dst[0] = uint4(spread3(ch[0]), spread3(ch[1]), spread3(ch[2]), spread3(ch[3]));
            dst[1] = uint4(spread3(ch[4]), spread3(ch[5]), spread3(ch[6]), spread3(ch[7]));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
""" + MMA + "    }\n" + EPI,
}


def kern(name):
    return mx.fast.metal_kernel(name=f"x0_{name.lower()}", input_names=["codes", "x", "xs"], output_names=["out"],
                                source=SRC[name], header=HDR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(1117)
    nwords = NSEG * 2 * NT * KG * 192
    codes = mx.array(rng.integers(0, 2**32, size=nwords, dtype=np.uint32))
    x = mx.array(rng.standard_normal((32, K)).astype(np.float32)).astype(mx.bfloat16)
    xs = mx.array(rng.standard_normal((32, KG)).astype(np.float32))
    mx.eval(codes, x, xs)
    K_ = {n: kern(n) for n in SRC}
    tmpl = [("T", mx.bfloat16), ("NT_C", NT), ("KG_C", KG), ("GPC_C", GPC), ("K_C", K)]

    def launch(n):
        return K_[n](inputs=[codes, x, xs], template=tmpl, grid=(NT * 256, NSEG, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(NSEG * NT * 256,)], output_dtypes=[mx.uint32])[0]

    t0 = time.time()
    for n in SRC:                                           # build + warm
        mx.eval(launch(n))
    mx.synchronize()
    tc = time.time() - t0
    res = {n: [] for n in SRC}
    for r in range(a.rounds):
        order = list(SRC) if r % 2 == 0 else list(SRC)[::-1]
        for n in order:
            mx.synchronize()
            c0 = MP.counts()["ncb"]
            m0 = MP.now()
            outs = []
            for _ in range(a.reps):                         # one command buffer per launch: eval each
                o = launch(n)
                mx.eval(o)
                outs.append(o)
            mx.synchronize()
            m1 = MP.now()
            c1 = MP.counts()["ncb"]
            cb = MP.cbs(c0, c1)
            gpu = [1e6 * (e - s) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s]
            res[n].append(dict(gpu_med=float(np.median(gpu)) if gpu else float("nan"), n_cb=len(gpu),
                               wall_per=1e6 * (m1 - m0) / a.reps))
    med = {n: float(np.median([d["gpu_med"] for d in res[n]])) for n in SRC}
    spread = {n: (max(d["gpu_med"] for d in res[n]) - min(d["gpu_med"] for d in res[n])) / med[n] for n in SRC}
    s1, s2, s3, s4 = (med[n] for n in ("S1", "S2", "S3", "S4"))
    s3f = med["S3f"]
    hid = (s1 + s2 - s3) / min(s1, s2)                      # share of the smaller term hidden under the larger
    verdict = ("FEASIBLE" if s3 <= 1.15 * max(s1, s2) else ("KILL" if s3 >= 0.90 * (s1 + s2) else "PARTIAL"))
    gb = nwords * 4 / 1e9
    lines = [f"X0 OVERLAP SCREEN {time.strftime('%a %d %b %H:%M:%S')} (pre-registered 0aee02c): MLX {mx.__version__}; "
             f"grid {NT} x {NSEG} threadgroups x 256; code bytes {gb:.3f} GB; {a.rounds} rounds x {a.reps} launches; "
             f"build + warm {tc:.1f} s",
             "| kernel | GPU us per launch (median of round medians) | spread (max-min)/med | GB/s on the codes |",
             "|---|---|---|---|"]
    for n in SRC:
        lines.append(f"| {n} | {med[n]:.1f} | {100 * spread[n]:.1f} % | {gb / (med[n] * 1e-6):.0f} |")
    lines.append(f"S3 {s3:.1f} vs max(S1, S2) {max(s1, s2):.1f} (FEASIBLE <= x1.15 = {1.15 * max(s1, s2):.1f}) and "
                 f"S1 + S2 {s1 + s2:.1f} (KILL >= x0.90 = {0.9 * (s1 + s2):.1f}); hidden share of the smaller term "
                 f"{100 * hid:.0f} % -> {verdict}")
    lines.append(f"info: S3f (loads folded BEFORE the MMAs, the smoke's form) {s3f:.1f} = {s3f / (s1 + s2):.2f} x (S1 + S2)")
    lines.append(f"info: S4 (today's dependent chain) {s4:.1f} = {s4 / (s1 + s2):.2f} x (S1 + S2), "
                 f"{s4 / max(s1, s2):.2f} x max(S1, S2)")
    print("\n".join(lines), flush=True)
    sfx = f"_{a.tag}" if a.tag else ""
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, f"x0_overlap{sfx}.txt"), "a").write("\n".join(lines) + "\n\n")
    json.dump(dict(med=med, spread=spread, verdict=verdict, rounds=res, hidden=hid),
              open(os.path.join(OUTD, f"x0_overlap{sfx}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
