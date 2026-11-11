"""HC8-K3C ablation (Leap, Sat 2026-11-07; PLAN 220 1a, NewIdeas QUEUE a85e5c8: the HC8 k3c tail first). hc8r's k3c (the
16-row HC read's up mix + tail: 80 threadgroups x 4 stream simdgroups, an 8-bit code MMA over K 320 in 5 groups, then
16 scattered hw loads per lane, the gate, and the ordered cross-stream sum) takes 26.6 us per call in the 16-row chain
(96 per check, 3.83 MB, 148 GB/s; v1' + wired, gap_p220b). Where do the 26.6 us go? Variants by source substitution:
  V0  hc8r._K3C as is (the reference)
  TGT the tail's hv / wv from threadgroup memory (a coalesced cooperative copy first): same values, bitwise (a candidate)
  A2  ablation: the codes not loaded (a constant word): the cost of the code stream
  A3  ablation: hv / wv constants: the cost of the scattered tail loads
  A4  ablation: no MMA (op.run skipped): the matrix-unit cost
Dependent chain of 96 calls (96 distinct cold up-weights, as the engine's 48 layers x 2 sides), hw rewritten each call
by the link (warm, as k1c's output is). Bits: TGT vs V0 exact. usage: python hc8_k3c_abl.py -> ~/Leap/data/hc8_k3c_abl.txt"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
import hc8r as H

OUT = os.path.expanduser("~/Leap/data/hc8_k3c_abl.txt")
BF = mx.bfloat16
D, HC, LR = 2560, 4, 320
HCD, KG = HC * D, LR // 64


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


_TAIL_OLD = """    T hv[16], wv[16];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
        hv[i] = hw[(size_t)m * HCD + col];
        wv[i] = wn[col];
    }"""
_TAIL_TGT = """    threadgroup T hws[HC][16][32];
    threadgroup T wns[HC][32];
    for (uint o = t; o < (uint)(HC * 16 * 32); o += 32 * HC) {          // coalesced: 32 consecutive channels per row
        const uint s_ = o / 512, m_ = (o / 32) % 16, c_ = o % 32;
        hws[s_][m_][c_] = hw[(size_t)m_ * HCD + s_ * D + ch0 + c_];
    }
    for (uint o = t; o < (uint)(HC * 32); o += 32 * HC) wns[o / 32][o % 32] = wn[(o / 32) * D + ch0 + o % 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T hv[16], wv[16];
    for (short i = 0; i < 16; i++) {
        const int m = fm + 8 * ((i & 7) >> 2);
        const int cl = (i >> 3) * 16 + 4 * fq + (i & 3);
        hv[i] = hws[sg][m][cl];
        wv[i] = wns[sg][cl];
    }"""
_TAIL_A3 = """    T hv[16], wv[16];
    for (short i = 0; i < 16; i++) { hv[i] = static_cast<T>(0.5f); wv[i] = static_cast<T>(1.0f); }"""
_Q0 = "        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);"
_QN = "                qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);"


def variants():
    src = H._K3C
    assert src.count(_TAIL_OLD) == 1 and src.count(_Q0) == 1 and src.count(_QN) == 1 and src.count("op.run(ct_a, ct_b, ct_c);") == 1
    return {"V0": src,
            "TGT": src.replace(_TAIL_OLD, _TAIL_TGT),
            "A2": src.replace(_Q0, "        q[r] = uint4(0x01020304u + uint(n & 3));").replace(_QN, "                qn[r] = q[r];"),
            "A3": src.replace(_TAIL_OLD, _TAIL_A3),
            "A4": src.replace("op.run(ct_a, ct_b, ct_c);", "ct_c[0] += float(ct_a[0]) * float(ct_b[0]);"),
            "PF5": _pf5(src), "KS5": H._MMA + _KS5}


_PF_OLD_INIT = """    uint4 q[4], qn[4];
    for (short r = 0; r < 4; r++) {
        const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);
    }"""
_PF_NEW_INIT = """    uint4 qa[KG][4];                                     // PF5: every group's codes in flight before the first MMA
    for (int g_ = 0; g_ < KG; g_++)
        for (short r = 0; r < 4; r++) {
            const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
            qa[g_][r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + g_ * G + 16 * fq);
        }"""
_PF_OLD_PF = """        if (g + 1 < KG) {
            for (short r = 0; r < 4; r++) {
                const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
                qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);
            }
        }"""


def _pf5(src):
    """PF5: all KG groups' code words loaded before the loop (same words, same arithmetic: bitwise)"""
    for x in (_PF_OLD_INIT, _PF_OLD_PF, "const uint word = q[r][s];", "        for (short r = 0; r < 4; r++) q[r] = qn[r];"):
        assert src.count(x) == 1, x[:60]
    return (src.replace(_PF_OLD_INIT, _PF_NEW_INIT).replace(_PF_OLD_PF, "")
               .replace("const uint word = q[r][s];", "const uint word = qa[g][r][s];")
               .replace("        for (short r = 0; r < 4; r++) q[r] = qn[r];", ""))


_KS5 = r"""
    constexpr int G = 64, KG = LR / G;
    const int sgi = int(simdgroup_index_in_threadgroup);   // KS5: HC x KG simdgroups, one per (stream, K group)
    const int sg = sgi % HC, gk = sgi / HC;
    const uint t = thread_index_in_threadgroup;
    constexpr uint NT_ = 32 * HC * KG;
    const int ch0 = int(threadgroup_position_in_grid.x) * 32;
    const int c0 = sg * D + ch0;
    threadgroup T Ms[16 * LR];
    threadgroup float tp[HC * 16 * 32];
    uint4 q[4];
    for (short r = 0; r < 4; r++) {
        const int n = c0 + (r >> 1) * 16 + fm + (r & 1) * 8;
        q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + gk * G + 16 * fq);
    }
    for (uint o = t; o < 16 * LR / 8; o += NT_)
        *(threadgroup uint4*)(&Ms[o * 8]) = *(const device uint4*)(mix1 + o * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int g = gk, k0 = g * G;
        threadgroup const T* a0 = &Ms[fm * LR + k0 + 16 * fq];
        threadgroup const T* a1 = &Ms[(fm + 8) * LR + k0 + 16 * fq];
        const float xs0 = xsum[fm * KG + g], xs1 = xsum[(fm + 8) * KG + g];
        for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
        for (short s = 0; s < 4; s++) {
            for (short j = 0; j < 4; j++) { ct_a[j] = a0[4 * s + j]; ct_a[4 + j] = a1[4 * s + j]; }
            for (short r = 0; r < 4; r++) {
                const uint word = q[r][s];
                const short base = (r >> 1) * 8 + (r & 1) * 4;
                for (short j = 0; j < 4; j++) ct_b[base + j] = static_cast<T>(float((word >> (8 * j)) & 0xFFu));
            }
            op.run(ct_a, ct_b, ct_c);
        }
        for (short bk = 0; bk < 2; bk++)
            for (short j = 0; j < 4; j++) {
                const int n = c0 + 16 * bk + 4 * fq + j;
                const float s_ = float(us[(size_t)n * KG + g]), b_ = float(ub[(size_t)n * KG + g]);
                const short i0 = bk * 8 + j, i1 = bk * 8 + 4 + j;
                acc[i0] += s_ * ct_c[i0] + b_ * xs0;
                acc[i1] += s_ * ct_c[i1] + b_ * xs1;
            }
    }
    for (int st = 0; st < KG; st++) {                    // the groups' terms summed in k3c's order g = 0 .. KG - 1
        if (gk == st) {
            for (short i = 0; i < 16; i++) {
                const int m = fm + 8 * ((i & 7) >> 2);
                const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
                threadgroup float* p_ = &tp[(sg * 16 + m) * 32 + nl];
                *p_ = (st == 0) ? acc[i] : (*p_ + acc[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gk == 0) {
        T hv[16], wv[16];
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int col = c0 + (i >> 3) * 16 + 4 * fq + (i & 3);
            hv[i] = hw[(size_t)m * HCD + col];
            wv[i] = wn[col];
        }
        const float r0 = rinv[fm * HC + sg], r1 = rinv[(fm + 8) * HC + sg];
        for (short i = 0; i < 16; i++) {
            const int m = fm + 8 * ((i & 7) >> 2);
            const int nl = (i >> 3) * 16 + 4 * fq + (i & 3);
            const float a_ = tp[(sg * 16 + m) * 32 + nl];
            const float gate = 1.0f / (1.0f + metal::precise::exp(-a_));
            const float nv = float(hv[i]) * (((i & 7) >> 2) ? r1 : r0) * float(wv[i]);
            tp[(sg * 16 + m) * 32 + nl] = gate * nv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = t; o < 16 * 32; o += NT_) {
        const uint m = o / 32, nl = o % 32;
        float sum = 0.0f;
        for (int s = 0; s < HC; s++) sum += tp[(s * 16 + m) * 32 + nl];
        mixed[(size_t)m * D + ch0 + nl] = static_cast<T>(sum / float(HC));
    }
"""


_KS = {}


def call(name, src, mix1, uq, us, ub, hw, rinv, wn, xsum):
    if name not in _KS:
        _KS[name] = mx.fast.metal_kernel(name=f"hc8r_k3c_abl_{name}", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn", "xsum"],
                                         output_names=["mixed"], source=src, header=H._HDR)
    (y,) = _KS[name](inputs=[mix1, uq, us, ub, hw, rinv, wn, xsum], template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
                     grid=(32 * HC * (KG if name == "KS5" else 1) * (D // 32), 1, 1),
                     threadgroup=(32 * HC * (KG if name == "KS5" else 1), 1, 1), output_shapes=[(16, D)], output_dtypes=[BF])
    return y


def main(reps=5, n=96):
    rng = np.random.default_rng(3)
    V = variants()
    W = []
    for i in range(n):                                   # 96 distinct up-mixers (8-bit codes + bf16 scales / biases)
        w = mx.array(rng.standard_normal((HCD, LR)).astype(np.float32) * 0.05).astype(BF)
        uq, us, ub = mx.quantize(w, group_size=64, bits=8)
        W.append((uq, us, ub))
    mx.eval(W)
    mix1 = mx.array(rng.standard_normal((16, LR)).astype(np.float32)).astype(BF)
    hw0 = mx.array(rng.standard_normal((16, HCD)).astype(np.float32)).astype(BF)
    rinv = mx.array(rng.uniform(0.5, 2.0, (16, HC)).astype(np.float32))
    wn = mx.array(rng.uniform(0.5, 1.5, (HCD,)).astype(np.float32)).astype(BF)
    xsum = mx.array(rng.standard_normal((16, KG)).astype(np.float32))
    mx.eval(mix1, hw0, rinv, wn, xsum)
    say(f"\nHC8-K3C ablation {time.strftime('%a %H:%M:%S')} ({os.environ.get('GS_COND', 'share lock')}; MLX {mx.__version__}); "
        f"bytes per call {(W[0][0].nbytes + W[0][1].nbytes + W[0][2].nbytes + hw0.nbytes) / 1e6:.2f} MB")
    ref = call("V0", V["V0"], mix1, *W[0], hw0, rinv, wn, xsum)
    tgt = call("TGT", V["TGT"], mix1, *W[0], hw0, rinv, wn, xsum)
    mx.eval(ref, tgt)
    for cand in os.environ.get("K3C_BITS", "TGT,PF5,KS5").split(","):
        eq = all(bool(mx.array_equal(call("V0", V["V0"], mix1, *W[i], hw0, rinv, wn, xsum),
                                      call(cand, V[cand], mix1, *W[i], hw0, rinv, wn, xsum))) for i in range(0, n, 8))
        say(f"  bits: {cand} == V0 on 12 weight sets: {eq}")
    arms = os.environ.get("K3C_ARMS", "V0,TGT,A2,A3,A4,PF5").split(",")

    def chain(name):
        mx.synchronize()
        t0 = time.perf_counter()
        y = None
        for i in range(n):
            hw = hw0 if y is None else hw0 + (0 * y[:, :1]).astype(BF)   # the link: hw rewritten (warm), dependent
            y = call(name, V[name], mix1, *W[i], hw, rinv, wn, xsum)
        mx.eval(y)
        return 1e6 * (time.perf_counter() - t0) / n

    for _ in range(4):                                   # warm the clock and every arm's pipeline
        for name in arms:
            chain(name)
    T_ = {name: [] for name in arms}
    for rep in range(reps):                              # interleaved: each round runs every arm once
        for name in arms:
            T_[name].append(chain(name))
    res = {name: sorted(v)[len(v) // 2] for name, v in T_.items()}
    say("  per call us (median of %d chains of %d, incl. the link): " % (reps, n) + ", ".join(f"{k} {v:.1f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
