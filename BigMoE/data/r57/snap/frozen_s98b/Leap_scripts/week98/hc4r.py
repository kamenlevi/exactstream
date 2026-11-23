"""hc4r.py (Leap, Mon 16 Nov 2026 13:1x; V3 = R57_HC4 for EvalSuite's G2 (Breakthrough writes the switch) and CHECK25's C5
bytes): the FAST HC read with 4-BIT mixers. hc8r v4's kernels (K1C -> K2B -> K3C, ref2's hc8_fast read at <= 16 rows) and
hc8r32's one-pass 17-32-row forms, with only the mixer code loads and decodes switched to MLX's 4-bit g64 layout (8 codes
per uint32, code j of a word at bits 4j; a lane's 16 codes of a row per 64-group = 8 bytes): every other instruction is
hc8r's, so the 4-bit read is the 8-bit read's arithmetic on 4-bit codes. Mixer bytes halve: ~0.63 -> ~0.32 GB per check.
FAST (the mixers' values change); the 32-row form == two 16-row 4-bit reads bit for bit (check()).
  read(P4, h [16, HC*D], bo, inj, NP) / read32(P4, h [32, HC*D], ...) -> (mixed, written, inject)
  prep4(g) -> P4 (a GatedResidual's mixers quantized to 4-bit g64; inject and norm weights as hc8r.prep)
  check()  vs hc8r.emulate's fp32 math on the 4-bit-dequantized mixers, and read32 vs two 16-row reads
usage: [BigMoE venv] hc4r.py check"""
from functools import lru_cache
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts/week98"))
import hc8r as H8
import hc8r32 as H32

BF = mx.bfloat16


def _to4(src, wname, rowlen):
    """hc8r's 8-bit code loads / decodes -> MLX 4-bit: uint4 (16 bytes = 16 codes) -> uint2 (8 bytes = 16 codes)"""
    reps = [
        ("uint4 q[4], qn[4];", "uint2 q[4], qn[4];"),
        (f"q[r] = *(const device uint4*)((const device uchar*){wname} + (size_t)n * {rowlen} + kp0 + 16 * fq);",
         f"q[r] = *(const device uint2*)((const device uchar*){wname} + ((size_t)n * {rowlen} + kp0 + 16 * fq) / 2);"),
        (f"qn[r] = *(const device uint4*)((const device uchar*){wname} + (size_t)n * {rowlen} + k0 + G + 16 * fq);",
         f"qn[r] = *(const device uint2*)((const device uchar*){wname} + ((size_t)n * {rowlen} + k0 + G + 16 * fq) / 2);"),
        ("const uint word = q[r][s];", "const uint word = q[r][s >> 1] >> (16 * (s & 1));"),
        ("static_cast<T>(float((word >> (8 * j)) & 0xFFu))", "static_cast<T>(float((word >> (4 * j)) & 0xFu))"),
    ]
    for a, b in reps:
        assert src.count(a) >= 1, (wname, a[:60])
        src = src.replace(a, b)
    return src


def _k3_to4(src):
    """K3C's up-mixer loads: its first load has no kp0 (the K loop starts at 0)"""
    reps = [
        ("uint4 q[4], qn[4];", "uint2 q[4], qn[4];"),
        ("q[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + 16 * fq);",
         "q[r] = *(const device uint2*)((const device uchar*)uq + ((size_t)n * LR + 16 * fq) / 2);"),
        ("qn[r] = *(const device uint4*)((const device uchar*)uq + (size_t)n * LR + k0 + G + 16 * fq);",
         "qn[r] = *(const device uint2*)((const device uchar*)uq + ((size_t)n * LR + k0 + G + 16 * fq) / 2);"),
        ("const uint word = q[r][s];", "const uint word = q[r][s >> 1] >> (16 * (s & 1));"),
        ("static_cast<T>(float((word >> (8 * j)) & 0xFFu))", "static_cast<T>(float((word >> (4 * j)) & 0xFu))"),
    ]
    for a, b in reps:
        assert src.count(a) >= 1, ("uq", a[:60])
        src = src.replace(a, b)
    return src


_K1C4 = _to4(H8._K1C, "wq", "HCD")
_K3C4 = _k3_to4(H8._K3C)
_K1C4_32 = _to4(H32._K1C32, "wq", "HCD")
_K3C4_32 = _k3_to4(H32._K3C32)


@lru_cache(None)
def _kern(which, eps=None):
    if which == "k1":
        return mx.fast.metal_kernel(name="hc4r_k1c", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1C4, header=H8._HDR)
    if which == "k3":
        return mx.fast.metal_kernel(name="hc4r_k3c", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn", "xsum"],
                                    output_names=["mixed"], source=_K3C4, header=H8._HDR)
    if which == "k1_32":
        return mx.fast.metal_kernel(name="hc4r32_k1c", input_names=["h", "bo", "injp", "wn", "wq", "ws", "wb", "wi"],
                                    output_names=["part_out", "ssq", "hw"], source=_K1C4_32, header=H8._HDR)
    return mx.fast.metal_kernel(name="hc4r32_k3c", input_names=["mix1", "uq", "us", "ub", "hw", "rinv", "wn", "xsum"],
                                output_names=["mixed"], source=_K3C4_32, header=H8._HDR)


def _read(P, h, bo, inj, NP, R):
    HC, D, LR = P["hc"], P["d"], P["lr"]
    HCD = HC * D
    KP, NT = HCD // NP, (LR + HC + 31) // 32
    assert D % KP == 0 and KP % 64 == 0 and NP % HC == 0 and h.shape == (R, HCD)
    pend = bo is not None
    k1 = _kern("k1" if R == 16 else "k1_32")
    part, ssq, hw = k1(
        inputs=[h, bo if pend else h, inj if pend else h, P["wn"], P["wq"], P["ws"], P["wb"], P["wi"]],
        template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC), ("NP", NP), ("KP", KP), ("NT", NT),
                  ("PEND", int(pend))],
        grid=(32 * NT * NP, 1, 1), threadgroup=(32 * NT, 1, 1),
        output_shapes=[(NP, R, NT * 32), (NP, R), (R, HCD)], output_dtypes=[mx.float32, mx.float32, BF])
    k2 = H8._kern(7, P["eps"]) if R == 16 else H32._kern(2, P["eps"])
    mix1, inject, rinv, xsum = k2(
        inputs=[part, ssq], template=[("T", BF), ("LR", LR), ("HC", HC), ("NP", NP), ("NC", NT * 32), ("D", D)],
        grid=(R * NT * 32, 1, 1), threadgroup=(NT * 32, 1, 1),
        output_shapes=[(R, LR), (R, HC), (R, HC), (R, LR // 64)], output_dtypes=[BF, BF, mx.float32, mx.float32])
    (mixed,) = _kern("k3" if R == 16 else "k3_32")(
        inputs=[mix1, P["uq"], P["us"], P["ub"], hw, rinv, P["wn"], xsum],
        template=[("T", BF), ("HCD", HCD), ("D", D), ("LR", LR), ("HC", HC)],
        grid=(32 * HC * (D // 32), 1, 1), threadgroup=(32 * HC, 1, 1),
        output_shapes=[(R, D)], output_dtypes=[BF])
    return mixed, hw, inject


def read(P, h, bo=None, inj=None, NP=20):
    return _read(P, h, bo, inj, NP, 16)


def read32(P, h, bo=None, inj=None, NP=20):
    return _read(P, h, bo, inj, NP, 32)


def prep4(g, group_size=64):
    """a GatedResidual's mixers at 4-bit g64 (the V3 copy), its bf16 inject and norm weights (as hc8r.prep)"""
    wq, ws, wb = mx.quantize(g.input_mix_weight_down.weight.astype(BF), group_size=group_size, bits=4)
    uq, us, ub = mx.quantize(g.input_mix_weight_up.weight.astype(BF), group_size=group_size, bits=4)
    P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=g.block_inject_weight.weight.astype(BF),
             wn=g.hc_norm.weight.astype(BF), hc=int(g.hc_count), d=int(g.hidden_size),
             lr=int(g.input_mix_weight_down.weight.shape[0]), eps=float(g.hc_norm.eps))
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def _emulate4(P, h, bo=None, inj=None):
    """hc8r.emulate's fp32 math with the 4-bit mixers dequantized exactly (fp32)"""
    W = dict(P)
    W["wq"], W["ws"], W["wb"] = _deq_as8(P["wq"], P["ws"], P["wb"])
    W["uq"], W["us"], W["ub"] = _deq_as8(P["uq"], P["us"], P["ub"])
    return H8.emulate(W, h, bo, inj)


def _deq_as8(q, s, b):
    """4-bit g64 -> an equivalent 8-bit g64 triple with the same values (codes 0..15 as uint8, same scales / biases), so
    hc8r.emulate's exact fp32 dequant reads them"""
    N = q.shape[0]
    w = q.reshape(N, -1)
    sh = mx.array([4 * j for j in range(8)], dtype=mx.uint32)
    codes = (mx.right_shift(w[..., None], sh) & 0xF).reshape(N, -1).astype(mx.uint8)
    return mx.view(codes, mx.uint32), s, b


def check(seeds=2, NP=20):
    import numpy as np
    rng = np.random.default_rng(1135)
    HC, D, LR = 4, 2560, 320
    lines, bad = [], 0
    for sd in range(seeds):
        f32 = lambda *s_: mx.array(rng.standard_normal(s_).astype(np.float32))
        wq, ws, wb = mx.quantize((f32(LR, HC * D) * 0.02).astype(BF), group_size=64, bits=4)
        uq, us, ub = mx.quantize((f32(HC * D, LR) * 0.05).astype(BF), group_size=64, bits=4)
        P = dict(wq=wq, ws=ws, wb=wb, uq=uq, us=us, ub=ub, wi=(f32(HC, HC * D) * 0.02).astype(BF),
                 wn=(1.0 + 0.1 * f32(HC * D)).astype(BF), hc=HC, d=D, lr=LR, eps=1e-6)
        mx.eval([v for v in P.values() if isinstance(v, mx.array)])
        for pend in (False, True):
            h = (f32(32, HC * D) * 0.8).astype(BF)
            bo = (f32(32, D) * 0.5).astype(BF) if pend else None
            inj = (mx.sigmoid(f32(32, HC)) * 2).astype(BF) if pend else None
            o16 = read(P, h[:16], bo[:16] if pend else None, inj[:16] if pend else None, NP=NP)
            e16 = _emulate4(P, h[:16], bo[:16] if pend else None, inj[:16] if pend else None)
            o32 = read32(P, h, bo, inj, NP=NP)
            ob = read(P, h[16:], bo[16:] if pend else None, inj[16:] if pend else None, NP=NP)
            mx.eval(*o16, *e16, *o32, *ob)
            err = float(mx.max(mx.abs(o16[0].astype(mx.float32) - e16[0].astype(mx.float32))) /
                        mx.max(mx.abs(e16[0].astype(mx.float32))))
            wok = bool(mx.array_equal(o16[1], e16[1]))
            iok = float(mx.max(mx.abs(o16[2].astype(mx.float32) - e16[2].astype(mx.float32))))
            b32 = all(bool(mx.array_equal(o32[i], mx.concatenate([o16[i], ob[i]]))) for i in range(3))
            bad += (err > 2e-2) + (not wok) + (not b32)
            lines.append(f"  seed {sd} pend {int(pend)}: mixed vs fp32 emulation (4-bit mixers) rel {err:.2e}; written == "
                         f"{wok}; inject max |diff| {iok:.2e}; read32 == two 16-row reads bit for bit {b32}")
            print(lines[-1], flush=True)
    hdr = f"HC4R check: {'PASS' if bad == 0 else f'{bad} FAILS'} (16-row 4-bit read vs the fp32 emulation; read32 vs 2 x 16)"
    print(hdr, flush=True)
    return bad, [hdr] + lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        b, lines = check()
        open(os.path.expanduser("~/Leap/data/week98/check25/hc4r_check.txt"), "w").write("\n".join(lines) + "\n")
        sys.exit(1 if b else 0)
