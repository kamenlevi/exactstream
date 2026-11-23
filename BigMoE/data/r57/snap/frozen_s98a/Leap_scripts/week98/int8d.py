"""int8d.py (Leap, Mon 16 Nov 2026 12:3x; CHECK25 / EvalSuite lossygate amendment 5 G1 = C6 + int8 dense): INT8 DENSE
ACTIVATIONS for the FAST stack's 4-bit dense projections, as an opt-in installer on top of ref2 (ref2's file untouched).

What changes (FAST, lossy; EvalSuite gates it; adopting it is Kamen's call):
  - every 4-bit g64 dense call on a target-model weight that ref2 runs natively (DENSE-N-RAM, DENSE-N, SMALL-N: GDN
    qkv|z, b|a, out_proj; attention q / k / v / o, indexer qk; shared gate|up|gate and down; PLE key / value), at 1-32
    rows, in the VERIFY phase (MTPLX 'decode_verify': the verify forward and its commit replay; never the prefill or the
    drafter), while ON[0] is set;
  - its activations are quantized per (row, 64-group) with NAX-C's rule (scale = max|x| / 127, round to nearest, clamp
    +-127; one quantizer dispatch per distinct input, shared by the calls on it), and the call runs dn32.dn32q (int8 x
    int4 from threadgroup memory -> int32 sums; y += sx * (s * (P + 8 qs) + b * qs)).
Unchanged: the 8-bit router, the HC8 mixers (hc8r), the head (SVH / V1), the experts (NAX6-N bf16), the prefill and every
call with ON[0] off (then each call is ref2's own, bit for bit: the bf16 weights are never modified).
Memory: none extra. The kernel reads ref2's native copies (DENSE-N-RAM / DENSE-N / SMALL-N, qmm26.repack's layout) and
stages them for the matrix unit itself (dn32 native=True; bitwise == the staged copy). Row-invariant (integer sums per
row): a row's output does not depend on its companions.
Scope: BigMoE's screen sets ON[0] per item (R57_INT8D_SETS, like R57_TIERS_SETS), so IFEval / chat keep their tokens.
  install() after ref2.apply() (and after v32 / f32one if used); ON[0] = True / False; COUNTS; remove()"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98a/Leap_scripts/week98"))
import ref2
import dn32 as DN

ON = [False]
HI = 32
COUNTS = dict(int8=0, int8_rows=0, quant=0, layout=0, layout_mb=0.0, passthru=0)
_ORIG = {}
_NAXD = {}                       # id(native weight) -> (native weight, wx, sx, bx)
_Q = []                          # the last few inputs and their int8 forms: [(x, (xq, xs, xqs))]
_PH = [None]


def _verify_phase():
    if _PH[0] is None:
        from mtplx.attention_context import current_attention_phase
        _PH[0] = current_attention_phase
    return _PH[0]() == "decode_verify"


def _quant(x, rows, K):
    """NAX-C's quantizer, once per distinct input array (the calls on one input, e.g. q / k / v or qkv|z and b|a, pass
    the same array object; the cache holds the last 4, alive, so identity is safe)"""
    for a, q in _Q:
        if a is x:
            return q
    q = DN.quant64(x.reshape(rows, K))
    COUNTS["quant"] += 1
    _Q.append((x, q))
    del _Q[:-4]
    return q


def _kd(ks, K):
    """the K split for dn32q: the dense call's own ks when it fits threadgroup memory, else the largest that does"""
    g = K // 64
    if DN._tg_ok(ks, 1) and g % ks == 0:
        return ks
    return next(k for k in (8, 5, 4, 2, 1) if g % k == 0 and DN._tg_ok(k, 1))


def _naxd(wn, sn, bn, N, K):
    e = _NAXD.get(id(wn))
    if e is None or e[0] is not wn:
        w, s, b = ref2._dn_unpack(wn, sn, bn, N, K)
        e = _NAXD[id(wn)] = (wn,) + DN.layout(w, s, b)
        mx.eval(e[1:])
        COUNTS["layout"] += 1
        COUNTS["layout_mb"] += sum(t.nbytes for t in e[1:]) / 1e6
    return e


def _native_of(x, w, args, kw):
    """(wn, sn, bn, N, K, ks) if this call is one of ref2's native 4-bit dense calls on a target weight, else None"""
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    if not (transpose and mode == "affine" and bits == 4 and group_size == 64 and scales is not None
            and scales.dtype == mx.bfloat16 and x.dtype == mx.bfloat16):
        return None
    K = x.shape[-1]
    e = ref2._DNR.get(id(w)) if ref2._DNR else None
    if e is not None and e[0] is w:                                       # DENSE-N-RAM: w is the native copy
        _, N, K0 = e
        return (w, scales, biases, N, K0, ref2._dn_ks(N, K0)) if K == K0 else None
    if biases is None or K % 64 or w.ndim != 2:
        return None
    N = int(w.shape[0])
    if N >= ref2.CFG["dense_min_n"] and N % 64 == 0:
        d = ref2._DN.get(id(w))                                          # DENSE-N's native copy (made by ref2's 16-row path)
        if d is not None and d[0] is w:
            return d[1], d[2], d[3], N, K, ref2._dn_ks(N, K)
        return None
    s = ref2._SN.get(id(w))                                              # SMALL-N's padded native copy
    if s is not None and s[0] is w:
        return s[1], s[2], s[3], s[4], K, ref2._dn_ks(s[4], K)
    if ref2.CFG["small_native"] and w is ref2._DN_SRC[0]:
        s = ref2._sn_get(w, scales, biases)
        if s is not None:
            return s[1], s[2], s[3], s[4], K, ref2._dn_ks(s[4], K)
    return None


def _qmm8(x, w, *args, **kw):
    if ON[0] and ref2._ON[0]:
        lead = x.shape[:-1]
        rows = 1
        for d in lead:
            rows *= d
        if 1 <= rows <= HI and _verify_phase():
            nat = _native_of(x, w, args, kw)
            if nat is not None:
                wn, sn, bn, N, K, ks = nat
                N0 = int(w.shape[0]) if not (ref2._DNR and ref2._DNR.get(id(w)) is not None) else N
                y = DN.dn32q(*_quant(x, rows, K), wn, sn, bn, N, _kd(ks, K), native=True)
                COUNTS["int8"] += 1
                COUNTS["int8_rows"] += rows
                if N0 != N:
                    y = y[:, :N0]
                return y.reshape(*lead, y.shape[-1])
    COUNTS["passthru"] += 1
    return _ORIG["qmm"](x, w, *args, **kw)


def install():
    """after ref2.apply() (and v32 / f32one, if used): wrap mx.quantized_matmul (idempotent); ON[0] switches it"""
    assert ref2._ORIG, "ref2.apply() first"
    if not _ORIG:
        _ORIG["qmm"] = mx.quantized_matmul
        mx.quantized_matmul = _qmm8


def remove():
    if _ORIG:
        mx.quantized_matmul = _ORIG["qmm"]
        _ORIG.clear()
    ON[0] = False


def check(seeds=2, Ms=(1, 2, 7, 16, 17, 24, 32)):
    """dn32q (now 1-32 rows) per FAST dense shape: error vs fp32 (max |err| / max |ref|) next to bf16 qmm16n / qmm32n,
    and row-invariance (each row of an M-row call == the same row in a 1-row call, bit for bit)"""
    import numpy as np
    import qmm26 as Q26
    import qmm32n as Q
    from qmm26m import qmm16n_m
    rng = np.random.default_rng(1132)
    lines, bad = [], 0
    for name, N, K, _ in Q.SHAPES:
        ks = Q.ks_of(N, K)
        kd = _kd(ks, K)
        w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=64, bits=4)
        wd = mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)
        wn, sn, bn = Q26.repack(wq, s, b)
        wx, sx, bx = DN.layout(wq, s, b)
        mx.eval(wn, sn, bn, wx, sx, bx, wd)
        e8 = e16 = 0.0
        inv = ninv = 0
        gnat = nnat = 0
        for _ in range(seeds):
            X = mx.array(rng.standard_normal((HI, K)).astype(np.float32)).astype(mx.bfloat16)
            for M in Ms:
                ref = X[:M].astype(mx.float32) @ wd.T
                y8 = DN.dn32q(*DN.quant64(X[:M]), wx, sx, bx, N, kd)
                y8n = DN.dn32q(*DN.quant64(X[:M]), wn, sn, bn, N, kd, native=True)
                ybn = DN.dn32(X[:M], wn, sn, bn, N, kd, native=True)
                ybc = DN.dn32(X[:M], wx, sx, bx, N, kd)
                mx.eval(y8n, ybn, ybc)
                nat_ok = bool(mx.array_equal(y8, y8n)) and bool(mx.array_equal(ybn, ybc))
                nnat += 1
                gnat += nat_ok
                yb = qmm16n_m(X[:M], wn, sn, bn, N, ks) if M <= 16 else Q.qmm32n(X[:M], wn, sn, bn, N, ks)
                mx.eval(ref, y8, yb)
                r = float(mx.max(mx.abs(ref)))
                e8 = max(e8, float(mx.max(mx.abs(y8.astype(mx.float32) - ref))) / r)
                e16 = max(e16, float(mx.max(mx.abs(yb.astype(mx.float32) - ref))) / r)
                if M > 1:
                    for m in (0, M // 2, M - 1):
                        y1 = DN.dn32q(*DN.quant64(X[m:m + 1]), wx, sx, bx, N, kd)
                        ninv += 1
                        inv += bool(mx.array_equal(y1[0], y8[m]))
        bad += (inv != ninv) + (gnat != nnat)
        lines.append(f"  {name:36s} N {N:6d} K {K:5d} ks {kd:2d}: int8 err {e8:.2e} (bf16 {e16:.2e}); row-invariant "
                     f"{inv}/{ninv}; native-staged == copy-staged (dn32q and dn32) {gnat}/{nnat}")
        print(lines[-1], flush=True)
    return bad, lines


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        b, lines = check()
        p = os.path.expanduser("~/Leap/data/week98/check25/int8d_check.txt")
        open(p, "w").write("INT8D check (dn32q at 1-32 rows): error vs fp32 next to the bf16 native form; row-invariance "
                           "(rows of an M-row call == 1-row calls)\n" + "\n".join(lines) + "\n")
        sys.exit(1 if b else 0)
