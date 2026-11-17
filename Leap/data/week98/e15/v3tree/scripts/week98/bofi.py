"""bofi.py (Leap, Sun 15 Nov 2026 18:1x; CHECK55 track A, BOF PASS b380726): an opt-in installer of BOF (bof.qmm16bof, a
blocked ordered fold of qmm16c ks 1: REF2-T's OWN bits, bitwise 5 shapes x 4 configs) over ref2 (+ v32 / v32one / refc).
install(ref2), then ON[0] = True: every <= 16-row 4-bit g64 call on a shape in TABLE runs qmm16bof instead of ref2's
qmm16c ks 1 (the same bits, so no reference change); everything else falls through. 17-32-row calls keep v32one's
qmm16c2a / qmm16c2 (ks 1: also the same bits).
TABLE (bof_bench, cold): o_proj / out_proj 2,560 x 6,144 bs 4 (85.9 -> 48.6 us); QSA qkv_fused 13,312 x 2,560 bs 8
(107.0 -> 101.4); shared down 2,560 x 640 bs 4 (10.8 -> 7.4); lm_head 248,320 x 2,560 bs 4 (1,612.7 -> 1,454.0). The GDN
in-proj (115.6 -> 114.5) stays on qmm16c.
Tue 17 Nov 01:0x (for ref2's REF2_BOF switch; BigMoE's review of patch v1):
  - Fires only while ref2 is on (RF._ON[0]), as ref2's own _qmm does (r57 removes ref2 before every draft).
  - BOF2 (17-32 rows, TABLE32) fires only where v32 is live (v32._live(rows): the verify forward / commit replay, never
    the prefill). There the call would otherwise become qmm16c ks-1 tiles (v32's split, v32one's qmm16c2 / 2a). In the
    prefill, ref2 sends 17-32 rows to MLX's own kernel, so BOF2 there would change the KV cache.
  - refc ON -> fall through (the contract's calls are refc's; E14 never had both on).
  - Drift guard: the first call of each specialization (form, N, K, bs, rg) also runs the reference route on the same
    real input and compares bit for bit (uint16 views). On a mismatch it prints a DRIFT line, returns the reference's bits and keeps
    that specialization off. Cost: one sync per specialization per process.
self_test(): synthetic, the small TABLE shapes + TABLE32 (lm_head left to the first-call guard: ~0.36 GB of weights)."""
import math
import os
import sys
import mlx.core as mx

_P98 = os.path.expanduser("~/Leap/scripts/week98")
if _P98 not in sys.path:                             # appended: no shadowing of later imports (BigMoE 01:1x)
    sys.path.append(_P98)
from bof import qmm16bof, qmm16bof2

ON = [False]
TABLE = {(2560, 6144): (4, 1), (13312, 2560): (8, 1), (2560, 640): (4, 1), (248320, 2560): (4, 1)}
TABLE32 = {}        # C10 (Sun 18:2x): {(2560, 6144): (4, 1)} turns on BOF2 for o_proj / out_proj at 17-32 rows (bof2_bench)
CNT = dict(bof=0, bof2=0, drift=0)
_S = {}


VER = {}            # (form, N, K, bs, rg) -> True (checked equal) / False (drift: off for this process)


def _ref16(x2, w, s, b):
    """ref2's own <= 16-row route for these shapes (4-bit g64, not DENSE-N): qmm16c ks 1 on the 16-row pad"""
    from qmm16 import qmm16c
    M = x2.shape[0]
    xp = x2 if M == 16 else mx.concatenate([x2, mx.zeros((16 - M, x2.shape[1]), x2.dtype)], 0)
    return qmm16c(xp, w, s, b)[:M]


def _ref32(x2, w, s, b):
    """17-32 rows as v32's split / v32one's qmm16c2 give them: two qmm16c ks-1 tiles"""
    return mx.concatenate([_ref16(x2[:16], w, s, b), _ref16(x2[16:], w, s, b)], 0)


def _bits_eq(a, b):
    """bit for bit (as ref2.self_test's eq): -0.0 vs +0.0 differ, equal NaN payloads match"""
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.all(
        mx.view(a.reshape(-1), mx.uint16) == mx.view(b.reshape(-1), mx.uint16)).item())


def _first(key, y, yref):
    ok = _bits_eq(y, yref)
    VER[key] = ok
    if not ok:
        CNT["drift"] += 1
        print(f"[bofi] DRIFT: {key} != the reference route bit for bit; this form stays OFF in this process", flush=True)
    return y if ok else yref


def _qmmb(x, w, *a, **kw):
    if ON[0]:
        RF = _S["ref2"]
        RC = sys.modules.get("refc")
        scales = a[0] if len(a) > 0 else kw.get("scales")
        biases = a[1] if len(a) > 1 else kw.get("biases")
        transpose = a[2] if len(a) > 2 else kw.get("transpose", True)
        group_size = a[3] if len(a) > 3 else kw.get("group_size")
        bits = a[4] if len(a) > 4 else kw.get("bits")
        mode = a[5] if len(a) > 5 else kw.get("mode", "affine")
        K, N = int(x.shape[-1]), int(w.shape[0])
        rows = math.prod(x.shape[:-1])
        if (RF._ON[0] and not (RC is not None and RC.ON[0]) and transpose and mode == "affine" and biases is not None
                and group_size == 64 and bits == 4 and x.dtype == mx.bfloat16 and w.ndim == 2 and w.dtype == mx.uint32
                and scales.dtype == mx.bfloat16 and RF.CFG["dense_min_n"] <= N and not RF.CFG["dense_native"]
                and (not RF._DNR or RF._DNR.get(id(w)) is None)):
            if 1 <= rows <= 16 and (N, K) in TABLE:
                bs, rg = TABLE[(N, K)]
                key = ("bof", N, K, bs, rg)
                if VER.get(key) is not False:
                    x2 = x.reshape(rows, K)
                    y = qmm16bof(x2, w, scales, biases, bs=bs, rg=rg, term=0)
                    if key not in VER:
                        y = _first(key, y, _ref16(x2, w, scales, biases))
                    CNT["bof"] += 1
                    return y.reshape(*x.shape[:-1], N)
            elif 16 < rows <= 32 and (N, K) in TABLE32:
                V = sys.modules.get("v32")
                if V is not None and V._live(rows):
                    bs, rg = TABLE32[(N, K)]
                    key = ("bof2", N, K, bs, rg)
                    if VER.get(key) is not False:
                        x2 = x.reshape(rows, K)
                        y = qmm16bof2(x2, w, scales, biases, bs=bs, rg=rg, term=0)
                        if key not in VER:
                            y = _first(key, y, _ref32(x2, w, scales, biases))
                        CNT["bof2"] += 1
                        return y.reshape(*x.shape[:-1], N)
    return _S["qmm"](x, w, *a, **kw)


def install(ref2):
    """after ref2.apply() (and v32 / v32one / refc if used): idempotent; ON[0] stays False"""
    if _S:
        return
    _S.update(ref2=ref2, qmm=mx.quantized_matmul)
    mx.quantized_matmul = _qmmb


def self_test(seed=0):
    """synthetic, no model: per small TABLE shape, qmm16bof == qmm16c ks 1 on the pad at 1 / 7 / 16 rows, and each row
    alone == the same row inside a 16-row call (SD == AR at the kernel); per TABLE32 shape, qmm16bof2 == two qmm16c
    tiles at 17 / 24 / 32 rows. lm_head is left to the first-call guard (its weights alone are ~0.36 GB).
    Returns (bad, lines)."""
    import numpy as np
    p = os.path.expanduser("~/Leap/scripts")       # qmm16 (the reference route) when run standalone
    if p not in sys.path:
        sys.path.append(p)
    rng = np.random.default_rng(seed)
    lines, bad = [], 0
    shapes = [(nk, TABLE[nk], "bof") for nk in TABLE if nk[0] <= 20000]
    shapes += [(nk, v, "bof2") for nk, v in {(2560, 6144): (4, 1), **TABLE32}.items()]
    for (N, K), (bs, rg), form in shapes:
        wf = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(wf, group_size=64, bits=4)
        for M in ((1, 7, 16) if form == "bof" else (17, 24, 32)):
            x2 = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
            if form == "bof":
                y, r = qmm16bof(x2, wq, s, b, bs=bs, rg=rg, term=0), _ref16(x2, wq, s, b)
                eq = _bits_eq(y, r)
                full = qmm16bof(mx.concatenate([x2, mx.zeros((16 - M, K), x2.dtype)], 0) if M < 16 else x2, wq, s, b,
                                bs=bs, rg=rg, term=0)
                alone = all(_bits_eq(qmm16bof(x2[i:i + 1], wq, s, b, bs=bs, rg=rg, term=0)[0], full[i])
                            for i in range(M))
            else:
                y, r = qmm16bof2(x2, wq, s, b, bs=bs, rg=rg, term=0), _ref32(x2, wq, s, b)
                eq, alone = _bits_eq(y, r), True
            bad += not (eq and alone)
            lines.append(f"  {form} N {N} K {K} bs {bs} rg {rg} rows {M:2d}: == reference {eq}; rows alone == in the "
                         f"call {alone} -> {'PASS' if eq and alone else 'FAIL'}")
    return bad, lines


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "self_test":
    b_, ls_ = self_test()
    print("\n".join(ls_) + f"\nbofi self_test: {'PASS' if not b_ else 'FAIL'}")
    sys.exit(1 if b_ else 0)
