"""Fit12 TK1 (2026-10-24): fewer experts per token on B12LL's verify rows (Leap's L1-17 on Fit12's captures: top-8 = 0.78x
MSB-3's MoE-block error at -16 % union; adaptive p0.8 1.43x at -24 %, p0.9 0.38x at -10 %). NON-EXACT. Patched into
bt_b12._install in this process (Breakthrough's file is read, never edited); apply(B) before B.load_b12.
  F12_TOPK=k   each verify row keeps its top-k experts (k <= 10)
  F12_TOPP=p   each verify row keeps the smallest rank prefix holding >= p of its top-10 router weight (1..10 experts)
Kept ranks are posted waited (W20), the rest flag 0 (never read); no resident substitutes; the kept weights are renormalized
over the kept set (Leap's definition; replaces B12_NORM=mass's top-10 mass). Dropped positions are B12's own weight-0
padding (rank 0's slots), so they add exactly 0. One-row calls (the late path), the MTP head (v3's non-tier path) and the
prefill (v3's tier path) keep the full top-10. No-ops (bitwise = B12LL): F12_TOPK=10, F12_TOPP=1.0.
Not with F12_OOO / F12_DUP / F12_ND (all rewrite the same source).
TK3 fix (Sun 25 Oct): a row that keeps only rank 0 (top-p's confident rows, or k = 1) left rank 1's LSB sidecar
slot at -1 (not waited), so tier_glu read outside the LSB pool at weight 0 -> NaN (the Eval suite's b12llka80 token-0
floods). Rank 1's LSB slot is now padded with rank 0's (the engine's own late-rank-1 rule); a no-op for k >= 2."""
import inspect, os

TOPK = int(os.environ.get("F12_TOPK", "0") or 0)
TOPP = float(os.environ.get("F12_TOPP", "0") or 0)
_P20 = '        p20 = mx.take_along_axis(sc, o, axis=-1)\n'
_FL = '        fl = (FL_ALL if last else FL) if one else FL_MR               # verify rows: late-eligible ranks are waited for\n'
_FO = '        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted\n'
_SL = '        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest\n'
_L2 = '        l2 = mx.where(used, mx.take_along_axis(sl, pos, axis=-1), sl[:, :1]) if PLAN4 else sl   # [S, 10] (PLAN4)\n'
_NM = '            w10 = w10 / p20[:, :K].sum(axis=-1, keepdims=True)        # weight (probability / the true top-10 mass); Wed TF\n'
_REPL = [
    (_P20, _P20 +
     "        _tk = (_F12_TK[0] or _F12_TK[1]) and S > 1                    # [Fit12 TK1] kept experts per verify row\n"
     "        if _tk and _F12_TK[1]:\n"
     "            _c = mx.cumsum(p20[:, :K], axis=-1) / p20[:, :K].sum(axis=-1, keepdims=True)\n"
     "            _kk = mx.minimum(1 + (_c < _F12_TK[1]).astype(mx.int32).sum(axis=-1), K)\n"
     "        elif _tk:\n"
     "            _kk = mx.full((S,), min(_F12_TK[0], K), mx.int32)\n"),
    (_FL, _FL +
     "        if _tk and not one:\n"
     "            fl = mx.where(R[None] < _kk[:, None], W20, 0).astype(mx.int32)\n"),
    (_FO, "        forced = (R[None] < _kk[:, None]) if (_tk and not one) else (R < NF)[None]   # [Fit12 TK1]\n"),
    (_SL, "        sel = forced if (_tk and not one) else ((pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced)\n"),
    (_L2, _L2 +
     "        if _tk and not one and not PLAN4:                             # [Fit12 TK3] rank 1 dropped: rank 0's LSB slot\n"
     "            l2 = mx.concatenate([sl[:, :1], mx.where((_kk >= 2)[:, None], sl[:, 1:2], sl[:, :1])], axis=-1)\n"),
    (_NM, "            w10 = (w10 / w10.sum(axis=-1, keepdims=True)) if (_tk and not one) else (w10 / p20[:, :K].sum(axis=-1, keepdims=True))\n"),
]


def apply(B, topk=None, topp=None):
    k = TOPK if topk is None else int(topk)
    p = TOPP if topp is None else float(topp)
    if not k and not p:
        return False
    src = inspect.getsource(B._install)
    for old, new in _REPL:
        assert src.count(old) == 1, ("fit_topk: bt_b12 changed", old[:70])
        src = src.replace(old, new)
    B.__dict__["_F12_TK"] = (k, p)
    exec(compile(src, B.__file__ + " [fit_topk]", "exec"), B.__dict__)
    return True
