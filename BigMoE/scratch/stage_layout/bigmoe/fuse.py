"""Optional kernel fusion around the streamed experts (each switch A/B'd for identical tokens).

BIGMOE_COMPILE_HC=1: mx.compile each hyper-connection read (MTPLX GatedResidual: the mixers, norms,
gating and the pending residual write, ~15 small kernels, called twice per layer). Pure function of
its inputs and fixed weights, so it can be traced once per shape; the streamed experts stay outside.
"""
from __future__ import annotations
import os
import mlx.core as mx

_SUB = {}
_CHUNK = int(os.environ.get("BIGMOE_DUPROW_CHUNK", "0"))
_VREAD = os.environ.get("BIGMOE_HC_VREAD") == "1"
_VREAD_MAX = 5 if (os.environ.get("BIGMOE_HC_PIN") == "1" or int(os.environ.get("BIGMOE_DUPROW_CHUNK", "0") or 0)) else 8   # read_rows follows MLX's per-M lanes (16 at M=6); wider reads take the chunked/pinned compiled path
_SIG = None


def _vread(hc, h, pending):
    """BIGMOE_HC_VREAD=1 (MoonShots): the whole HC read in 3 dispatches via MTPLX hc_verify_read.read_rows
    (2..8 rows). S=1 runs on [x; x] like the dup-row reference; bit-identical to the compiled read at S=1/3/6."""
    global _SIG
    from mtplx.kernels import hc_verify_read as V
    if _SIG is None:
        _SIG = V.sigmoid_table(mx.bfloat16)
    S = h.shape[1]
    x = h.reshape(S, -1)
    bo = inj = None
    if pending is not None:
        bo, inj = (p.reshape(S, -1) for p in pending)
    if S == 1:
        x = mx.concatenate([x, x])
        if bo is not None:
            bo, inj = mx.concatenate([bo, bo]), mx.concatenate([inj, inj])
    out = V.read_rows(x, hc.hc_norm.weight, hc.input_mix_weight_down.weight, hc.input_mix_weight_up.weight,
                      hc.block_inject_weight.weight, bo, inj, hc=hc.hc_count, eps=float(hc.hc_norm.eps), sigmoid=_SIG)
    return tuple(a[:S].reshape(1, S, -1) for a in out)


def _compiled_class(cls):
    if cls in _SUB:
        return _SUB[cls]

    def __call__(self, hyper_input, pending=None):
        if _VREAD and "block_inject_weight" in self and hyper_input.ndim == 3 and hyper_input.shape[0] == 1 and hyper_input.shape[1] <= _VREAD_MAX:
            return _vread(self, hyper_input, pending)
        if pending is None:
            return self._bm_f0(hyper_input)
        return self._bm_f1(hyper_input, *pending)

    sub = type(cls.__name__ + "Compiled", (cls,), {"__call__": __call__})
    _SUB[cls] = sub
    return sub


def compile_hyper_connections(model):
    n = 0
    for layer in model.language_model.model.layers:
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            hc = getattr(layer, name, None)
            if hc is None:
                continue
            base = type(hc)
            f0 = mx.compile(lambda h, _hc=hc, _b=base: _b.__call__(_hc, h))
            f1 = mx.compile(lambda h, bo, inj, _hc=hc, _b=base: _b.__call__(_hc, h, (bo, inj)))
            object.__setattr__(hc, "_bm_f0", f0)
            object.__setattr__(hc, "_bm_f1", f1)
            hc.__class__ = _compiled_class(base)
            n += 1
    return n


def duprow():
    """Batch-invariant reference: a 1-row dense linear (quantized or bf16) runs on [x; x] and keeps row 0,
    so decode (S=1) uses the same matmul kernel as verify passes (S>=2). Experts never see the duplicate."""
    import mlx.nn as nn
    for cls in (nn.QuantizedLinear, nn.Linear):
        if getattr(cls, "_bm_dup", False):
            continue
        base = cls.__call__

        dense = cls is nn.Linear

        def _call(self, x, _base=base, _dense=dense):
            if x.ndim >= 2 and x.shape[-2] == 1:
                return _base(self, mx.concatenate([x, x], axis=-2))[..., :1, :]
            # BIGMOE_DUPROW_CHUNK=c (SpecDecode B6): MLX's bf16 dense matmul switches kernels between 5 and 6 rows
            # (2560->2560: {1} {2-5} {6-12}), so verify widths 6..64 run as chunks of <= c rows (a 1-row remainder
            # dup-rowed to 2): every chunk stays in the reference's {2-5} band -> S=6.. reproduce the S<=5 arithmetic.
            if _dense and _CHUNK and x.ndim >= 2 and _CHUNK < x.shape[-2] <= 64:
                M = x.shape[-2]
                parts = []
                for r0 in range(0, M, _CHUNK):
                    xs = x[..., r0:min(r0 + _CHUNK, M), :]
                    if xs.shape[-2] == 1:
                        parts.append(_base(self, mx.concatenate([xs, xs], axis=-2))[..., :1, :])
                    else:
                        parts.append(_base(self, xs))
                return mx.concatenate(parts, axis=-2)
            return _base(self, x)

        cls.__call__ = _call
        cls._bm_dup = True
    # MTPLX_FUSED_GDN_INPROJ=1 builds one fused quantized in_proj (qkv|z|b|a) that bypasses nn.QuantizedLinear, so the
    # 1-row wrap must be applied to it too (MoonShots: bit-identical to the four separate projections for M=1..8)
    try:
        from mtplx.models.qwen4_exp import _FusedGDNInProj as F
    except ImportError:
        F = None
    if F is not None and not getattr(F, "_bm_dup", False):
        fbase = F.__call__

        def _fcall(self, x, _base=fbase):
            if x.ndim >= 2 and x.shape[-2] == 1:
                return [y[..., :1, :] for y in _base(self, mx.concatenate([x, x], axis=-2))]
            return _base(self, x)

        F.__call__ = _fcall
        F._bm_dup = True


def verify_rows_sdpa(lo=int(os.environ.get("BIGMOE_VERIFY_ROWS_LO", "1000")), hi=2048):
    """LONG1 (exact spec past ~1K tokens): MLX's fused vector SDPA runs a 2-pass kernel once the call has >= 1024 keys;
    a verify call (all rows see the full key array, newest keys masked) then differs from the decode call of some rows
    (each saw exactly its own causal length). The probe (scripts/sdpa_band.py, b21, random data at a few T) found only
    windows straddling 1024, but the model also diverged at other T in 1024..2048 when the fix was narrowed to the
    straddle (b22: exact adaptive @1112, v3 k3 @1399), so the fix covers the whole dense range: a verify call (S > 1) with
    lo <= T <= hi runs each row as its own S = 1 call on keys [0, T - S + i] with no mask = the decode call, bit for bit
    (b20: exact and v3 ar2 = k1 = k3 = adaptive on math430 at 1,536 tokens). hi = the QSA indexer budget: past it the
    selection is sparse, and spec == plain held there unpatched (b21: 6K prompt, 512 tokens, both stacks)."""
    import mtplx.models.qwen4_exp as Q
    if getattr(Q, "_bm_rows_sdpa", False):
        return
    base = Q._verify_sdpa

    def rows(q, k, v, *, scale, mask):
        S, T = q.shape[2], k.shape[2]
        if S <= 1 or not (lo <= T <= hi) or mask is None or isinstance(mask, str):
            return base(q, k, v, scale=scale, mask=mask)
        outs = [base(q[:, :, i:i + 1], k[:, :, :T - S + i + 1], v[:, :, :T - S + i + 1], scale=scale, mask=None)
                for i in range(S)]
        return mx.concatenate(outs, axis=2)

    Q._verify_sdpa = rows
    Q._bm_rows_sdpa = True


def apply(model):
    out = {}
    if os.environ.get("BIGMOE_VERIFY_ROWS_SDPA") == "1":
        verify_rows_sdpa(); out["verify_rows_sdpa"] = True
    if os.environ.get("BIGMOE_DUPROW") == "1":
        duprow(); out["duprow"] = True
    if os.environ.get("BIGMOE_HC_PIN") == "1":           # before compiling the HC reads
        from .hcpack import pin_hyper_connections
        out["hc_pinned_linears"] = pin_hyper_connections(model)
    if os.environ.get("BIGMOE_HC_PACK") == "1":          # before compiling the HC reads
        from .hcpack import pack_hyper_connections
        out["hc_packed_linears"] = pack_hyper_connections(model)
    if os.environ.get("BIGMOE_COMPILE_HC") == "1":
        out["compiled_hc"] = compile_hyper_connections(model)
    return out


def reset_generation_state(model):
    """LC1 fix: MTPLX keeps QSA state across generations (compiled indexer cores, functools.lru_cache entries in
    qwen4_exp / kernels.qsa_*), and at long context (T >= 16K) a second generation in the same process reads it, so its
    tokens depend on the previous request. Reset it before every generation: plain -> reset -> spec3 at 32K then equals
    the fresh-process result (a67851027e). Returns (cores reset, caches cleared)."""
    import functools, importlib, pkgutil, sys
    n_core = 0
    for _, m in model.named_modules():
        if hasattr(type(m), "_get_compiled_indexer_core"):
            object.__setattr__(m, "_compiled_indexer_core", None)
            object.__setattr__(m, "_compiled_indexer_parameter_signature", None)
            n_core += 1
    n_lru = 0
    if os.environ.get("BIGMOE_RESET_LRU") != "1":       # night bisect: resetting the cores alone already fixes LC1
        return n_core, n_lru
    import mtplx.models.qwen4_exp as Q, mtplx.kernels as K
    mods = [Q] + [sys.modules[f"mtplx.kernels.{x.name}"] for x in pkgutil.iter_modules(K.__path__)
                  if x.name.startswith("qsa_") and f"mtplx.kernels.{x.name}" in sys.modules]
    for mod in mods:
        for v in vars(mod).values():
            if isinstance(v, functools._lru_cache_wrapper):
                v.cache_clear(); n_lru += 1
    return n_core, n_lru
