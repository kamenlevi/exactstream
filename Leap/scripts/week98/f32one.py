"""f32one.py (Leap, Mon 16 Nov 2026 12:2x; CHECK25, data/week98/check25_plan.md): the FAST (lossy-45) stack's
one-pass forms for 17-32-row calls, as an opt-in installer on top of ref2 (ref2's file untouched). Today the FAST stack
has no > 16-row forms: DENSE-N-RAM's weights fall to _dn_unpack (181 ms dense-only at 32 rows), SMALL-N / the HC8 read /
SVH / the verify SDPA fall back to MLX's or MTPLX's generic paths (other bits, slower).

Forms (each FAST-safe: per row the arithmetic of ref2's own <= 16-row FAST form, so a 32-row check's rows carry the
bits a 16-row check would give them; tested by scripts/week98/f32one_test.py):
  dense   DNR / DENSE-N / SMALL-N 4-bit weights at 17-32 rows -> qmm32n (one dispatch, one weight pass; 110 / 110
          bitwise == two qmm16n_m calls at the same DN-KS ks). FORM="dn32" swaps in dn32 (not bitwise; timing arm).
  HC read the HC8 FAST read in ONE pass (hc8r32.read32: == hc8r.read v4 on rows 0-15 + 16-31 bit for bit, 18 / 18); other
          HC forms as two 16-row reads (the pending write split by rows)
  SDPA    a 17-32-row verify call as ONE fused SDPA (att_single's form, R-62) instead of MTPLX's 12 head-chunked calls
  SVH     the screened head at 17-32 rows with one32.qmm16c2a on the subset (one weight pass, bitwise == two qmm16c)
  GDN     a 17-32-row GDN layer in ONE call (v32 runs chunk 16 + rest): ref2's _gdn_call at n rows, its qkv | z and b | a
          projections through _InProjExact's concat branch (allowed to 32 rows here) -> the forms above; MTPLX's conv,
          recurrence (one T-row step) and gated norm at n rows; the capture stash is the n-row call's own
  att     a 17-32-row attention layer in ONE call (v32 runs two chunks): q / k / v / o through the forms above, the
          verify SDPA as one fused call
  chunk   (CHUNK[0], FAST) the gated delta rule of a 17-32-row window in chunked form (gdn32c.py: one dispatch per layer,
          the sums reassociated) in place of mlx_lm's sequential step kernel
Layering: ref2.apply() -> v32.install(ref2) (V32_MOE1: the MoE block in one call, the HC read 2 x 16, everything else
split) -> f32one.install(): its wrappers sit on top and act only where v32 would (v32._live: 17-32 rows, the verify phase).
FORM "dn32" swaps the dense form to dn32.py (bf16 x uint4 from threadgroup memory; FAST, not bitwise: -1.1 ms per
check vs qmm32n, data/week98/check25/README.md), reading the native copies directly (staged in the kernel, no copy).
  install(form="qmm32n") / remove() / COUNTS"""
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
import ref2
import qmm32n as Q32

LO, HI = 16, 32
COUNTS = dict(dense32=0, small32=0, dn32=0, hc32=0, hc32x2=0, sdpa32=0, svh32=0, passthru=0, gdn32=0, att32=0, inproj32=0,
              gdn_chunk=0)
_ORIG = {}
FORM = ["qmm32n"]


ON = [True]                               # timing harnesses switch the one-pass forms off / on (installed = on)


def _mid(rows):
    """17-32 rows in the verify phase (v32._live: the verify forward or the commit replay, never the prefill), while ON"""
    import v32
    return ON[0] and v32._live(rows)


_NAXD = {}                                # dn32: id(native weight) -> (native weight, wx, sx, bx)


def _dense32(x2, wn, sn, bn, N, ks):
    """a native 4-bit weight at 17-32 rows: qmm32n (bitwise == two 16-row calls), or dn32 on its NAX-dense copy"""
    if FORM[0] == "dn32":
        import dn32 as DN
        import int8d
        COUNTS["dn32"] += 1
        return DN.dn32(x2, wn, sn, bn, N, int8d._kd(ks, x2.shape[-1]), native=True)
    return Q32.qmm32n(x2, wn, sn, bn, N, ks)


# MLX's own 16-row call for the small shapes ref2 leaves to MLX: qmv_wide at these k_lanes (v32one's table; one32_test:
# bitwise == MLX's 16-row call at every row count); vectors per threadgroup from one32_bench (0 = two MLX calls)
QMV_KL = {(96, 2560, 4): 8, (512, 2560, 4): 8, (640, 2560, 4): 8, (512, 2560, 8): 8, (1281, 2560, 4): 8,
          (640, 2560, 8): 8, (1, 2560, 4): 8, (1, 2560, 8): 8, (48, 2560, 4): 8}
QMV_BEST = {(512, 2560, 8): 8, (1281, 2560, 4): 8, (512, 2560, 4): 0, (96, 2560, 4): 2, (640, 2560, 4): 8,
            (640, 2560, 8): 8, (1, 2560, 4): 2, (1, 2560, 8): 2, (48, 2560, 4): 2}
BIG_NK = 8_000_000                        # qmm16c shapes >= 4 MB at 4 bits take qmm16c2a (one weight pass), else qmm16c2
MISS = {}                                 # (N, K, bits) of 17-32-row calls that still run as two 16-row calls


def _two16(x2, f):
    """two <= 16-row calls of f (rows 0-15, 16-M): the fallback, bitwise to ref2's 16-row form by construction"""
    return mx.concatenate([f(x2[:LO]), f(x2[LO:])], axis=0)


def _qmm32(x, w, *args, **kw):
    """mx.quantized_matmul at 17-32 rows, mirroring ref2._qmm's branches with one-dispatch forms:
    DENSE-N-RAM / DENSE-N -> qmm32n; ref2's qmm16c shapes -> one32.qmm16c2a / qmm16c2; SMALL-N -> qmm32n on the padded
    native copy; MLX's 16-row call -> one32.qmvw32 (bitwise) or two MLX 16-row calls. Every form: ref2's 16-row bits."""
    if not (ref2._ON[0] and x.dtype == mx.bfloat16):
        return ref2._qmm(x, w, *args, **kw)
    lead, rows = ref2._rows(x)
    if not _mid(rows):
        return ref2._qmm(x, w, *args, **kw)
    scales = args[0] if len(args) > 0 else kw.get("scales")
    biases = args[1] if len(args) > 1 else kw.get("biases")
    transpose = args[2] if len(args) > 2 else kw.get("transpose", True)
    group_size = args[3] if len(args) > 3 else kw.get("group_size")
    bits = args[4] if len(args) > 4 else kw.get("bits")
    mode = args[5] if len(args) > 5 else kw.get("mode", "affine")
    e = ref2._DNR.get(id(w)) if ref2._DNR else None
    if e is not None and e[0] is w:                                     # DENSE-N-RAM (ref2._dnr_qmm's conditions)
        _, N, K = e
        if (transpose and mode == "affine" and bits == 4 and group_size == 64 and x.shape[-1] == K
                and scales.dtype == mx.bfloat16):
            COUNTS["dense32"] += 1
            return _dense32(x.reshape(rows, K), w, scales, biases, N, ref2._dn_ks(N, K)).reshape(*lead, N)
        return ref2._qmm(x, w, *args, **kw)                             # (its unpack path, as today)
    if not (transpose and mode == "affine" and biases is not None):
        return ref2._qmm(x, w, *args, **kw)
    K_, N = x.shape[-1], int(w.shape[0])
    x2 = x.reshape(rows, K_)
    if bits == 4 and group_size == 64 and N >= ref2.CFG["dense_min_n"] and N % 64 == 0 and K_ % 64 == 0:
        d = ref2._dn_get(w, scales, biases) if ref2.CFG["dense_native"] else None
        if d is not None:                                               # DENSE-N's native copy
            COUNTS["dense32"] += 1
            return _dense32(x2, d[1], d[2], d[3], N, ref2._dn_ks(N, K_)).reshape(*lead, N)
        import one32                                                    # ref2's qmm16c (KS 1): one32's twins
        COUNTS["passthru"] += 1
        f = one32.qmm16c2a if N * K_ >= BIG_NK else one32.qmm16c2
        return f(x2, w, scales, biases).reshape(*lead, N)
    sn = (ref2._sn_get(w, scales, biases) if (ref2.CFG["small_native"] and bits == 4 and group_size == 64 and K_ % 64 == 0
                                              and scales is not None and scales.dtype == mx.bfloat16) else None)
    if sn is not None:                                                  # SMALL-N's padded native copy
        COUNTS["small32"] += 1
        return _dense32(x2, sn[1], sn[2], sn[3], sn[4], ref2._dn_ks(sn[4], K_))[:, :N].reshape(*lead, N)
    key = (N, K_, bits)                                                 # MLX's 16-row call in ref2
    import one32
    if group_size == 64 and key in QMV_KL and QMV_BEST.get(key, 2):
        COUNTS["passthru"] += 1
        return one32.qmvw32(x2, w, scales, biases, bits, QMV_KL[key], QMV_BEST.get(key, 2)).reshape(*lead, N)
    MISS[key] = MISS.get(key, 0) + 1
    orig = _ORIG["qmm"]
    return _two16(x2, lambda xs: orig(xs, w, *args, **kw)).reshape(*lead, N)


def _gr32(self, hyper_input, pending=None):
    """the HC read at 17-32 rows as two <= 16-row ref2 reads (rows split; D5's carried write resolved first)"""
    if not ref2._ON[0]:
        return _ORIG["gr"](self, hyper_input, pending)
    c = ref2._CARRY[0]
    _, rows = ref2._rows(hyper_input)
    if not _mid(rows):
        return _ORIG["gr"](self, hyper_input, pending)
    if pending is None and c is not None and c[0] is hyper_input and ref2.CFG["hc_pend_mlp"] \
            and "block_inject_weight" in self:
        ref2._CARRY[0] = None
        hyper_input, pending = c[1], (c[2], c[3])
        ref2.COUNTS["hc_pend"] += 1
    lead, rows = ref2._rows(hyper_input)
    h2 = hyper_input.reshape(rows, -1)
    p2 = None if pending is None else tuple(p.reshape(rows, -1) for p in pending)
    q8 = ref2._hc8_params(self) if "block_inject_weight" in self else None
    bf = mx.bfloat16
    if (q8 is not None and ref2.CFG["hc8_fast"] and ref2.CFG["hc8_v"] == 4 and h2.dtype == bf
            and (p2 is None or all(t.dtype == bf for t in p2))):      # HC8 in ONE pass (hc8r32: == two 16-row reads)
        import hc8r32
        z = lambda a: a if a.shape[0] == HI else mx.concatenate([a, mx.zeros((HI - a.shape[0],) + a.shape[1:], a.dtype)])
        mixed, written, inject = hc8r32.read32(q8, z(h2), None if p2 is None else z(p2[0]),
                                               None if p2 is None else z(p2[1]), NP=ref2.CFG["hc8_np"])
        COUNTS["hc32"] += 1
        ref2.COUNTS["hc8_fast"] += 1
        return (mixed[:rows].reshape(*lead, -1), written[:rows].reshape(hyper_input.shape),
                inject[:rows].reshape(*lead, -1))
    parts = ((0, LO), (LO, rows))
    outs = [_ORIG["gr"](self, h2[a:b].reshape(1, b - a, -1),
                        None if p2 is None else tuple(p[a:b].reshape(1, b - a, -1) for p in p2)) for a, b in parts]
    COUNTS["hc32x2"] += 1
    cat = lambda xs: mx.concatenate([o.reshape(b - a, -1) for o, (a, b) in zip(xs, parts)], axis=0)
    if isinstance(outs[0], tuple):
        return (cat([o[0] for o in outs]).reshape(*lead, -1), cat([o[1] for o in outs]).reshape(hyper_input.shape),
                cat([o[2] for o in outs]).reshape(*lead, -1))
    return cat(outs).reshape(*lead, -1)


def _vsdpa32(q, k, v, *, scale, mask):
    """a 17-32-row verify SDPA as one fused call (att_single's form); else ref2's _verify_sdpa"""
    if ref2._ON[0] and ref2.CFG["att_single"] and _mid(q.shape[2]):
        COUNTS["sdpa32"] += 1
        ref2.COUNTS["att_single"] += 1
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return _ORIG["vsdpa"](q, k, v, scale=scale, mask=mask)


def _screened32(e, x, scales, biases):
    """SVH at 17-32 rows: the subset (+ session ids) with qmm16c2a; else ref2's form (<= 16 rows) or None"""
    lead, rows = ref2._rows(x)
    if not (_mid(rows) and x.dtype == mx.bfloat16):
        return _ORIG["screened"](e, x, scales, biases)
    import one32
    x2 = x.reshape(rows, x.shape[-1])
    y = one32.qmm16c2a(x2, e["wb"], e["sb"], e["bb"])
    ids = e["ids"]
    n = len(e["extra"])
    if n:
        yx = one32.qmm16c2a(x2, e["xw"], e["xs"], e["xb"])[:, :n]
        y = mx.concatenate([y, yx], axis=1)
        ids = mx.concatenate([ids, e["xids"][:n]])
    out = mx.put_along_axis(mx.full((rows, e["V"]), -float("inf"), y.dtype),
                            mx.broadcast_to(ids[None].astype(mx.int32), y.shape), y, axis=1)
    COUNTS["svh32"] += 1
    ref2.COUNTS["screened_head"] += 1
    return out.reshape(*lead, e["V"])


CHUNK = [False]                           # the chunked gated delta rule (gdn32c.py, FAST) for 17-32-row windows


def _gdk32(q, k, v, g, beta, state, mask=None):
    """mlx_lm's gated_delta_kernel: a 17-32-row verify window (or its commit replay) in chunked form while CHUNK[0]"""
    if CHUNK[0] and mask is None and g.ndim == 3 and _mid(k.shape[1]) and k.shape[-1] == 128 and v.shape[-1] == 128:
        import gdn32c
        COUNTS["gdn_chunk"] += 1
        return gdn32c.chunk(q, k, v, g, beta, state)
    return _ORIG["gdk"](q, k, v, g, beta, state, mask)


def _gdn32(self, inputs, mask=None, cache=None):
    """a 17-32-row GDN layer in one call (ref2's _gdn_call at n rows); else the installed chain (v32's chunks)"""
    S = inputs.shape[1]
    if inputs.shape[0] == 1 and mask is None and cache is not None and _mid(S):
        COUNTS["gdn32"] += 1
        import v32
        return v32._W["gdn"](self, inputs, mask, cache)
    return _ORIG["gdn"](self, inputs, mask, cache)


def _att32(self, x, cache):
    """a 17-32-row attention layer in one call (ref2's _att at n rows); else the installed chain (v32's two chunks)"""
    if x.shape[0] == 1 and _mid(x.shape[1]):
        COUNTS["att32"] += 1
        import v32
        return v32._W["att"](self, x, cache)
    return _ORIG["att"](self, x, cache)


def _inproj32(self, x):
    """ref2._InProjExact at 17-32 rows in the verify phase: its concat branch (qkv | z and b | a, two calls through
    mx.quantized_matmul -> the forms above), as at <= 16 rows; else ref2's own (the four stock calls at prefill widths)"""
    if ref2._ON[0] and ref2.CFG["gdn_inproj_x"] and _mid(ref2._rows(x)[1]):
        COUNTS["inproj32"] += 1
        q = lambda w: mx.quantized_matmul(x, *w, transpose=True, group_size=self.gs, bits=self.bits, mode=self.mode)
        n0, n2 = self.n[0], self.n[2]
        prev, ref2._DN_SRC[0] = ref2._DN_SRC[0], self.qz[0]
        try:
            yqz = q(self.qz)
        finally:
            ref2._DN_SRC[0] = prev
        prev, ref2._DN_SRC[0] = ref2._DN_SRC[0], self.ba[0]
        try:
            yba = q(self.ba)
        finally:
            ref2._DN_SRC[0] = prev
        return yqz[..., :n0], yqz[..., n0:], yba[..., :n2], yba[..., n2:]
    return _ORIG["inproj"](self, x)


def install(form="qmm32n"):
    """switch the 17-32-row forms on (ref2.apply() first); idempotent"""
    assert ref2._ORIG, "ref2.apply() first"
    FORM[0] = form
    if _ORIG:
        return
    import v32
    assert v32._W and v32.ST["on"], "v32.install(ref2) first (f32one sits on v32's 17-32-row path)"
    _ORIG.update(qmm=mx.quantized_matmul, gr=ref2.Q.GatedResidual.__call__, vsdpa=ref2.Q._verify_sdpa,
                 screened=ref2._screened, gdn=ref2.Q.GatedDeltaNet.__call__, att=ref2.Q.Attention.__call__,
                 inproj=ref2._InProjExact.__call__)
    mx.quantized_matmul = _qmm32
    ref2.Q.GatedResidual.__call__ = _gr32
    ref2.Q._verify_sdpa = _vsdpa32
    ref2._screened = _screened32
    ref2.Q.GatedDeltaNet.__call__ = _gdn32
    ref2.Q.Attention.__call__ = _att32
    ref2._InProjExact.__call__ = _inproj32
    import mlx_lm.models.gated_delta as _GD
    _ORIG["gdk"] = _GD.gated_delta_kernel
    _GD.gated_delta_kernel = _gdk32


def remove():
    if not _ORIG:
        return
    mx.quantized_matmul = _ORIG["qmm"]
    ref2.Q.GatedResidual.__call__ = _ORIG["gr"]
    ref2.Q._verify_sdpa = _ORIG["vsdpa"]
    ref2._screened = _ORIG["screened"]
    ref2.Q.GatedDeltaNet.__call__ = _ORIG["gdn"]
    ref2.Q.Attention.__call__ = _ORIG["att"]
    ref2._InProjExact.__call__ = _ORIG["inproj"]
    import mlx_lm.models.gated_delta as _GD
    _GD.gated_delta_kernel = _ORIG["gdk"]
    _ORIG.clear()
