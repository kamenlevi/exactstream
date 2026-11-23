"""v32one.py (Leap, Sun 15 Nov 2026 11:4x; CHECK55 step 1, data/week98/check55_plan.md): the exact 17-32-row check with
ONE dispatch per op where v32 runs two 16-row calls. install(v32, ref2) after v32.install(ref2); off by default for every
loop (BigMoE imports v32 live: nothing here runs unless a script installs it).
Each piece is bitwise per row with REF2's certified 16-row form, tested on its own (data/week98/screens/README.md):
  - quantized matmuls (one32_test): the shapes REF2 sends to qmm16c (4-bit g64, N >= 1024, N % 64 == 0) -> qmm16c2; the
    shapes REF2 leaves to MLX's 16-row call -> MLX's qmv_wide_impl at k_lanes 8 over 32 rows (qmvw32), only for the
    (N, K, bits) verified there; anything else (or DENSE-N / DNR weights) keeps v32's split;
  - the HC read with block inject on ref2's leapk path (one32_hc_test) -> hc_read32b; the head mixer keeps v32's split;
  - a GDN layer with R-83's exact in-proj (one32_gdn_test): its qkv | z and b | a calls as above, gdn_body32, the
    capture-commit stash at n rows, out_proj as above; anything else keeps v32's split;
  - an attention layer at exactly 32 rows: the module once, its verify SDPA through ref2's class logic generalized to 32
    rows on 2-head chunks (att32_inv: 1248 / 1248 rows); 17-31 rows keep v32's split (no cache padding).
C3 (c3_v32.py with V32_ONE=1) tests the whole forward and the commit against <= 16-row checks."""
import os
import sys
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98p/Leap_scripts/week98"))
import one32

R, HI = 16, 32
CNT = dict(qmm16c2=0, qmvw32=0, qmm_split=0, hc1=0, hc_split=0, gdn1=0, gdn_split=0, att1=0, att_split=0, vsdpa32=0)
QMV_KL = {(96, 2560, 4): 8, (512, 2560, 4): 8, (640, 2560, 4): 8, (512, 2560, 8): 8, (1281, 2560, 4): 8,
          (640, 2560, 8): 8, (1, 2560, 4): 8, (1, 2560, 8): 8, (48, 2560, 4): 8}
W1 = [True]        # True: the per-shape forms one32_bench picked (cold weights, engine counts; bitwise all); False: C7's
QMV_VPT = [4]      # C7's qmv_wide vectors per threadgroup (W1 False); C7b: 32 for every shape LOST (+0.22 x A16)
BIG_NK = 8_000_000                     # qmm16c shapes with N x K >= 8M (>= 4 MB at 4 bits) take qmm16c2a (one weight pass)
QMV_BEST = {(512, 2560, 8): 8, (1281, 2560, 4): 8, (512, 2560, 4): 0, (96, 2560, 4): 2, (640, 2560, 4): 8,
            (640, 2560, 8): 8, (1, 2560, 4): 2, (1, 2560, 8): 2, (48, 2560, 4): 2}   # vectors per tg; 0 = MLX's split
_S = {}
_ATT32 = [False]
ON = [True]                                              # timing harnesses switch the one-dispatch forms off / on
MISS = {}                                                # (N, K, bits, group, mode, dtype, biases) of split matmuls


def _qmm1(x, w, *a, **kw):
    V, RF = _S["v32"], _S["ref2"]
    rows = V._rows(x)
    if not (ON[0] and V._live(rows)):
        return _S["qmm"](x, w, *a, **kw)
    scales = a[0] if len(a) > 0 else kw.get("scales")
    biases = a[1] if len(a) > 1 else kw.get("biases")
    transpose = a[2] if len(a) > 2 else kw.get("transpose", True)
    group_size = a[3] if len(a) > 3 else kw.get("group_size")
    bits = a[4] if len(a) > 4 else kw.get("bits")
    mode = a[5] if len(a) > 5 else kw.get("mode", "affine")
    lead, K = x.shape[:-1], x.shape[-1]
    N = int(w.shape[0])
    ok = (transpose and mode == "affine" and biases is not None and group_size == 64 and x.dtype == mx.bfloat16
          and not RF.CFG["dense_native"] and not RF.CFG["small_native"] and not RF.CFG["pad_in"]
          and (not RF._DNR or RF._DNR.get(id(w)) is None))
    if ok:
        x2 = x.reshape(rows, K)
        if bits == 4 and N >= RF.CFG["dense_min_n"] and N % 64 == 0 and K % 64 == 0:
            CNT["qmm16c2"] += 1
            f = one32.qmm16c2a if (W1[0] and N * K >= BIG_NK) else one32.qmm16c2
            return f(x2, w, scales, biases).reshape(*lead, N)
        if (N, K, bits) in QMV_KL:
            vpt = QMV_BEST.get((N, K, bits), 2) if W1[0] else QMV_VPT[0]
            if vpt:
                CNT["qmvw32"] += 1
                return one32.qmvw32(x2, w, scales, biases, bits, QMV_KL[(N, K, bits)], vpt).reshape(*lead, N)
    CNT["qmm_split"] += 1
    key = (N, K, bits, group_size, mode, str(x.dtype), "biases" if biases is not None else "nobias")
    MISS[key] = MISS.get(key, 0) + 1                  # diagnostics only: which shapes still split
    return _S["qmm"](x, w, *a, **kw)


def _gr1(self, hyper_input, pending=None):
    V, RF = _S["v32"], _S["ref2"]
    rows = V._rows(hyper_input)
    if not (ON[0] and V._live(rows)):
        return _S["gr"](self, hyper_input, pending)
    BF = mx.bfloat16
    ok = (RF.CFG["hc_gate"] and not RF.CFG["hc8_fast"] and not RF.CFG["hc_pend_mlp"] and "block_inject_weight" in self
          and hyper_input.dtype == BF and (pending is None or all(t.dtype == BF for t in pending))
          and RF._hc8_params(self) is None)
    if not ok:
        CNT["hc_split"] += 1
        return _S["gr"](self, hyper_input, pending)
    lead = hyper_input.shape[:-1]
    pad = lambda t: t if t.shape[0] == HI else mx.concatenate([t, mx.zeros((HI - t.shape[0],) + t.shape[1:], t.dtype)], 0)
    h32 = pad(hyper_input.reshape(rows, -1)).reshape(1, HI, -1)
    p32 = None if pending is None else tuple(pad(p.reshape(rows, -1)).reshape(1, HI, -1) for p in pending)
    if id(self) not in RF._WCAT:
        RF._WCAT[id(self)] = mx.concatenate([self.input_mix_weight_down.weight, self.block_inject_weight.weight], axis=0)
    one32.S4_FORM[0] = "2i" if W1[0] else "2"         # one32_bench: interleaved S4 tiles, second read ~free
    mixed, written, inject = one32.hc_read32b(self, RF._WCAT[id(self)], h32, p32)
    CNT["hc1"] += 1
    RF.COUNTS["hc_read"] += 1
    return (mixed[0, :rows].reshape(*lead, -1), written[0, :rows].reshape(hyper_input.shape),
            inject[0, :rows].reshape(*lead, -1))


def _gdn1(self, inputs, mask=None, cache=None):
    """ref2._gdn_body_call's one-dispatch body for 17-32 rows (the stash, the cache update and out_proj as there)"""
    V, RF = _S["v32"], _S["ref2"]
    import mtplx.models.qwen4_exp as Q
    B, S = inputs.shape[0], inputs.shape[1]
    if not (ON[0] and B == 1 and mask is None and cache is not None and V._live(S)):
        return _S["gdn"](self, inputs, mask, cache)
    fi = self.__dict__.get("in_proj_fused")
    ok = (RF.CFG["gdn_exact"] and RF.CFG["gdn_inproj_x"] and isinstance(fi, RF._InProjExact) and fi.bits == 4
          and fi.gs == 64 and fi.mode == "affine" and inputs.dtype == mx.bfloat16 and cache[0] is not None
          and cache[1] is not None and cache[1].dtype == mx.float32 and RF._gdnx_ok(self, B, mask, cache)
          and isinstance(self.norm, Q.SigmoidRMSNormGated) and self.norm.weight.dtype == mx.bfloat16
          and RF._lz_form(cache) != 3 and (not RF._DNR or RF._DNR.get(id(fi.qz[0])) is None)
          and not RF.CFG["dense_native"] and (int(fi.ba[0].shape[0]), int(inputs.shape[-1]), 4) in QMV_KL
          and not self._fused_out_applies(B, S) and not self._fused_step_applies(B, S, mask, cache)
          and not Q._boundary_capture_offsets(S, B, cache, mask))
    if not ok:
        CNT["gdn_split"] += 1
        return _S["gdn"](self, inputs, mask, cache)
    x2 = inputs.reshape(S, -1)
    nqz, kx = int(fi.qz[0].shape[0]), int(x2.shape[-1])
    yqz = (one32.qmm16c2a if (W1[0] and nqz * kx >= BIG_NK) else one32.qmm16c2)(x2, *fi.qz)
    kba = (int(fi.ba[0].shape[0]), kx, 4)
    yba = one32.qmvw32(x2, *fi.ba, 4, QMV_KL[kba], (QMV_BEST.get(kba, 2) or 2) if W1[0] else QMV_VPT[0])
    n0, n2 = fi.n[0], fi.n[2]
    qkv, z = yqz[:, :n0].reshape(1, S, -1), yqz[:, n0:].reshape(1, S, -1)
    b, a = yba[:, :n2].reshape(1, S, -1), yba[:, n2:].reshape(1, S, -1)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    y, q, k, v, cs, st = one32.gdn_body32(qkv, z, a, b, cache[0], self.conv1d.weight, self.A_log, self.dt_bias, cache[1],
                                          self.norm.weight, self.norm.eps, RF._gdnx()._scale())
    if Q._VERIFY_CAPTURE.get():                        # MTPLX's capture-commit stash, as _gdn_body_call leaves it
        cache._mtplx_verify_rows = (qkv, q, k, v, a, b)
    cache[0] = cs
    cache[1] = st
    cache.advance(S)
    CNT["gdn1"] += 1
    RF.COUNTS["gdnb"] += 1
    return Q._linear(self.out_proj, y.reshape(B, S, -1))


def _vsdpa1(q, k, v, *, scale, mask):
    """inside _att1 only: ref2._verify_sdpa's class logic for a 32-row call (att32_inv's cand32 on 2-head chunks)"""
    RF = _S["ref2"]
    import mtplx.models.qwen4_exp as Q
    S, N = q.shape[2], k.shape[2]
    if not (_ATT32[0] and S == HI and mask is not None and mask.dtype == mx.bool_ and RF.CFG["sdpa_classes"]
            and not RF.CFG["fatt_verify"] and not RF.CFG["att_single"]):
        return _S["vsdpa"](q, k, v, scale=scale, mask=mask)
    fn = lambda q_, k_, v_, mask_: Q._sdpa_head_chunked(q_, k_, v_, scale=scale, mask=mask_, heads_per_chunk=2)

    def at(n):
        if n == N:
            return fn(q, k, v, mask)
        if n < N:
            return fn(q, k[:, :, :n], v[:, :, :n], mask[..., :n])
        pad = n - N
        kp = mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, k.shape[3]), k.dtype)], axis=2)
        vp = mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, v.shape[3]), v.dtype)], axis=2)
        mp = mx.concatenate([mask, mx.zeros(mask.shape[:-1] + (pad,), mx.bool_)], axis=-1)
        return fn(q, kp, vp, mp)

    N1 = N - S + R
    runN = lambda c, n: c if c >= RF.CFG["kb_from"] else n if c == 0 else max(n, RF.CFG["sdpa_bounds"][c - 1])
    c0 = RF._klass(N)
    cls = [RF._klass(N - S + R + j) for j in range(S)]
    out = at(runN(c0, N))
    for c in sorted(set(cls) - {c0}):
        sel = mx.array([cls[j] == c for j in range(S)])[None, None, :, None]
        out = mx.where(sel, at(runN(c, N) if c > c0 else runN(c, N1)), out)
        RF.COUNTS["sdpa_split"] += 1
    CNT["vsdpa32"] += 1
    return out


def _att1(self, x, cache):
    V = _S["v32"]
    S = x.shape[1]
    if not (ON[0] and x.shape[0] == 1 and V._live(S) and S == HI):
        if x.shape[0] == 1 and V._live(S):
            CNT["att_split"] += 1
        return _S["att"](self, x, cache)
    _ATT32[0] = True
    try:
        out = V._W["att"](self, x, cache)              # ref2's _att -> MTPLX's forward at 32 rows (no padding needed)
    finally:
        _ATT32[0] = False
    CNT["att1"] += 1
    return out


def install(v32, ref2):
    """after v32.install(ref2): v32's wrappers become the fallbacks of the one-dispatch forms (idempotent)"""
    import mtplx.models.qwen4_exp as Q
    if not v32._W:
        raise RuntimeError("v32one: v32.install(ref2) first")
    if _S:
        return
    _S.update(v32=v32, ref2=ref2, qmm=mx.quantized_matmul, gr=Q.GatedResidual.__call__, gdn=Q.GatedDeltaNet.__call__,
              att=Q.Attention.__call__, vsdpa=Q._verify_sdpa)
    mx.quantized_matmul = _qmm1
    Q.GatedResidual.__call__ = _gr1
    Q.GatedDeltaNet.__call__ = _gdn1
    Q.Attention.__call__ = _att1
    Q._verify_sdpa = _vsdpa1
