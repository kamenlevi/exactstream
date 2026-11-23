"""v32mr2.py (Leap, Sun 15 Nov 2026 01:2x; WEEK98 pillar 2, build step 2 of the 17-32-row path): the GDN layer at 17-32
rows with its projections at FULL width (one weight read: D32's mr2 through v32's qmm wrapper when V32_MR2=1 and the
weight has a DENSE-N / DENSE-N-RAM native copy; else v32's 2 x 16 split) and only the recurrence core chunked: ref2's
one-dispatch GDN body (gdn_body, the <= 16-row form) on rows [0, 16) then [16, n), the conv state and the recurrent state
carried, the capture stash joined. Bits per row are those of the 16-row path: mr2 == qmm16n per row at the same K split
(C2), qmm16n == qmm16c at ks 1 (DENSE-N), and the body runs at <= 16 rows as certified.
A separate module so v32.py stays as C3b tested it (BigMoE's g32 test imports v32). install() after v32.install():
it replaces v32's GDN hook with _gdn32, which hands every call it does not take back to v32's chunked _gdn.
Needs CFG gdn_body (R57_GDNB: exact, bitwise == off) and, for mr2, REF2_DENSE_NATIVE=1 (native copies, +~1.5 GB) or
R57_DNR. Not taken: the lazy GDN commit, fused GDN in-projections other than R-83's _InProjExact, boundary captures."""
import mlx.core as mx

R = 16
CNT = dict(gdn32=0, gdn32_declined=0)
_S = {}


def _eligible(self, inputs, mask, cache, RF, Q):
    B, S = inputs.shape[0], inputs.shape[1]
    return (RF.CFG["gdn_body"] and RF.CFG["gdn_exact"] and not RF.CFG["gdn_lazy_commit"] and B == 1 and mask is None
            and cache is not None and R < S <= 32 and inputs.dtype == mx.bfloat16 and cache[0] is not None
            and cache[1] is not None and cache[1].dtype == mx.float32 and RF._gdnx_ok(self, B, mask, cache)
            and isinstance(self.norm, Q.SigmoidRMSNormGated) and self.norm.weight.dtype == mx.bfloat16
            and not self._fused_out_applies(B, S) and not self._fused_step_applies(B, S, mask, cache)
            and not Q._boundary_capture_offsets(S, B, cache, mask) and RF._lz_form(cache) != 3)


def _gdn32(self, inputs, mask=None, cache=None):
    V, RF, Q = _S["v32"], _S["ref2"], _S["Q"]
    S = inputs.shape[1]
    if not (V._live(S) and _eligible(self, inputs, mask, cache, RF, Q)):
        if V._live(S):
            CNT["gdn32_declined"] += 1
        return V._gdn(self, inputs, mask, cache)
    if cache is not None and getattr(cache, "_ref2_lz_c", None) is not None:
        cache._ref2_lz_c = None
    B = inputs.shape[0]
    fused_in = self.__dict__.get("in_proj_fused") or getattr(self, "in_proj_fused", None)
    if isinstance(fused_in, RF._InProjExact):            # R-83: qkv | z and b | a as two full-width calls
        q = lambda w: mx.quantized_matmul(inputs, *w, transpose=True, group_size=fused_in.gs, bits=fused_in.bits,
                                          mode=fused_in.mode)
        prev, RF._DN_SRC[0] = RF._DN_SRC[0], fused_in.qz[0]
        try:
            yqz = q(fused_in.qz)
        finally:
            RF._DN_SRC[0] = prev
        yba = q(fused_in.ba)
        n0, n2 = fused_in.n[0], fused_in.n[2]
        qkv, z, b, a = yqz[..., :n0], yqz[..., n0:], yba[..., :n2], yba[..., n2:]
    elif fused_in is None and all(k in self for k in RF._GDN_IN):
        qkv, z = self.in_proj_qkv(inputs), self.in_proj_z(inputs)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
    else:
        CNT["gdn32_declined"] += 1
        return V._gdn(self, inputs, mask, cache)
    z = z.reshape(B, S, self.num_v_heads, self.head_v_dim)
    GB, sc = RF._gbmod(), RF._gdnx()._scale()
    w_, A_, dt_, nw_, eps_ = self.conv1d.weight, self.A_log, self.dt_bias, self.norm.weight, self.norm.eps
    y1, q1, k1, v1, cs1, st1 = GB.gdn_body(qkv[:, :R], z[:, :R], a[:, :R], b[:, :R], cache[0], w_, A_, dt_, cache[1],
                                           nw_, eps_, sc)
    y2, q2, k2, v2, cs2, st2 = GB.gdn_body(qkv[:, R:], z[:, R:], a[:, R:], b[:, R:], cs1, w_, A_, dt_, st1, nw_, eps_, sc)
    if Q._VERIFY_CAPTURE.get():                          # MTPLX's capture-commit stash, one window of S rows
        cat = lambda u, v: mx.concatenate([u, v], axis=1)
        cache._mtplx_verify_rows = (qkv, cat(q1, q2), cat(k1, k2), cat(v1, v2), a, b)
    cache[0] = cs2
    cache[1] = st2
    cache.advance(S)
    RF.COUNTS["gdnb"] += 2
    CNT["gdn32"] += 1
    return Q._linear(self.out_proj, mx.concatenate([y1, y2], axis=1).reshape(B, S, -1))


def install(v32, ref2):
    """after v32.install(ref2): the GDN hook becomes _gdn32 (which falls back to v32._gdn)"""
    import mtplx.models.qwen4_exp as Q
    if not v32._W:
        raise RuntimeError("v32mr2: v32.install(ref2) first")
    _S.update(v32=v32, ref2=ref2, Q=Q)
    Q.GatedDeltaNet.__call__ = _gdn32


def uninstall():
    import mtplx.models.qwen4_exp as Q
    if _S:
        Q.GatedDeltaNet.__call__ = _S["v32"]._gdn
