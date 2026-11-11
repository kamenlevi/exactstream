"""Fit12 GP3 (2026-10-24): pipelined marginal cost of each piece of B12LL's MoE block, by doing that piece twice on the
verify rows (S > 1) and making the original's consumer wait for the copy (mx.depends: same values, so shas must equal U1).
The pass time added by an arm = that piece's cost on the critical path (GP2: the synced profile cannot split this side).
Patched into bt_b12._install inside this process (Breakthrough's file is read, never edited); apply(B) before B.load_b12.
F12_DUP = comma list of:
  tg   tier_glu again, y waits for it                       (expert GEMVs + combine)
  rt   router + top-20 sort again, the post waits for it     (gate GEMV, softmax, argpartition, sorts, gathers)
  sh   shared expert again, wherever the original lands      (0 added -> it already hides behind the loader stall)
  sel  the post-spin selection chain again, tier_glu waits   (~15 small kernels: cumsum, argsort, gathers, where)
  k10  10 tiny dependent kernels after the spin, tier_glu waits (one small kernel launch on the GPU, x 10)
  hca / hcm / hcw   (GP4) the attention HC read / the MLP HC read (with the pending attention write) / the residual
       write again, in B12's layer_call; the next consumer waits for the copy
F12_DUPP (after load, apply_model(model)): gdnp = the 36 GDN layers' projections (in_proj_qkv / z / b / a, out_proj) twice,
attp = the 12 attention layers' q / k / v / o projections twice; on S > 1 calls, the output waits for the copy.
gdnpx / attpx (GP6): the copy uses the next such layer's same-shape weights (an SLC-proof re-read).
F12_ND=n (D2, BigMoE LEDGER D1-3): the MTP head's draft calls (L == NL, one row) wait only for ranks < n; the other
ranks take resident experts from its top-20 (B12's own 'first resident of the rest'). Drafts change, the verify decides
every token -> the output must stay U1's (checked by sha).
Not with F12_OOO (both rewrite the same source)."""
import inspect, os

DUP = {s for s in os.environ.get("F12_DUP", "").split(",") if s}
ND = int(os.environ.get("F12_ND", "0") or 0)
_TG = '        y = TG(x2, t["base"], t["sideP"], s10, l2, plan="".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel\n'
_SH = '            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)\n        x2 = x.reshape(S, x.shape[-1])\n'
_RT = '        p20 = mx.take_along_axis(sc, o, axis=-1)\n'
_SEL = '        l2 = mx.where(used, mx.take_along_axis(sl, pos, axis=-1), sl[:, :1]) if PLAN4 else sl   # [S, 10] (PLAN4)\n'
_FL = '        fl = (FL_ALL if last else FL) if one else FL_MR               # verify rows: late-eligible ranks are waited for\n'
_FO = '        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted\n'
_SL = '        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest\n'
_REPL = [
    (_TG, _TG +
     "        if 'tg' in _F12_DUP and not one:                              # [Fit12 GP3] tier_glu twice\n"
     "            y = mx.depends(y, [TG(x2, t[\"base\"], t[\"sideP\"], s10, l2, plan=\"\".join(T.PLAN), scores=w_now)])\n"),
    (_SH, _SH +
     "        if 'sh' in _F12_DUP and S > 1:                                # [Fit12 GP3] shared expert twice\n"
     "            shared = mx.depends(shared, [attention_gate(self.shared_expert(x), self.shared_expert_gate(x))\n"
     "                                         if current_attention_phase() == \"decode_verify\" else\n"
     "                                         mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)])\n"),
    (_RT, _RT +
     "        if 'rt' in _F12_DUP and S > 1:                                # [Fit12 GP3] router + top-20 again, pre-post\n"
     "            g2 = mx.softmax(self.gate(x2), axis=-1, precise=True)\n"
     "            i2 = mx.argpartition(g2, kth=-KC, axis=-1)[..., -KC:]\n"
     "            c2 = mx.take_along_axis(g2, i2, axis=-1)\n"
     "            o2 = mx.argsort(-c2, axis=-1)\n"
     "            ids = mx.depends(ids, [mx.take_along_axis(i2, o2, axis=-1), mx.take_along_axis(c2, o2, axis=-1)])\n"),
    (_SEL, _SEL +
     "        if 'sel' in _F12_DUP and not one:                             # [Fit12 GP3] selection chain again\n"
     "            pool2 = ((sm >= 0) & (sl >= 0) if PLAN4 else (sm >= 0)) & ~forced\n"
     "            sel2 = (pool2 & (mx.cumsum(pool2.astype(mx.int32), axis=-1) <= K - NF)) | forced\n"
     "            pos2 = mx.argsort(mx.where(sel2, R, R + 1000), axis=-1)[:, :K]\n"
     "            used2 = mx.take_along_axis(sel2, pos2, axis=-1)\n"
     "            w2 = mx.where(used2, mx.take_along_axis(p20, pos2, axis=-1), 0.0)\n"
     "            s10 = mx.depends(s10, [mx.where(used2, mx.take_along_axis(sm, pos2, axis=-1), sm[:, :1]),\n"
     "                                   w2 / p20[:, :K].sum(axis=-1, keepdims=True)])\n"
     "        if 'k10' in _F12_DUP and not one:                             # [Fit12 GP3] 10 tiny kernels after the spin\n"
     "            z = sm[:1, :1].astype(mx.float32)\n"
     "            for _ in range(10):\n"
     "                z = z * 1.5\n"
     "            s10 = mx.depends(s10, [z])\n"),
    ("        mixed, hyper, inject = self.attn_hyper_connection(hidden)\n",
     "        mixed, hyper, inject = self.attn_hyper_connection(hidden)\n"
     "        if 'hca' in _F12_DUP and hidden.shape[1] > 1:                 # [Fit12 GP4] attention HC read again\n"
     "            mixed = mx.depends(mixed, [self.attn_hyper_connection(hidden)[0]])\n"),
    ("        mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))\n",
     "        _h0, _i0 = hyper, inject\n"
     "        mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))\n"
     "        if 'hcm' in _F12_DUP and mixed.shape[1] > 1:                  # [Fit12 GP4] MLP HC read again\n"
     "            mixed = mx.depends(mixed, [self.mlp_hyper_connection(_h0, pending=(block_out, _i0))[0]])\n"),
    ("        hidden = _hyper_residual_write(hyper, block_out, inject)\n",
     "        hidden = _hyper_residual_write(hyper, block_out, inject)\n"
     "        if 'hcw' in _F12_DUP and hidden.shape[1] > 1:                 # [Fit12 GP4] residual write again\n"
     "            hidden = mx.depends(hidden, [_hyper_residual_write(hyper, block_out, inject)])\n"),
    (_FL, _FL +
     "        nf_eff = NF\n"
     "        if _F12_ND and one and L == NL:                               # [Fit12 D2] the MTP head drafts on ranks < ND\n"
     "            nf_eff = _F12_ND                                          # + resident experts (drafts only: exact output)\n"
     "            fl = mx.where(R < _F12_ND, W20, 0).astype(mx.int32)[None]\n"),
    (_FO, "        forced = (R < nf_eff)[None]                                   # [Fit12 D2] nf_eff = NF except D2 drafts\n"),
    (_SL, "        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - nf_eff)) | forced   # [Fit12 D2]\n"),
]


def apply(B):
    if not DUP and not ND:
        return False
    src = inspect.getsource(B._install)
    for old, new in _REPL:
        assert src.count(old) == 1, ("fit_dup: bt_b12 changed", old[:70])
        src = src.replace(old, new)
    B.__dict__["_F12_DUP"] = DUP
    B.__dict__["_F12_ND"] = ND
    exec(compile(src, B.__file__ + " [fit_dup]", "exec"), B.__dict__)
    return True


DUPP = {s for s in os.environ.get("F12_DUPP", "").split(",") if s}
_PROJ = {"gdnp": ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")),
         "attp": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))}
_PROJ["gdnpx"], _PROJ["attpx"] = _PROJ["gdnp"], _PROJ["attp"]   # x: the copy runs the NEXT such layer's same-shape weights
                                                                  # (untouched for ~a pass: no SLC hit on the re-read)


def apply_model(model):
    """F12_DUPP: class-level wrappers keyed by module id (trunk layers only); returns the number of wrapped modules."""
    if not DUPP:
        return 0
    import mlx.core as mx
    ids, classes, part = set(), set(), {}
    for k in DUPP:
        mixer, names = _PROJ[k]
        for n in names:
            mods = [l[mixer][n] for l in model.language_model.model.layers if mixer in l and n in l[mixer]]
            for i, m in enumerate(mods):
                ids.add(id(m)); classes.add(type(m))
                part[id(m)] = mods[(i + 1) % len(mods)] if k.endswith("x") else m
    for cls in classes:
        orig = cls.__call__

        def call(self, x, *a, _orig=orig, **kw):
            y = _orig(self, x, *a, **kw)
            if id(self) in ids and x.ndim == 3 and x.shape[1] > 1:
                y = mx.depends(y, [_orig(part[id(self)], x, *a, **kw)])
            return y
        cls.__call__ = call
    return len(ids)
