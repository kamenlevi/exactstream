"""B12 (NewIdeas' PLAN_B candidate for one Eval suite decode verdict, Breakthrough 2026-10-21): frugal routing at ~10 % on
BigMoE's v3 tier engine, in a pinned copy (~/Breakthrough/v3f = BigMoE 7b8c7b4 bigmoe/ + ext/ as bigmoe_v3f, plus a
flag-mode loader: bit 20 = read and wait, bit 21 = late-eligible, unflagged = used only if resident; tierglu.py and hc8.py
pinned from MoonShots). Levers, each an env switch so the suite can run B12 minus one without a rebuild:
  B12_CM=51 B12_CL=20   MSB-3 base slots / LSB-1 sidecar slots per layer (plan 4433333333: ranks 0-1 at 4 bits, the rest
                        MSB-3; RAM ~ 48 x (CM x 2.15 + CL x 0.77) MB). Default CM = round(FRAC x 512).
  frugal (always)       router top-20 by score; ranks 0-1 read and waited for; the first 10 resident in rank order are
                        used (missing ranks 2-9 -> resident ranks 10-19), weights renormalised over those 10 (w2f).
  B12_HC=4 | 8 | 0      hyper-connection mixers (+ PLE K/V) quantised to 4 / 8 bits (MoonShots' patches/hc8); default 8
                        (8-bit: KLD +0.0000 hard / +0.0001 easy, -0.62 GB, not slower; 4-bit: +0.024 hard. Quantised mixers turn
                        BigMoE's HC fusions off; B12_HC=0 = bf16 mixers with the compiled HC).
  B12_LATE1=1 | 0       (default 0: no speed gain on v3) wait for rank 0 only; a rank 1 with either record part cold is read after phase A and its output
                        (on layer L's MoE input, rank 1's renormalised weight, at 4 bits) is written into the HC streams
                        after layer L+1's attention, before its MLP read (tf_frugal_var.py l1t2); default 1.
  B12_PF=1 [B12_SPARE=4] prefetch arm 1 (NewIdeas I8): layer L+1's corrected d=1 top-2 per row posted on L's base and
                        sidecar rows (both record parts), prefetched after L's demand reads (non-JIT), spares from C.
  B12_PRE=1 | 0         TrainLab's pre-routed frugal (replaces late rank 1; default 0 until validated): in layers 8-39
                        (B12_PRE_BAND) the two waited (4-bit) experts are layer L's router + MoEPrefetch's correction
                        on layer L-1's MoE input, top-2; the rest = the true router's best outside them, resident only;
                        weights = the TRUE router's probabilities over the chosen 10, renormalised. Here the pre-picks
                        ride in layer L's own row (quality-faithful; the one-layer slack needs an early post).
  B12_LATE3=2,4         "B8" (NewIdeas): MSB-3 ranks 2-3, when cold, are read after phase A and added half a layer
                        late (base pool only, BIGMOE_DEFER = 2 pairs); with FRAC 0.05 = B8 (frugal 5 % on v3).
Agreed configs (NewIdeas 22:05, HC8 default from 01:00): B12 = load_b12(0.10) with the defaults (8-bit mixers, no late);
  B8 = load_b12(0.05) with B12_LATE3=2,4; B8-fast = load_b12(0.05) with B12_W=3 (wait for ranks 0-2, no phase B).
Decode 1-row calls take this path; prefill and multi-row calls stay v3's exact paths. Note: the compiled module is named
bigmoe_ext like BigMoE's: import nothing from ~/BigMoE/bigmoe in the same process.
API (Eval suite): load_b12(frac=0.10) -> (rt, st).   CLI: bt_b12.py FRAC speed (PROMPT code|prose, BT_N) | check | gsmN"""
import glob, hashlib, json, os, re, sys, time

V3F = os.path.expanduser("~/TrainLab/d1i/v3f")             # D1i (TrainLab): a copy of Breakthrough's v3f with the D1i patch
K, KC = 10, 20
STASH, PENDH = {}, {}
CFG = {}


def _env():
    os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
    for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
        os.environ.setdefault(k, "0")
    os.environ["BIGMOE_TIER"] = os.environ.get("B12_PLAN", "4433333333")   # TrainLab: 3333333333 = ranks 0-1 from MSB-3 only
    os.environ.setdefault("BIGMOE_TIER_GLU", "3c")
    os.environ["BIGMOE_SPARE"] = str(CFG["spare"]) if CFG["pf"] else "0"   # frugal: no background reads, unless B12_PF
    os.environ.pop("BIGMOE_PF_JIT", None)            # the loader turns JIT on if the variable exists at all
    os.environ["BIGMOE_PF_NOPUBLISH"] = "1"
    os.environ["BIGMOE_DEFER"] = str(CFG["l3"][1] - CFG["l3"][0]) if CFG["l3"] else ("1" if CFG["late1"] else "0")   # pairs


def load_b12(frac=0.10, mtp=False):
    cm = int(os.environ.get("B12_CM", "0")) or round(frac * 512)
    CFG.update(frac=frac, hc=int(os.environ.get("B12_HC", "8")), late1=os.environ.get("B12_LATE1", "0") == "1",
               cm=cm, cl=int(os.environ.get("B12_CL", "0")) or round(0.4 * cm),
               pre=os.environ.get("B12_PRE", "0") == "1",
               band=tuple(int(v) for v in os.environ.get("B12_PRE_BAND", "8,40").split(",")),
               l3=tuple(int(v) for v in os.environ["B12_LATE3"].split(",")) if os.environ.get("B12_LATE3") else None,
               w=int(os.environ.get("B12_W", "2")),
               pf=os.environ.get("B12_PF") == "1", spare=int(os.environ.get("B12_SPARE", "4")))
    # TrainLab: w = 1 allowed (flag mode: bit 20 on rank 0 only, ranks 1-9 resident-only; Breakthrough 05:05) when no
    # late path is on; otherwise as before
    assert (CFG["w"] >= 2 or (CFG["w"] == 1 and not CFG["l3"] and not CFG["late1"])) and \
        (not CFG["l3"] or CFG["l3"][0] >= CFG["w"]), (CFG["w"], CFG["l3"])
    if CFG["pre"]:
        CFG["late1"] = False                         # pre-routing replaces late rank 1 (both decide the waited pair)
    if CFG["l3"]:                                    # B8: MSB-3 ranks A..B-1 late (base pool only); not with late rank 1
        assert CFG["l3"][0] >= 2 and CFG["l3"][1] > CFG["l3"][0] and not CFG["late1"], CFG["l3"]
    _env()
    os.environ["BIGMOE_TIER_CM"], os.environ["BIGMOE_TIER_CL"] = str(CFG["cm"]), str(CFG["cl"])
    if CFG["hc"]:     # hc8 swaps the mixers for quantised ones: MTPLX's eager HC path only (hc8.py: not with BigMoE's HC
        for k in ("BIGMOE_COMPILE_HC", "BIGMOE_HC_VREAD", "BIGMOE_HC_PACK"):   # fusions; COMPILE_HC's kernels expect
            os.environ[k] = "0"                                                  # bf16 mixers: Metal build error)
    if V3F not in sys.path:
        sys.path.insert(0, V3F)
    import mlx.core as mx
    from bigmoe_v3f.loader import load_streamed
    from bigmoe_v3f import tier as T
    md = os.path.expanduser("~/BigMoE/models/bare")
    cfg = json.load(open(os.path.join(md, "config.json"))); cfg = cfg.get("text_config", cfg)
    NL = cfg["num_hidden_layers"]
    rt, st = load_streamed(md, CFG["cm"], mtp=mtp)
    model = rt.model if hasattr(rt, "model") else rt
    TM = model.language_model.model if hasattr(model, "language_model") else model.model
    for i, l in enumerate(TM.layers):
        l._b12_L = i                                  # layer index for the late writes (layer_call)
    CFG["routers"] = [l.mlp.gate for l in TM.layers]   # B12_PRE: layer L+1's router, applied at layer L
    if CFG["hc"]:
        import hc8                                   # v3f/hc8.py (MoonShots 3055bf7)
        print("b12: hc", hc8.install(model, bits=CFG["hc"]), flush=True)
        mx.clear_cache()
    _install(mx, T, NL)
    if os.environ.get("B12_CACHE_MB"):                # cap MLX's buffer cache AFTER loading (MTPLX's memory plan sets
        mx.set_cache_limit(int(os.environ["B12_CACHE_MB"]) << 20)   # its own at load); freed buffers count as RAM
    print(f"b12: C_m={CFG['cm']} C_l={CFG['cl']} ({100 * CFG['cm'] / 512:.0f} %), HC {CFG['hc'] or 'bf16'}, late rank 1 "
          f"{'on' if CFG['late1'] else 'off'}, late MSB-3 ranks {CFG['l3'] or 'none'}, pools {st.resident_gb():.2f} GB, "
          f"active {mx.get_active_memory() / 1e9:.2f} GB",
          flush=True)
    return rt, st


def _install(mx, T, NL):
    from mtplx.models import qwen4_exp as Q
    from mtplx.models.qwen4_exp import _hyper_residual_write
    from mtplx.attention_context import current_attention_phase
    from mtplx.attention_math import attention_gate
    TG = T._tier_glu()
    orig = Q.SparseMoeBlock.__call__                 # v3's tier block (prefill, multi-row): installed by load_streamed
    R = mx.arange(KC)
    W20, L21 = 1 << 20, 1 << 21
    PREP, CORR, ROUT = {}, [], []
    R22 = mx.arange(KC + 2)
    FL22 = mx.where(R22 < 2, W20, 0).astype(mx.int32)[None]          # pre-routed rows: the two pre-picks waited
    if CFG["pre"] or CFG["pf"]:                      # layer L+1's routers + MoEPrefetch's d=1 correction (source-indexed)
        import numpy as np
        z = np.load(os.path.expanduser(os.environ.get("BIGMOE_PF_CORR", "~/MoEPrefetch/data/corr125_r128.npz")))
        CORR = [(mx.array(z["A"][i]), mx.array(z["B"][i]), mx.array(z["mu"][i]), mx.array(z["delta"][i]))
                for i in range(z["A"].shape[0])]
        ROUT = CFG["routers"]
    A3, B3 = CFG["l3"] or (0, 0)
    W = CFG["w"]                                                      # B12_W: ranks < W read and waited for (default 2)
    R1 = (L21 if CFG["late1"] else W20) if W >= 2 else 0               # TrainLab: W = 1 -> rank 1 resident-only
    FL = mx.where(R == 0, W20, mx.where(R == 1, R1,
                                        mx.where(R < W, W20, mx.where((R >= A3) & (R < B3), L21, 0)))).astype(mx.int32)[None]
    NF = max(2 if W >= 2 else 1, B3, W)                               # ranks 0..NF-1 always among the 10 used
    FL_MR = mx.where(R < NF, W20, 0).astype(mx.int32)[None]          # verify rows: every forced rank waited
    from bigmoe_v3f.gated import MAXK
    SMAX = MAXK // KC                                                 # rows per call on the frugal path
    FL_ALL = mx.where(R < max(NF, CFG["w"]), W20, 0).astype(mx.int32)[None]   # last layer: nothing late, all forced waited
    WK_M, WK_L = 255 | (K << 8), 255 | (2 << 8)                       # flag mode; LRU-touch ranks < 10 / < 2

    def late3_fn(x2, wl, sml, dep_m, n_m, t, gate, L):
        """B8: the late MSB-3 ranks A3..B3-1 of layer L (base pool only), after layer L+1's attention."""
        Mp = B3 - A3

        def f(after):
            o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
            pr = gate.spin(mx.concatenate([dep_m[:1] + o, mx.array([n_m, 0] * Mp, mx.int32)]), L, 2 * Mp, phase=1)
            p, s = pr[0::2], pr[1::2]                                 # (position, slot) pairs; pads (n, 0)
            match = p[None, :] == mx.arange(A3, B3)[:, None]          # [late ranks, pairs]
            sl_late = mx.where(match, s[None, :], 0).sum(axis=-1)     # each late rank's slot (0 if it was resident)
            ms = mx.where(sml[0] >= 0, sml[0], sl_late)[None]         # resident at phase A: weight 0 here anyway
            return TG(x2, t["base"], t["sideP"], ms, None, plan="3" * Mp, scores=wl).reshape(1, 1, -1)
        return f

    def late_fn(x2, wl, sm1, sl1, dep_m, dep_l, n_m, n_l, t, gate, L):
        def f(after):                                                 # spins ordered after layer L+1's attention
            o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
            pm = gate.spin(mx.concatenate([dep_m[:1] + o, mx.array([n_m, 0], mx.int32)]), L, 2, phase=1)
            pl = gate.spin(mx.concatenate([dep_l[:1] + o, mx.array([n_l, 0], mx.int32)]), t["side"], 2, phase=1)
            ms = mx.where(sm1 >= 0, sm1, pm[1:2].reshape(1, 1))       # each record part: phase A's slot if it was in,
            ls = mx.where(sl1 >= 0, sl1, pl[1:2].reshape(1, 1))       # else the late read's
            return TG(x2, t["base"], t["sideP"], ms, ls, plan="4", scores=wl).reshape(1, 1, -1)   # weight 0 unless late
        return f

    def pre_routed(x, x2, gates, ids, pre, L, t, gate):
        """B12_PRE: the pre-picked pair (read and waited, 4-bit), then the true router's best resident outside it."""
        row = mx.concatenate([pre, ids], axis=-1)                     # [1, 22]: pre-picks, then the true top-20
        dep_m, n_m = gate.post(L, row | FL22, wk=WK_M)
        dep_l, n_l = gate.post(t["side"], pre | FL22[:, :2], wk=WK_L)
        sm = gate.spin(dep_m, L, n_m).reshape(1, KC + 2)
        sl = gate.spin(dep_l, t["side"], n_l).reshape(1, 2)
        dup = (ids == pre[:, 0:1]) | (ids == pre[:, 1:2])             # a pre-pick also in the true top-20: used once
        pool = (sm[:, 2:] >= 0) & ~dup
        selr = pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - 2)
        sel = mx.concatenate([mx.ones((1, 2), mx.bool_), selr], axis=-1)
        pos = mx.argsort(mx.where(sel, R22, R22 + 1000), axis=-1)[:, :K]
        used = mx.take_along_axis(sel, pos, axis=-1)
        s10 = mx.where(used, mx.take_along_axis(sm, pos, axis=-1), sm[:, :1])
        ids10 = mx.take_along_axis(row, pos, axis=-1)
        w10 = mx.where(used, mx.take_along_axis(gates, ids10, axis=-1), 0.0)   # the TRUE router's probabilities
        w10 = w10 / w10.sum(axis=-1, keepdims=True)
        y = TG(x2, t["base"], t["sideP"], s10, sl, plan="".join(T.PLAN), scores=w10)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        S = x.shape[-2] if x.ndim == 3 else 0
        if bm is None or "tier" not in bm or x.ndim != 3 or x.shape[0] != 1 or not 1 <= S <= SMAX:
            return orig(self, x)                                      # prefill / long verify: v3's exact tier paths
        L, t, gate = bm["L"], bm["tier"], bm["gate"]
        if L == bm["first"]:
            gate.new_pass()
        ae = bm.get("async_every", 0)
        if ae and L % ae == ae - 1:
            mx.async_eval(x)
        if current_attention_phase() == "decode_verify":             # v3's shared-expert form per phase (numerics)
            shared = attention_gate(self.shared_expert(x), self.shared_expert_gate(x))
        else:
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        x2 = x.reshape(S, x.shape[-1])
        gates = mx.softmax(self.gate(x2), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-KC, axis=-1)[..., -KC:]
        sc = mx.take_along_axis(gates, inds, axis=-1)
        o = mx.argsort(-sc, axis=-1)
        ids = mx.take_along_axis(inds, o, axis=-1).astype(mx.int32)   # [S, 20] rank order per row
        p20 = mx.take_along_axis(sc, o, axis=-1)
        if CFG["pre"] and S == 1:
            b0, b1 = CFG["band"]
            if b0 <= L + 1 < b1 and L < len(CORR):                    # pick layer L+1's waited pair from this input
                A, Bm, mu, dl = CORR[L]
                x32 = x2.astype(mx.float32)
                lg = ROUT[L + 1](x2).astype(mx.float32) + ((x32 - mu) @ A.astype(mx.float32)) @ Bm.astype(mx.float32) + dl
                PREP[L + 1] = mx.argpartition(-lg, kth=1, axis=-1)[:, :2].astype(mx.int32)   # [1, 2]
            pre = PREP.pop(L, None)
            if pre is not None:
                return pre_routed(x, x2, gates, ids, pre, L, t, gate) + shared
        last = L == NL - 1
        one = S == 1
        fl = (FL_ALL if last else FL) if one else FL_MR               # verify rows: late-eligible ranks are waited for
        rows = (ids | fl).T.reshape(-1)                               # rank-major: every row's rank 0, then rank 1, ...
        pk = {}
        if CFG["pf"] and L + 1 < NL and L < len(CORR):               # B12_PF arm 1: layer L+1's corrected d=1 top-2 per
            A, Bm, mu, dl = CORR[L]                                   # row, prefetched (both record parts) after L's
            x32 = x2.astype(mx.float32)                               # demand reads (non-JIT loader, spares from C)
            lg = ROUT[L + 1](x2).astype(mx.float32) + ((x32 - mu) @ A.astype(mx.float32)) @ Bm.astype(mx.float32) + dl
            pr = mx.argpartition(-lg, kth=1, axis=-1)[:, :2].reshape(-1).astype(mx.int32)
            pk = dict(preds=pr, counts=(2 * S,))
        dep_m, n_m = gate.post(L, rows, wk=255 | ((K * S) << 8), **pk)   # flag mode; LRU-touch every row's ranks < 10
        dep_l, n_l = gate.post(t["side"], (ids[:, :2] | fl[:, :2]).T.reshape(-1), wk=255 | ((2 * S) << 8), **pk)
        sm = gate.spin(dep_m, L, n_m).reshape(KC, S).T                # [S, 20] base (MSB-3) slots, -1 = absent / late
        sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T         # [S, 2] LSB sidecar slots of ranks 0-1
        forced = (R < NF)[None]                                       # ranks 0-1 (+ waited / late ranks) always counted
        pool = (sm >= 0) & ~forced
        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced   # + first resident of the rest
        pos = mx.argsort(mx.where(sel, R, R + 1000), axis=-1)[:, :K]   # the 10 used positions, rank order (0, 1, ...)
        used = mx.take_along_axis(sel, pos, axis=-1)                  # fewer than 10 resident: the rest are padding
        s10 = mx.take_along_axis(sm, pos, axis=-1)
        s10 = mx.where(used, s10, s10[:, :1])                         # padding reads rank 0's slots at weight 0
        w10 = mx.where(used, mx.take_along_axis(p20, pos, axis=-1), 0.0)
        w10 = w10 / w10.sum(axis=-1, keepdims=True)                   # renormalised over the 10 used (w2f)
        l2 = sl
        if CFG["late1"] and not last and one:
            late = (sm[:, 1:2] < 0) | (sl[:, 1:2] < 0)                # rank 1 not in at phase A (either record part)
            s10 = mx.concatenate([s10[:, :1], mx.where(late, s10[:, :1], s10[:, 1:2]), s10[:, 2:]], axis=-1)
            l2 = mx.concatenate([sl[:, :1], mx.where(late, sl[:, :1], sl[:, 1:2])], axis=-1)
            w_now = mx.concatenate([w10[:, :1], mx.where(late, 0.0, w10[:, 1:2]), w10[:, 2:]], axis=-1)
            STASH[L] = late_fn(x2, mx.where(late, w10[:, 1:2], 0.0), sm[:, 1:2], sl[:, 1:2], dep_m, dep_l, n_m, n_l,
                               t, gate, L)
        elif CFG["l3"] and not last and one:                          # B8: cold ranks A3..B3-1 land half a layer late
            late = sm[:, A3:B3] < 0                                    # [1, B3-A3] (positions = ranks for r < NF)
            s10 = mx.concatenate([s10[:, :A3], mx.where(late, s10[:, :1], s10[:, A3:B3]), s10[:, B3:]], axis=-1)
            w_now = mx.concatenate([w10[:, :A3], mx.where(late, 0.0, w10[:, A3:B3]), w10[:, B3:]], axis=-1)
            STASH[L] = late3_fn(x2, mx.where(late, w10[:, A3:B3], 0.0), sm[:, A3:B3], dep_m, n_m, t, gate, L)
        else:
            w_now = w10
        y = TG(x2, t["base"], t["sideP"], s10, l2, plan="".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel
        return y.reshape(*x.shape[:-1], y.shape[-1]) + shared

    def layer_call(self, hidden, *, input_ids, ssm_mask, cache):     # MTPLX DecoderLayer.__call__ + the late writes
        if "ple" in self:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        mixed, hyper, inject = self.attn_hyper_connection(hidden)
        if self.is_linear:
            block_out = self.linear_attn(mixed, ssm_mask, cache)
        else:
            block_out = self.self_attn(mixed, cache)
        L = self._b12_L
        for f, inj in PENDH.pop(L, []):                               # layer L-1's late rank 1, before this MLP read
            hyper = _hyper_residual_write(hyper, f(block_out), inj)
        mixed, hyper, inject = self.mlp_hyper_connection(hyper, pending=(block_out, inject))
        block_out = self.mlp(mixed)
        hidden = _hyper_residual_write(hyper, block_out, inject)
        f = STASH.pop(L, None)
        if f is not None:
            PENDH.setdefault(L + 1, []).append((f, inject))
        return hidden

    Q.SparseMoeBlock.__call__ = block
    if CFG["late1"] or CFG["l3"]:
        Q.DecoderLayer.__call__ = layer_call


if __name__ == "__main__":
    for _k, _v in (("BIGMOE_COMPILE_HC", "1"), ("BIGMOE_HC_VREAD", "1"), ("BIGMOE_ASYNC_EVERY", "1")):
        os.environ.setdefault(_k, _v)                                 # speed flags for the CLI only (the suite refuses them)
    frac, TASK = float(sys.argv[1]), sys.argv[2]
    SPEC = TASK.startswith("spec")                                    # spec3:3:0 = MTP k:max_draft:lookup (spec125)
    if SPEC:
        for _k, _v in (("BIGMOE_MTP_STREAM", "1"), ("BIGMOE_MTP_C", "64"), ("BIGMOE_MAXK", "80"), ("BIGMOE_FRSPEC", "1"),
                       ("MTPLX_FAMILY_CAPTURE_COMMIT", "1"), ("MTPLX_DROP_EVENTS", "1"), ("MTPLX_LAZY_MTP_HISTORY_APPEND", "1"),
                       ("MTPLX_DEFER_REPAIR_EVAL", "1")):
            os.environ.setdefault(_k, _v)
    rt, st = load_b12(frac, mtp=SPEC)
    import mlx.core as mx
    from bigmoe_v3f.gated import ext
    model, tok = rt.model, rt.tokenizer
    EOS = set()
    for _a in ("eos_token_ids", "eos_token_id"):
        _v = getattr(tok, _a, None)
        if _v is not None:
            EOS |= set(_v) if isinstance(_v, (list, set, tuple)) else {_v}
    for _t in ("<|im_end|>", "<|endoftext|>"):
        try:
            _i = tok.convert_tokens_to_ids(_t)
            if isinstance(_i, int) and _i >= 0: EOS.add(_i)
        except Exception:
            pass

    def chat_ids(text):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
        return list(ids if isinstance(ids[0], int) else ids["input_ids"])

    def prefill_decoupled(ids, cache, s_attn, s_moe, keep=False):
        """D1i (3), TrainLab: per layer, the attention part (PLE, HC read, attention, HC write-into-read) in query chunks of
        s_attn tokens, then the MoE ONCE over the whole s_moe chunk (its experts read once per layer per s_moe tokens),
        then the HC residual write per query chunk. Same ops on the same shapes as a chunk-s_attn prefill except the MoE
        block, whose per-token outputs do not depend on the batch (grouped / sorted gather per pair, per-token combine),
        so it should equal B12_PREFILL_CHUNK=s_attn bit for bit while reading experts like chunk s_moe."""
        import mtplx.models.qwen4_exp as Q
        from contextlib import nullcontext
        from bigmoe_v3f import tier as _T
        # B8's late writes (B12_LATE3 / B12_LATE1: bt_b12's layer_call) come only from decode rows (the frugal block takes
        # S <= SMAX); prefill rows take v3's exact paths, so the late queues stay empty and MTPLX's own layer pieces apply
        assert not STASH and not PENDH, "decoupled prefill: late writes pending"
        tm = model.language_model.model
        P = ids.shape[1]
        scope = Q._rope_table_scope() if Q.qwen4_opdiet_enabled("rope") else nullcontext()
        st = lambda c: [a for a in (c.state if hasattr(c, "state") else []) if isinstance(a, mx.array)]
        SPLIT = os.environ.get("D1I_SPLIT", "0") == "1"

        def moe_split(blk, xs_):
            """mlx_lm Qwen3NextSparseMoeBlock.__call__ (the prefill path of the tier layers: bt_b12's block -> v3's tier
            block -> MTPLX SparseMoeBlock -> super) with router, shared expert and combine per query chunk (the
            matmul shapes of a chunk-s_attn prefill) and the routed experts ONCE over all chunks (grouped, per-pair)."""
            k, parts = blk.top_k, []
            for x in xs_:
                gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if blk.norm_topk_prob:
                    scores = scores / scores.sum(axis=-1, keepdims=True)
                parts.append((inds, scores))
            y = blk.switch_mlp(mx.concatenate(xs_, axis=1) if len(xs_) > 1 else xs_[0],
                               mx.concatenate([p_[0] for p_ in parts], axis=1) if len(xs_) > 1 else parts[0][0])
            outs, off = [], 0
            for x, (inds, scores) in zip(xs_, parts):
                n_ = x.shape[1]
                yi = (y[:, off:off + n_] * scores[..., None]).sum(axis=-2)
                shared_y = blk.shared_expert(x)
                shared_y = mx.sigmoid(blk.shared_expert_gate(x)) * shared_y
                outs.append(yi + shared_y); off += n_
            return outs

        kept = []
        with scope:
            for m0 in range(0, P, s_moe):
                m1 = min(m0 + s_moe, P)
                # D1I_STREAM=1: fixed-order streaming of every layer's experts (tier.start_stream), for MoE chunks of
                # >= D1I_STREAM_MIN tokens; reads are pumped between the attention launches below
                SM = os.environ.get("D1I_STREAM", "0")
                stream = None
                if SM in ("1", "2", "3") and m1 - m0 >= int(os.environ.get("D1I_STREAM_MIN", "1024")):
                    bms_ = [l.mlp.switch_mlp._bm for l in tm.layers]
                    R_ = int(os.environ.get("D1I_RING", "4"))
                    th_ = int(os.environ.get("D1I_THREADS", "8" if SM == "2" else "16"))
                    stream = (_T.start_stream2(bms_, R=R_, threads=th_) if SM == "2" else
                              _T.start_stream3(bms_, R=R_, threads=th_) if SM == "3" else _T.start_stream(bms_, R=R_))
                xs = [ids[:, a:min(a + s_attn, m1)] for a in range(m0, m1, s_attn)]
                hs = [mx.tile(tm.embed_tokens(x), (1, 1, tm.args.hc_count)) for x in xs]
                masks = [Q.create_ssm_mask(h, cache[tm.ssm_idx]) for h in hs]
                for L, (layer, c) in enumerate(zip(tm.layers, cache)):
                    _ta = time.perf_counter()
                    mixed, hyper, inject = [], [], []
                    for i, x in enumerate(xs):
                        if L == tm._ple_stage_idx and Q._COMPILED_VERIFY_PLE.get() is None:
                            layer.ple.ple_embedding.stage(x, c, layer.ple.NGRAM_IDX)
                        h = hs[i]
                        if "ple" in layer:
                            h = h + layer.ple(h, x, c)
                        mx_, hy, inj = layer.attn_hyper_connection(h)
                        bo = layer.linear_attn(mx_, masks[i], c) if layer.is_linear else layer.self_attn(mx_, c)
                        mx_, hy, inj = layer.mlp_hyper_connection(hy, pending=(bo, inj))
                        if stream is not None:
                            mx.async_eval(mx_, hy, inj, *st(c))
                            stream.pump(1) if SM == "1" else None  # v1: a group read while the GPU runs attention
                                                                   # (v2 reads in its own threads)
                        else:
                            mx.eval(mx_, hy, inj, *st(c))
                        mixed.append(mx_); hyper.append(hy); inject.append(inj)
                    _tb = time.perf_counter()
                    _T.PF["t_attn_build"] = _T.PF.get("t_attn_build", 0.0) + (_tb - _ta)
                    if SPLIT:                                  # dense MoE parts per query chunk: exact vs chunk s_attn
                        bos = moe_split(layer.mlp, mixed)
                    else:
                        bo = layer.mlp(mx.concatenate(mixed, axis=1) if len(mixed) > 1 else mixed[0])
                        bos, off = [], 0
                        for x in xs:
                            bos.append(bo[:, off:off + x.shape[1]]); off += x.shape[1]
                    _tc = time.perf_counter()
                    for i in range(len(xs)):
                        hs[i] = Q._hyper_residual_write(hyper[i], bos[i], inject[i])
                    mx.eval(*hs)
                    _T.PF["t_post"] = _T.PF.get("t_post", 0.0) + (time.perf_counter() - _tc)
                    _T.PF["t_mlp_call"] = _T.PF.get("t_mlp_call", 0.0) + (_tc - _tb)
                    del mixed, hyper, inject, bos
                if stream is not None:
                    for k_, v_ in _T.stop_stream().items():
                        _T.PF["s_" + k_] = round(_T.PF.get("s_" + k_, 0) + v_, 2)
                tm._last_widened = hs[-1]
                if keep:                                               # (span, final widened stream) per query chunk
                    kept += [((a, a + x.shape[1]), h) for a, x, h in zip(range(m0, m1, s_attn), xs, hs)]
                del hs
                mx.clear_cache()
        return kept

    def d1i_prefill_state(prompt_ids, s_attn, s_moe):
        """spec125-compatible prefilled state (spec125.prefill_state's keys: cache, mtp_cache, logits, h) from the
        decoupled trunk prefill; the MTP head's history is appended per query chunk as spec125._prefill does per
        prefill chunk (draft logits for the last row only). Pass as spec125.generate(..., prefilled=state)."""
        sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
        import spec125
        cache = model.make_cache()
        ids = mx.array([list(prompt_ids)[:-1]])
        tp = time.perf_counter()
        kept = prefill_decoupled(ids, cache, s_attn, s_moe, keep=True)
        head = spec125._MTPHead(model)
        for (a, b), h in kept:
            head.append(h, [int(v) for v in prompt_ids[a + 1:b + 1]])
            mx.eval(head.logits)
        del kept
        mx.clear_cache()
        PREF.update(n=ids.shape[1], t=time.perf_counter() - tp)
        return {"cache": cache, "mtp_cache": head.cache, "logits": head.logits, "h": head.h}

    def generate(prompt_ids, n, stop_eos=False, per_tok=None):
        from bigmoe_v3f.fuse import reset_generation_state
        from bigmoe_v3f import tier as _T
        reset_generation_state(model)
        STASH.clear(); PENDH.clear()
        out, cache = [], model.make_cache()
        ids = mx.array([list(prompt_ids)[:-1]])
        PFC = int(os.environ.get("B12_PREFILL_CHUNK", "2048"))      # smaller chunks: smaller exact-prefill scratch
        NOLOG = os.environ.get("D1I_NOLOGITS") == "1"               # D1i: prefill chunks skip the vocab head (unused)
        tp = time.perf_counter()
        S_ATTN = int(os.environ.get("D1I_ATTN", "0") or 0)            # D1i (3): decoupled attention / MoE chunks
        for i in (range(0, ids.shape[1], PFC) if not S_ATTN else []):
            _T.PF["last"] = i + PFC >= ids.shape[1]                  # D1i warming: the last chunk's experts -> pools
            if NOLOG:
                model(ids[:, i:i + PFC], cache=cache, emit_logits=False)
                mx.eval([c.state for c in cache if hasattr(c, "state")])
            else:
                lg = model(ids[:, i:i + PFC], cache=cache)
                mx.eval(lg[0] if isinstance(lg, tuple) else lg)
        if S_ATTN:
            prefill_decoupled(ids, cache, S_ATTN, int(os.environ.get("D1I_MOE", "4096")))
        _T.PF["last"] = False
        PREF.update(n=ids.shape[1], t=time.perf_counter() - tp)
        y = int(prompt_ids[-1])
        ext.reset_stats(); ext.defer_reset()
        m0, t0 = ext.stats()["misses"], time.perf_counter()
        while len(out) < n:
            lg = model(mx.array([[y]]), cache=cache)
            y = int(mx.argmax((lg[0] if isinstance(lg, tuple) else lg)[0, -1]).item())
            if stop_eos and y in EOS:
                break
            out.append(y)
            if per_tok is not None:
                per_tok.append(ext.defer_stats()[1])                 # cumulative waited misses since the decode began
        t = time.perf_counter() - t0
        st.gate.check()
        rows, waited, late, over = ext.defer_stats()
        return out, dict(t=t, n=len(out), reads=ext.stats()["misses"] - m0, waited=waited, late=late, over=over, rows=rows)

    sha = lambda t: hashlib.sha1(str(list(t)).encode()).hexdigest()[:10]
    import resource, ctypes

    class _RU(ctypes.Structure):                                      # rusage_info_v4: uuid, then 64-bit counters
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]

    def footprint_peak():
        """lifetime max physical footprint (ri_lifetime_max_phys_footprint; Activity Monitor's Memory), GB"""
        ru = _RU()
        if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) != 0:
            return float("nan")
        return ru.u28 / 1e9

    mem = lambda: (f"peak active {mx.get_peak_memory() / 1e9:.1f} GB, cache {mx.get_cache_memory() / 1e9:.1f} GB, "
                   f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.1f} GB, "
                   f"PEAK FOOTPRINT {footprint_peak():.1f} GB")
    PREF = {}
    PROMPTS = {"ctx32k": lambda: ("Here are several Python scripts from one repository. List what each does in one line.\n\n"
                                 + "".join(f"### {os.path.basename(f)}\n" + open(f).read() for f in
                                           sorted(glob.glob(os.path.expanduser("~/Breakthrough/scripts/*.py"))))[:115000]),
               "long": lambda: ("Here is a Python script. Summarise what it does, section by section.\n\n```python\n"
                               + open(os.path.expanduser("~/Breakthrough/scripts/tf_frugal_var.py")).read()[:7600] + "\n```"),
               "code": lambda: open(os.path.expanduser("~/BigMoE/data/prompt_code.txt")).read(),
               "prose": lambda: ("Explain, in plain prose for a curious 14-year-old, how a CPU cache works and why it makes "
                                 "programs faster. Use one everyday analogy and keep it to about 300 words.")}
    tag = f"b12 C_m={CFG['cm']} C_l={CFG['cl']} hc{CFG['hc']} late1={int(CFG['late1'])} late3={CFG['l3']} w={CFG['w']} pre={int(CFG['pre'])} pf={int(CFG['pf'])}"
    if TASK == "check":
        out, s = generate(chat_ids(PROMPTS["code"]()), 48)
        print(f"check {tag}: sha {sha(out)} reads {s['reads']} waited {s['waited']} late {s['late']} over {s['over']} | "
              f"{tok.decode(out)[:160]!r}", flush=True)
    elif TASK == "d1i":                                               # D1i (TrainLab): one cold call after load, then a repeat
        import numpy as np
        from bigmoe_v3f import tier as _T
        P = os.environ.get("PROMPT", "long")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "64"))
        tag2 = (f"chunk {os.environ.get('B12_PREFILL_CHUNK', '2048')} attn/moe {os.environ.get('D1I_ATTN', '-')}/{os.environ.get('D1I_MOE', '-')} split {os.environ.get('D1I_SPLIT', '0')} stream {os.environ.get('D1I_STREAM', '0')} "
                f"gpu {os.environ.get('BIGMOE_TIER_PF_GPU', '0')} nologits {os.environ.get('D1I_NOLOGITS', '0')} group "
                f"{os.environ.get('BIGMOE_TIER_PF_GROUP', '0')} warm {os.environ.get('BIGMOE_TIER_WARM', '0')}")
        for run in ("cold", "repeat")[:int(os.environ.get("D1I_RUNS", "2"))]:
            for k_ in ("grouped", "single", "fallback", "groups", "warm_base", "warm_side"):
                _T.PF[k_] = 0
            for k_ in [k for k in _T.PF if k.startswith("s_") or k.startswith("t_")]:
                del _T.PF[k_]
            _T.PF["t_read"] = _T.PF["t_warm"] = 0.0
            _T.PF["t_read"] = _T.PF["t_warm"] = 0.0
            waits = []
            out, s = generate(ids, N, per_tok=waits)
            w = np.diff([0] + waits)
            seg = lambda a, b: float(w[a:b].mean()) if len(w) > a else float("nan")
            pf = {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in _T.PF.items() if k_ != "last"}
            print(f"[{P}] d1i {run} {tag2}: prefill {PREF['n']} tokens in {PREF['t']:.2f} s ({PREF['n'] / PREF['t']:.0f} tok/s); "
                  f"waited misses/token 0-16 {seg(0, 16):.1f} 16-32 {seg(16, 32):.1f} 32-64 {seg(32, 64):.1f}; decode "
                  f"{s['n'] / s['t']:.1f} tok/s, reads/token {s['reads'] / max(1, s['n']):.1f}; {mem()}; sha {sha(out)} | "
                  f"pf {pf} | {tok.decode(out)[:60]!r}", flush=True)
    elif TASK == "speed":
        P = os.environ.get("PROMPT", "code")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "128"))
        generate(ids, 16)
        out, s = generate(ids, N)
        nt = max(1, s["n"])
        print(f"[{P}] {tag}: {s['n'] / s['t']:.1f} tok/s, SSD reads/token {s['reads'] / nt:.1f} (waited misses "
              f"{s['waited'] / nt:.1f}, late {s['late'] / nt:.1f}, overflow {s['over'] / nt:.2f}), peak "
              f"{mx.get_peak_memory() / 1e9:.1f} GB, active {mx.get_active_memory() / 1e9:.1f} GB, {mem()}, sha {sha(out)} | "
              f"{tok.decode(out)[:70]!r}", flush=True)
        print(f"[{P}] prefill {PREF.get('n', 0)} tokens in {PREF.get('t', 0):.2f} s ({PREF.get('n', 0) / max(PREF.get('t', 1e-9), 1e-9):.0f} tok/s, "
              f"chunk {os.environ.get('B12_PREFILL_CHUNK', '2048')})", flush=True)
    elif SPEC:                                                       # MTP speed: verify passes take the multi-row path
        sys.path.insert(1, os.path.expanduser("~/SpecDecode"))
        import spec125
        from bigmoe_v3f.fuse import reset_generation_state
        k, md_, lk = (int(v) for v in TASK[4:].split(":")[:3])
        P = os.environ.get("PROMPT", "code")
        ids = chat_ids(PROMPTS[P]())
        N = int(os.environ.get("BT_N", "128"))

        def run(n):
            reset_generation_state(model); STASH.clear(); PENDH.clear()
            out, first, s = [], {}, {}

            def cb(tk):
                if not first:
                    first.update(t=time.perf_counter(), m=ext.stats()["misses"], st=ext.stats(), d=ext.defer_stats())
                out.append(int(tk))
            pre = PRESTATE.get("s")                                   # D1i: one decoupled prefill shared by the runs
            spec125.generate(model, ids, n, k_mtp=k, max_draft=md_, lookup=lk, stats=s, on_token=cb,
                             prefill_chunk=int(os.environ.get("B12_PREFILL_CHUNK", "2048")), prefilled=pre)
            st.gate.check()
            e1, d1 = ext.stats(), ext.defer_stats()
            cnt = {k: e1[k] - first["st"][k] for k in ("pf_issued", "pf_used", "pf_ontime", "pf_late")}
            cnt["waited"] = d1[1] - first["d"][1]
            s["cnt"] = cnt
            return out, s, time.perf_counter() - first["t"], ext.stats()["misses"] - first["m"]
        PRESTATE = {}
        if os.environ.get("D1I_ATTN") and os.environ.get("D1I_FRESH") == "1":
            # as Breakthrough's D1I_FRESH (4d42a59+): every generate runs the decoupled prefill itself (spec125._prefill
            # swapped), so no shared state is held and deep-copied beside the live cache (the server's footprint)
            S_A, S_M = int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096"))

            def _fresh_prefill(model_, prompt_ids, with_head, prefill_chunk=2048):
                s_ = d1i_prefill_state(prompt_ids, S_A, S_M)
                head = spec125._MTPHead(model_) if with_head else None
                if head is not None:
                    head.cache, head.logits, head.h = s_["mtp_cache"], s_["logits"], s_["h"]
                return s_["cache"], head
            spec125._prefill = _fresh_prefill
        elif os.environ.get("D1I_ATTN"):
            PRESTATE["s"] = d1i_prefill_state(ids, int(os.environ["D1I_ATTN"]), int(os.environ.get("D1I_MOE", "4096")))
            print(f"[{P}] d1i prefilled state: {PREF['n']} tokens in {PREF['t']:.2f} s ({PREF['n'] / PREF['t']:.0f} tok/s), "
                  f"{mem()}", flush=True)
        run(16)                                                       # warm-up
        out, s, tsec, rd = run(N)
        if os.environ.get("D1I_FRESH") == "1" and PREF:
            print(f"[{P}] d1i fresh prefill (no state copy): {PREF['n']} tokens in {PREF['t']:.2f} s "
                  f"({PREF['n'] / PREF['t']:.0f} tok/s)", flush=True)
        nt = max(1, len(out) - 1)                                     # decode from the first token on
        c, npass = s["cnt"], max(1, s.get("passes", 1))
        print(f"[{P}] {tag} spec {TASK[4:]}: {nt / tsec:.1f} tok/s, {s.get('tok_per_pass', float('nan')):.2f} tok/pass, "
              f"SSD reads/token {rd / nt:.1f}; per pass: waited misses {c['waited'] / npass:.1f}, reads {rd / npass:.1f}, "
              f"prefetch issued {c['pf_issued'] / npass:.1f} used {c['pf_used'] / npass:.1f} (on time {c['pf_ontime'] / npass:.1f}, "
              f"late {c['pf_late'] / npass:.1f}); {mem()}, sha {sha(out)} | "
              f"{tok.decode(out)[:70]!r}", flush=True)
    elif TASK.startswith("gsm"):
        N = int(TASK[3:]); OFF = int(os.environ.get("OFFSET", "0"))
        G8 = [json.loads(l) for l in open(os.path.expanduser("~/MoEPrefetch/data/gsm8k/test.jsonl"))][OFF:OFF + N]
        INSTR = "Solve the problem step by step, briefly. End with a line of the form 'Answer: <number>'.\n\n"
        okn, same = 0, 0
        ref = json.load(open(os.path.expanduser("~/MoEPrefetch/data/w125_gsm_eager.json")))["answers"]
        nt = tt = 0
        for j, p in enumerate(G8):
            out, s = generate(chat_ids(INSTR + p["question"]), 600, stop_eos=True)
            text = tok.decode(out)
            m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
            gold = p["answer"].split("####")[-1].strip().replace(",", "")
            a = m[-1].replace(",", "").rstrip(".") if m else None
            try:
                okn += int(a is not None and abs(float(a) - float(gold)) < 1e-6)
            except ValueError:
                pass
            same += int(OFF + j < len(ref) and ref[OFF + j] == a)
            nt += s["n"]; tt += s["t"]
        print(f"125B {tag}: GSM8K {okn}/{len(G8)} (offset {OFF}), answers equal to exact eager {same}, "
              f"{nt / max(tt, 1e-9):.1f} tok/s decode", flush=True)
