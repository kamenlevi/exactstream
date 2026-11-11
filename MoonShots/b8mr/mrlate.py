"""Multi-row phase B for B12-family configs with late MSB-3 ranks (B8 = load_b12(0.05) + B12_LATE3=2,4), for MTP verify passes
(Breakthrough's IDEAS D1m; NewIdeas approved 19:50 Thu). Breakthrough's block (~/Breakthrough/scripts/bt_b12.py, imported
read-only) defers cold ranks A3..B3-1 only in 1-row decode; verify rows (1 < S <= SMAX) post FL_MR and WAIT for every forced
rank, so B8 + MTP pays those reads in every pass. This wraps the installed block: verify rows of 2..SMAX rows (or 1..SMAX in
the S=1 test) take a copy of its frugal path with the 1-row flags FL (ranks < W waited, A3..B3-1 late-eligible, the rest
used only if resident). The cold late ranks get weight 0 in phase A, and a phase-3 call over all rows' late pairs
(plan '3' x Mp, the same TG and weights as late3_fn) runs after layer L+1's attention: v3f's ext publishes up to
BIGMOE_DEFER (position, slot) pairs, positions rank-major = rank x S + row, so the capacity must be >= Mp x SMAX (8 for B8
at SMAX = 4; set by run.py). The late output goes through bt_b12's own STASH / PENDH / layer_call.
Everything else (S == 1 unless MR_MIN=1, the last layer, prefill, S > SMAX, pre-routing, prefetch arms) stays in
bt_b12's block.
Test modes (env MR_TEST):
  force  every rank A3..B3-1 is late, resident or not (phase B runs on every layer)
  ref    force's numerics with the late slots taken synchronously at layer L: phase B is spun at once, the late
         call runs at layer L and only the write waits for layer L+1's attention (force == ref bit for bit)"""
import os


def install(mx, B, T, model):
    """B = bt_b12 module (load_b12 already called), T = bigmoe_v3f.tier, model = rt.model. Returns the stats dict."""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_context import current_attention_phase
    from mtplx.attention_math import attention_gate
    from bigmoe_v3f.gated import MAXK
    CFG, KC, K = B.CFG, B.KC, B.K
    assert CFG["l3"] and not CFG["late1"] and not CFG["pre"] and not CFG["pf"], CFG
    TG = T._tier_glu()
    PLAN = "".join(T.PLAN)
    b12_block = Q.SparseMoeBlock.__call__                          # bt_b12's installed block
    TM = model.language_model.model if hasattr(model, "language_model") else model.model
    NL = len(TM.layers)
    R = mx.arange(KC)
    W20, L21 = 1 << 20, 1 << 21
    A3, B3 = CFG["l3"]
    Mp = B3 - A3
    W = CFG["w"]
    NORM_MASS = CFG.get("norm") == "mass"
    FL = mx.where(R == 0, W20, mx.where(R == 1, W20, mx.where(R < W, W20, mx.where((R >= A3) & (R < B3), L21, 0)))
                  ).astype(mx.int32)[None]                          # bt_b12's 1-row flags (late1 off)
    NF = max(2, B3, W)
    SMAX = MAXK // KC
    MR_MIN = int(os.environ.get("MR_MIN", "2"))
    TEST = os.environ.get("MR_TEST", "")
    assert int(os.environ.get("BIGMOE_DEFER", "0")) >= Mp * SMAX, ("phase-B pair capacity", os.environ.get("BIGMOE_DEFER"))
    ST = {"calls": 0, "rows": 0, "nl": NL}
    TGT = {}

    def late3_mr(x2, wl, sml, dep_m, n_m, t, gate, L, S):
        """the late MSB-3 ranks of layer L for all S rows; pairs (position, slot), position = rank x S + row"""
        if S not in TGT:
            TGT[S] = mx.array([[(A3 + j) * S + m for j in range(Mp)] for m in range(S)], mx.int32)   # [S, Mp]
        tgt = TGT[S]

        def slots(dep):
            pr = gate.spin(dep, L, 2 * Mp * S, phase=1)
            p, s = pr[0::2], pr[1::2]                                 # pads (n, 0)
            match = p[None, None, :] == tgt[:, :, None]               # [S, Mp, pairs]
            sl_late = mx.where(match, s[None, None, :], 0).sum(axis=-1)
            return mx.where(sml >= 0, sml, sl_late)                   # resident at phase A: its own slot

        pads = mx.array([n_m, 0] * (Mp * S), mx.int32)
        if TEST == "ref":                                             # synchronous: phase B at once, the call at layer L
            y = TG(x2, t["base"], t["sideP"], slots(mx.concatenate([dep_m[:1], pads])), None, plan="3" * Mp, scores=wl)
            y = y.reshape(1, S, -1)
            return lambda after: y

        def f(after):                                                 # after layer L+1's attention
            o = mx.isnan(after.reshape(-1)[:1]).astype(mx.int32) * 0
            ms = slots(mx.concatenate([dep_m[:1] + o, pads]))
            return TG(x2, t["base"], t["sideP"], ms, None, plan="3" * Mp, scores=wl).reshape(1, S, -1)
        return f

    def block(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        S = x.shape[-2] if x.ndim == 3 else 0
        if (bm is None or "tier" not in bm or x.ndim != 3 or x.shape[0] != 1 or not MR_MIN <= S <= SMAX
                or bm["L"] >= NL - 1):                                # last trunk layer and the MTP head's layer
            return b12_block(self, x)
        L, t, gate = bm["L"], bm["tier"], bm["gate"]
        ST["calls"] += 1; ST["rows"] += S
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
        rows = (ids | FL).T.reshape(-1)                               # rank-major; ranks A3..B3-1 late-eligible
        dep_m, n_m = gate.post(L, rows, wk=255 | ((K * S) << 8))      # flag mode; LRU-touch every row's ranks < 10
        dep_l, n_l = gate.post(t["side"], (ids[:, :2] | FL[:, :2]).T.reshape(-1), wk=255 | ((2 * S) << 8))
        sm = gate.spin(dep_m, L, n_m).reshape(KC, S).T                # [S, 20] base (MSB-3) slots, -1 = absent / late
        sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T         # [S, 2] LSB sidecar slots of ranks 0-1
        forced = (R < NF)[None]
        pool = (sm >= 0) & ~forced
        sel = (pool & (mx.cumsum(pool.astype(mx.int32), axis=-1) <= K - NF)) | forced
        pos = mx.argsort(mx.where(sel, R, R + 1000), axis=-1)[:, :K]
        used = mx.take_along_axis(sel, pos, axis=-1)
        s10 = mx.take_along_axis(sm, pos, axis=-1)
        s10 = mx.where(used, s10, s10[:, :1])
        w10 = mx.where(used, mx.take_along_axis(p20, pos, axis=-1), 0.0)
        if NORM_MASS:
            w10 = w10 / p20[:, :K].sum(axis=-1, keepdims=True)
        else:
            w10 = w10 / w10.sum(axis=-1, keepdims=True)
        late = sm[:, A3:B3] < 0                                        # [S, Mp] (positions = ranks for r < NF)
        if TEST in ("force", "ref"):
            late = mx.ones(late.shape, mx.bool_)
        s10 = mx.concatenate([s10[:, :A3], mx.where(late, s10[:, :1], s10[:, A3:B3]), s10[:, B3:]], axis=-1)
        w_now = mx.concatenate([w10[:, :A3], mx.where(late, 0.0, w10[:, A3:B3]), w10[:, B3:]], axis=-1)
        B.STASH[L] = late3_mr(x2, mx.where(late, w10[:, A3:B3], 0.0), sm[:, A3:B3], dep_m, n_m, t, gate, L, S)
        y = TG(x2, t["base"], t["sideP"], s10, sl, plan=PLAN, scores=w_now)
        return y.reshape(*x.shape[:-1], y.shape[-1]) + shared

    Q.SparseMoeBlock.__call__ = block
    return ST
