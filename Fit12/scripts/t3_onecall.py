"""Fit12 S1u (Mon 9 Nov 21:3x; /Users/kamen/Fit12/PLAN_12GB_90_fit12.md '12 GB -> 90: WHY <= 4 ROWS'): a COPY of
t3_twopass.py (sha 7d424eb4a4fe, which stays the live module) plus the one-call check S1u. With _U01B on and theta 0, every
check is ONE wide 16-row call: U = every row's ranks 0-1 + the last (bonus) row's ranks 2-9 (<= 2 x 16 + 8 = 40 ids: MAXK 40,
C_m 47). Each row computes its ranks 0-9 that are in U, at REF2-T's tiers (ranks 0-1 at 4 bits, 2-9 at 3 bits) and the
model's weights; ranks not in U get weight 0 on rank 0's slot and are never read. The bonus row is the full top-10. The rows
are committed up to the first miss from this call alone (no exact pass). Checks of <= 2 rows (the cap's end) take the exact
check. LOSSY by design (dropped ranks 2-9 on drafted rows): a speed screen; the map, then the gate, judge quality.
--- the original header ---
Fit12 T3 (Mon 2 Nov 23:3x; Breakthrough ~/Breakthrough/PLAN_12GB_TWOPASS.md section 3 + its 22:19 / 23:03 answers): the
top-2 two-pass check on B12 at 12 GB with DFlash2 L1-20 at 4 bits (D7b) drafting 15 tokens per check (16-row blocks).
FAST track (non-exact by design; EvalSuite's same-scores test judges it; adopting it is Kamen's call).

apply(B), before B.load_b12:
  - t2_twopass's T1 patch (pass 1 = ranks 0-1 only at their mass weights, ranks 2-9 weight 0 and never read), plus a WIDE
    pass-1 branch: a pass-1 call of 5-16 rows posts ranks 0-1 only (2 ids per row; gated.MAXK <= 92 caps a 20-id post at 4
    rows) and runs tier_glu with ranks 2-9 at weight 0 on rank 0's slot, the narrow pass 1's per-row inputs exactly;
  - a tap hook in b12_prefill (after each layer's HC residual write: layers 5 / 15 / 25 / 35 / 45, the mean of the 4 streams).
load_draft(B, model): L1-20 (~/Leap/data/l120_ft.safetensors, ~/Leap/third_party/dflash_model_mlx), d7b_live.apply (4-bit
  layers; fc, selector, head bf16), bound to B12's embedding + lm_head; tap hooks on the trunk's decoder layers.
generate(...): the prompt minus its last token takes the exact AR prefill (B.b12_prefill, taps on); then per check:
  1. propose 15 drafts + each pick's selector cf (Leap's screen: head2s two-stage top-16 + the candidate selector chain)
  2. exact check if the first draft's cf < cfmin (Breakthrough's cf gate) or the history rule says so; else:
  3. pass 1 on all 16 rows; certify (max softmax >= theta) and walk on pass 1's argmax: n = the first uncertified row or miss
  4. row n certified -> commit pass 1's rows 0..n; else keep pass 1's rows 0..n-1 and run pass 2 on rows n..j (j = pass 1's
     first predicted miss at or after n, or the last row), in <= 4-row exact calls (B12's block), walked as an exact check;
     the check ends there (option (a): pass 1's certified tail after j is dropped)
  The exact check = 16 rows verified in consecutive <= 4-row calls, stopping at the first miss (B12's verify is width- and
  full-accept-chunk-invariant: Fit12 WV1 / WV2).
  KV: certified rows keep pass 1's entries; the drafter's context = the committed rows' taps from the pass that produced them.
theta > 1 with cfmin None: every check runs pass 1, rolls back and runs the exact check (gate G1 = the exact loop, theta None).
"""
import contextlib, hashlib, inspect, os, time

import t2_twopass as T2

TAPS = (5, 15, 25, 35, 45)
TAP = {"on": False, "x": [[] for _ in TAPS]}
_WIDE = (
    '    def block(self, x):\n'
    '        bm = getattr(self.switch_mlp, "_bm", None)\n'
    '        S = x.shape[-2] if x.ndim == 3 else 0\n',
    '    def _f12_wide_p1(self, x, bm, S):                               # [Fit12 T3] pass 1 on 5-16 rows: ranks 0-1 only\n'
    '        L, t, gate = bm["L"], bm["tier"], bm["gate"]\n'
    '        if L == bm["first"]:\n'
    '            gate.new_pass()\n'
    '        ae = bm.get("async_every", 0)\n'
    '        if ae and L % ae == ae - 1:\n'
    '            mx.async_eval(x)\n'
    '        if current_attention_phase() == "decode_verify":\n'
    '            shared = attention_gate(self.shared_expert(x), self.shared_expert_gate(x))\n'
    '        else:\n'
    '            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)\n'
    '        x2 = x.reshape(S, x.shape[-1])\n'
    '        gates = mx.softmax(self.gate(x2), axis=-1, precise=True)\n'
    '        inds = mx.argpartition(gates, kth=-KC, axis=-1)[..., -KC:]\n'
    '        sc = mx.take_along_axis(gates, inds, axis=-1)\n'
    '        o = mx.argsort(-sc, axis=-1)\n'
    '        ids = mx.take_along_axis(inds, o, axis=-1).astype(mx.int32)\n'
    '        p20 = mx.take_along_axis(sc, o, axis=-1)\n'
    '        if _F12_QOM0[0]:                                            # [Fit12 S1q] one call: U + resident ranks 2-9 (QOM0)\n'
    '            y = _F12_QOM0_FN(mx, TG, x2, t, gate, L, S, ids, p20, K, W20, NORM_MASS, "".join(T.PLAN))\n'
    '            return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '        if _F12_U01B[0]:                                            # [Fit12 S1u] one call: U = ranks 0-1 + the bonus row\n'
    '            y = _F12_U01B_FN(mx, TG, x2, t, gate, L, S, ids, p20, K, W20, NORM_MASS, "".join(T.PLAN))\n'
    '            return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '        if _F12_K1U[0]:                                             # [Fit12 P1-K1] top-1 + union members (NewIdeas)\n'
    '            y = _F12_K1U_FN(mx, TG, x2, t, gate, L, S, ids, p20, _F12_K1U[0], K, W20, NORM_MASS, m3=_F12_M3[0])\n'
    '            return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '        if _F12_TOPK[0] > 2:                                        # [Fit12 lever 2] top-k pass 1: REF2-T tiers on ranks 0..k-1\n'
    '            k_ = _F12_TOPK[0]                                       # ranks 0-1 at 4 bits (base + side), 2..k-1 at 3 bits (base)\n'
    '            rk = (ids[:, :k_] | W20).T.reshape(-1)                  # rank-major: every row\'s rank 0, 1, .., k-1\n'
    '            r2 = (ids[:, :2] | W20).T.reshape(-1)\n'
    '            dep_m, n_m = gate.post(L, rk, wk=255 | ((k_ * S) << 8))\n'
    '            dep_l, n_l = gate.post(t["side"], r2, wk=255 | ((2 * S) << 8))\n'
    '            sm = gate.spin(dep_m, L, n_m)[:k_ * S].reshape(k_, S).T\n'
    '            sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T\n'
    '            wk_ = p20[:, :k_] / (p20[:, :K].sum(axis=-1, keepdims=True) if NORM_MASS else p20[:, :k_].sum(axis=-1, keepdims=True))\n'
    '            y = TG(x2, t["base"], t["sideP"], sm, sl, plan="".join(T.PLAN[:k_]), scores=wk_)\n'
    '            return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '        rows = (ids[:, :2] | W20).T.reshape(-1)                     # rank-major: every row\'s rank 0, then rank 1\n'
    '        if _F12_M3[0]:                                              # [Fit12 M3] ranks 0-1 at 3 bits: MSB-3 parts only\n'
    '            dep_m, n_m = gate.post(L, rows, wk=255 | ((2 * S) << 8))\n'
    '            sm = gate.spin(dep_m, L, n_m).reshape(2, S).T\n'
    '            w2 = p20[:, :2] / (p20[:, :K].sum(axis=-1, keepdims=True) if NORM_MASS else p20[:, :2].sum(axis=-1, keepdims=True))\n'
    '            y = TG(x2, t["base"], t["sideP"], sm, None, plan="33", scores=w2)\n'
    '            return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '        rb = rows                                                   # [Fit12 P2-PREFETCH] + late-flagged ids\n'
    '        if _F12_PF2[0] and _F12_PF2_ROWS[0] is not None and _F12_PF2_ROWS[0].shape[0] == S:\n'
    '            rb = mx.concatenate([rows, _F12_PF2_FN(mx, ids, _F12_PF2_ROWS[0], ids[:, :2].reshape(-1), _F12_PF2[0], K, W20)])\n'
    '        dep_m, n_m = gate.post(L, rb, wk=255 | ((2 * S) << 8))\n'
    '        dep_l, n_l = gate.post(t["side"], rows, wk=255 | ((2 * S) << 8))\n'
    '        sm = gate.spin(dep_m, L, n_m)[:2 * S].reshape(2, S).T\n'
    '        sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T\n'
    '        s10 = mx.concatenate([sm, mx.repeat(sm[:, :1], K - 2, axis=1)], axis=1)   # ranks 2-9: rank 0\'s slot, weight 0\n'
    '        w10 = mx.concatenate([p20[:, :2], mx.zeros((S, K - 2), p20.dtype)], axis=1)\n'
    '        w10 = w10 / p20[:, :K].sum(axis=-1, keepdims=True) if NORM_MASS else w10 / w10.sum(axis=-1, keepdims=True)\n'
    '        if _F12_K2[0]:                                              # [Fit12 T3] 2 ranks in tier_glu (+0 terms dropped)\n'
    '            y = TG(x2, t["base"], t["sideP"], sm, sl, plan="".join(T.PLAN[:2]), scores=w10[:, :2])\n'
    '        else:\n'
    '            y = TG(x2, t["base"], t["sideP"], s10, sl, plan="".join(T.PLAN), scores=w10)\n'
    '        return y.reshape(*x.shape[:-1], y.shape[-1]) + shared\n'
    '\n'
    '    def block(self, x):\n'
    '        bm = getattr(self.switch_mlp, "_bm", None)\n'
    '        S = x.shape[-2] if x.ndim == 3 else 0\n'
    '        if _F12_T1[0] and bm is not None and "tier" in bm and x.ndim == 3 and x.shape[0] == 1 and SMAX < S <= 16:\n'
    '            return _f12_wide_p1(self, x, bm, S)\n')
_PFTAP = ('                mx.eval(*hs)\n'
          '                del mixed, hyper, inject, bos\n',
          '                mx.eval(*hs)\n'
          '                if _F12_PFTAP[0] is not None:                  # [Fit12 T3] the drafter\'s taps from the prefill\n'
          '                    _F12_PFTAP[0](L, hs)\n'
          '                del mixed, hyper, inject, bos\n')
_PFTAP_FN = [None]
_TOPK = [0]      # (Thu 5 Nov 20:1x, 12 GB plan lever 2, Breakthrough 19:5x) k > 2: the WIDE pass 1 runs each row's ranks
                 # 0..k-1 at REF2-T's tiers (0-1 at 4 bits, 2..k-1 at 3 bits) at their top-10 mass weights; 0 = off
_U01B = [False]  # (Mon 9 Nov 21:3x, S1u) the wide call = the one-call check's union form (u01b_moe); False = off
_QOM0 = [False]  # (Tue 10 Nov 10:2x, S1q) the wide call = QOM0 at cf 0 with the bonus row exact (qom0_moe); needs BIGMOE_MAXK=92
QOM0_ACC = []    # per wide QOM0 call: (lazy [residents added, residents dropped for room], rows)
_QK = {}         # the probe kernel, built on first use


def _SMAXQ():
    """bt_b12's SMAX = gated.MAXK // KC (KC 20): the rows a narrow (frugal-path) call takes; 4 at MAXK 92"""
    from bigmoe_v3f.gated import MAXK
    return MAXK // 20
_K2 = [os.environ.get("F12_P1K2", "0") == "1"]   # pass 1's tier_glu on ranks 0-1 only (plan '44'): the 8 weight-0 pairs dropped
S_MOE = [int(os.environ.get("F12_SMOE", "4096"))]   # (Sun 8 Nov, 12 GB prefill) b12_prefill's MoE chunk: experts read once per
                 # layer per S_MOE prompt rows (a multiple of 512); 8192 = one MoE call for prompts up to 8K (BigMoE's PFX form)
W1 = [os.environ.get("F12_W1", "4")]   # (Sun 8 Nov, T2 CHAT WIDTH) the exact check's FIRST call width (later calls 4 rows):
                 # "4" = today's; "rand" = 1-4 per check from random.Random(F12_W1SEED) (the per-width cost fit); "cf:C:W[:C2:W2]"
                 # = W rows when the first draft's cf < C (else W2 when < C2, ...), else 4. Rows 0..k verified in any chunking give
                 # the same bits for calls of >= 2 rows (WV1 / WV2); a 1-row call takes B12_LATE3's late rule unless W1AW
WL = [int(os.environ.get("F12_WL", "4"))]   # (Sun 8 Nov, C2c: Strata's chat width) the exact check's LATER call width (4 = today's)
W1TAIL = [os.environ.get("F12_W1TAIL", "0") == "1"]   # (Sun 8 Nov, T2 stage 1b) no 1-row last chunk in an exact check
W1AW = [os.environ.get("F12_W1AW", "1") == "1"]   # (T2) a 1-row exact call runs with B.AW[0] (ALLWAIT: the verify rows' flags)
GQ = dict(calls=0, sorted=0, maxm=0, exposed=0, padded=0)   # (Sun 8 Nov, MLX 0.32.2 sorted-gather bug; Leap a521e91) the
                 # gather_qmm audit when installed: every call, the sorted ones, their largest row count, the EXPOSED ones
                 # (rows > 32,767 and rows % 64 != 0: affine_gather_qmm_rhs_nax's short overflow leaves rows unwritten), padded


def gq_install(pad=False):
    """(Sun 8 Nov) wrap mx.gather_qmm in this process: count every call (GQ) and, with pad, take Leap's exact fix for an
    exposed sorted call (rows padded to a multiple of 64 with copies of the last row and its expert; output sliced back)"""
    import mlx.core as mx
    if getattr(mx.gather_qmm, "_f12", False):
        return
    orig = mx.gather_qmm

    def gq(x, w, scales, biases=None, lhs_indices=None, rhs_indices=None, *a, **kw):
        GQ["calls"] += 1
        if kw.get("sorted_indices", False) and rhs_indices is not None:
            M = int(rhs_indices.size)
            GQ["sorted"] += 1
            GQ["maxm"] = max(GQ["maxm"], M)
            if M > 32767 and M % 64:
                GQ["exposed"] += 1
                if pad:
                    p_ = (-M) % 64
                    x = mx.concatenate([x, mx.repeat(x[-1:], p_, axis=0)], axis=0)
                    rhs_indices = mx.concatenate([rhs_indices, mx.repeat(rhs_indices[-1:], p_, axis=0)], axis=0)
                    GQ["padded"] += 1
                    return orig(x, w, scales, biases, lhs_indices, rhs_indices, *a, **kw)[:M]
        return orig(x, w, scales, biases, lhs_indices, rhs_indices, *a, **kw)
    gq._f12 = True
    mx.gather_qmm = gq


CALLS = [None]   # (T2) a list: one row per exact call (check, call index, width, rows kept in the call, then _mk() at start, at end)
MEMLOG = [None]  # (Tue 10 Nov 11:3x, S1q memory, F12_MEMLOG=1) per check and phase: (check, phase, MLX active, MLX peak since
                 # the check began, MLX cache) in bytes; None = off
TS1 = [None]     # (Mon 9 Nov 23:0x, S1u profile, F12_GSTAT=1) one row per wide pass-1 call: (check, -1, rows, kept) + uptime
                 # stamps t0 (start), t1 (after forward: graph built), t2 (after eval), t3 (after the commit) + the loader's
                 # (proc_s, io_s, cpu_wait_s, misses, stalled, uses) at t0, t1, t2 + the main thread's CPU at t0, t1, t2 + the
                 # process's disk bytes at t0, t2. Measures the call's loader-busy vs loader-idle wall time ([K], not the fit).
TS = [None]      # (T2 PROF, F12_GSTAT=1) a list: one row per exact call: (check, call index, width, kept) + uptimestamps at start,
                 # after forward() (graph built), after mx.eval, after tolist + compare, after commit, at the end (after the
                 # lsha hash) + the loader's (proc_s, io_s, cpu_wait_s, misses, stalled, uses) at start, after forward, after eval
PHLOG = [None]   # (Tue 3 Nov, T3P) a list: one row per timed phase (name, check, then at start and at end: CLOCK_UPTIME_RAW s,
                 # the gate's pass seq, disk bytes read by this process, loader misses); None = off (no extra calls)
CAL = [None]     # (Thu 5 Nov 20:0x, 12 GB plan lever 1, certificate calibration) a list: one dict per check (path, the draft d,
                 # its per-row cf, pass 1's per-row max softmax p1 / top-2 margin m1 / argmax g1 when pass 1 ran, the
                 # committed tokens new); None = off: no extra op, the code path is unchanged
_K1U = [0]       # (Tue 3 Nov 10:3x, P1-K1, NewIdeas 10:24 / ~/NewIdeas/QUEUE.md 'P1-OEA') KU > 0: a WIDE pass-1 call (3-16
                 # rows) reads only U = the rows' rank-0 experts, and each row's MoE = its ranks 0-9 that are in U (the map's
                 # oea_k1 mask, scripts/dm_oea_masks.py), the first KU of them in rank order, at 4 bits (U's LSB parts are
                 # read anyway; the map runs 4-bit experts) and the model's weights (B12_NORM=mass: no renorm). 0 = off.
                 # Narrow pass-1 calls (<= SMAX rows, the last check of a capped run) stay top-2.
K1U_ACC = []     # per wide K1U call: (lazy [kept pairs, rows over KU], rows); generate() evaluates them with pass 1
_M3 = [False]    # (Tue 3 Nov 10:4x, Breakthrough 03:10 item 4 / M-18) True: a WIDE pass-1 call reads and runs its experts at 3
                 # bits (MSB-3 parts only, no LSB sidecar post): top-2 at plan '33', or P1-K1 at '3' x KU. INFO timing arms.


_PF2 = [0]         # (Tue 3 Nov 15:0x, NewIdeas' P2-PREFETCH, exact) B > 0: each WIDE pass-1 call appends to its base-pool
PF2_ROWS = [None]  # post up to B extra ids flagged late (bit 21: read after the GPU's release, never waited): the uncertain
PF2_ACC = []       # rows' (PF2_ROWS: draft cf < 0.99) ranks 1-9 not already posted, row then rank order, deduped.
PF2_CF = 0.99
                   # Pass 1 ignores their slots; pass 2 then finds them resident. Needs BIGMOE_DEFER >= B (T3: 8).


def pf2_extra(mx, ids, rows_u, posted, B, K, W20):
    """the <= B prefetch ids (flag bit 21) to append to a wide pass-1 base post, padded with an unflagged posted id
    (no read, no LRU touch past the call's nk): uncertain rows' ranks 1..K-1 not in `posted`, row then rank order"""
    S = ids.shape[0]
    c = ids[:, 1:K]                                                  # [S, K-1] ranks 1..K-1, rank order per row
    v = rows_u[:, None] & ~(c[:, :, None] == posted[None, None, :]).any(axis=-1)
    f, v = c.reshape(-1), v.reshape(-1)                              # row-major: row then rank order
    N = f.shape[0]
    ar = mx.arange(N)
    dup = ((f[:, None] == f[None, :]) & (ar[:, None] < ar[None, :]) & v[:, None]).any(axis=0)   # an earlier valid copy
    v = v & ~dup
    pos = mx.argsort(mx.where(v, ar, ar + N))[:B]
    sel, ok = mx.take(f, pos), mx.take(v, pos)
    PF2_ACC.append(ok.astype(mx.int32).sum())
    return mx.where(ok, sel | (2 * W20), posted[:1] & (W20 - 1)).astype(mx.int32)


def k1u_moe(mx, TG, x2, t, gate, L, S, ids, p20, KU, K, W20, norm_mass, dbg=None, m3=False):
    """P1-K1's routed MoE for one wide pass-1 call: post U (each row's rank 0, both record parts, waited), then
    tier_glu over each row's ranks in U, compacted to KU columns (padding = rank 0's slots at weight 0).
    m3: U's MSB-3 parts only (no sidecar post), the KU columns at 3 bits."""
    id0 = ids[:, 0]
    r0 = id0 | W20
    pf = _PF2[0] and PF2_ROWS[0] is not None and PF2_ROWS[0].shape[0] == S
    rb = mx.concatenate([r0, pf2_extra(mx, ids, PF2_ROWS[0], id0, _PF2[0], K, W20)]) if pf else r0
    dep_m, n_m = gate.post(L, rb, wk=255 | (S << 8))                 # flag mode; LRU-touch the S U ids (+ late prefetch)
    sm0 = gate.spin(dep_m, L, n_m)[:S].reshape(S) if m3 else None
    if not m3:
        dep_l, n_l = gate.post(t["side"], r0, wk=255 | (S << 8))
        sm0 = gate.spin(dep_m, L, n_m)[:S].reshape(S)
        sl0 = gate.spin(dep_l, t["side"], n_l).reshape(S)
    eq = ids[:, :K, None] == id0[None, None, :]                      # [S, 10, S]: row i's rank r = row j's rank 0
    inU = eq.any(axis=-1)                                            # rank 0 always in
    jm = mx.argmax(eq.astype(mx.int32), axis=-1)                     # the first such row j
    RK = mx.arange(K)
    pos = mx.argsort(mx.where(inU, RK, RK + 100), axis=-1)[:, :KU]   # the kept ranks first, in rank order
    used = mx.take_along_axis(inU, pos, axis=-1)
    jj = mx.take_along_axis(jm, pos, axis=-1)
    s_m = mx.take(sm0, jj)
    s_m = mx.where(used, s_m, s_m[:, :1])
    s_l = None
    if not m3:
        s_l = mx.take(sl0, jj)
        s_l = mx.where(used, s_l, s_l[:, :1])
    w = mx.where(used, mx.take_along_axis(p20, pos, axis=-1), 0.0)
    w = w / p20[:, :K].sum(axis=-1, keepdims=True) if norm_mass else w / w.sum(axis=-1, keepdims=True)
    K1U_ACC.append((mx.stack([used.astype(mx.int32).sum(), (inU.astype(mx.int32).sum(axis=-1) > KU).astype(mx.int32).sum()]), S))
    if dbg is not None:
        dbg.update(inU=inU, pos=pos, used=used, s_m=s_m, w=w, **({} if m3 else {"s_l": s_l}))
    return TG(x2, t["base"], t["sideP"], s_m, s_l, plan=("3" if m3 else "4") * KU, scores=w)


def u01b_moe(mx, TG, x2, t, gate, L, S, ids, p20, K, W20, norm_mass, plan):
    """(Mon 9 Nov 21:3x) S1u's routed MoE for one wide call of S rows: post U = every row's ranks 0-1 (base + sidecar parts)
    + the last (bonus) row's ranks 2-9 (base), all waited; then tier_glu over each row's ranks 0-9 at REF2-T's plan, a rank
    not in U at weight 0 on rank 0's slot (never read). Duplicate ids in a post are allowed (as the top-k pass 1's posts)."""
    r01 = ids[:, :2].T.reshape(-1)                                   # [2S] rank-major: every row's rank 0, then rank 1
    U = mx.concatenate([r01, ids[S - 1, 2:K]])                       # [2S + K - 2] the posted base ids (<= 40 at S 16)
    nU = 2 * S + K - 2
    dep_m, n_m = gate.post(L, U | W20, wk=255 | (nU << 8))
    dep_l, n_l = gate.post(t["side"], r01 | W20, wk=255 | ((2 * S) << 8))
    su = gate.spin(dep_m, L, n_m)[:nU]
    sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T             # [S, 2] the sidecar slots of every row's ranks 0-1
    eq = ids[:, :K, None] == U[None, None, :]                        # [S, K, nU]: row i's rank r is posted at position u
    inU = eq.any(axis=-1)                                            # ranks 0-1 always; the bonus row's ranks 0-9 always
    s10 = mx.take(su, mx.argmax(eq.astype(mx.int32), axis=-1))       # [S, K] the base slot of each rank in U
    s10 = mx.where(inU, s10, s10[:, :1])
    w = mx.where(inU, p20[:, :K], 0.0)
    w = w / p20[:, :K].sum(axis=-1, keepdims=True) if norm_mass else w / w.sum(axis=-1, keepdims=True)
    K1U_ACC.append((mx.stack([inU.astype(mx.int32).sum(), mx.array(0, mx.int32)]), S))   # kept pairs (generate counts them)
    return TG(x2, t["base"], t["sideP"], s10, sl, plan=plan, scores=w)


_QSRC = r"""
uint t = thread_position_in_grid.x;
threadgroup int res[128];
int off = meta[0], seq = meta[1], nc = meta[2], nu = meta[3], cap = meta[4], cm = meta[5];
const volatile device int* tb = (const volatile device int*)tables;
int r = 0;
if ((int)t < nc) {                         // the residency table published for (pass, layer), read as post_ids reads it
  int e = cand[t] & 0xFFFFF, li = e / 31;
  int tag = tb[off + li * 32], w = tb[off + li * 32 + 1 + (e - li * 31)];
  bool fresh = tag == seq && ((((uint)w) & 0xFFC00000u) == ((((uint)seq) & 0x3FFu) << 22));
  r = (fresh && ((((uint)w) & 0x3FFFFFu) != 0x3FFFFFu)) ? 1 : 0;
}
res[t] = r;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (t == 0) {                              // unique residents not in U, in row then rank order, at most cap
  uint bm[16];
  for (int i = 0; i < 16; i++) bm[i] = 0u;
  for (int i = 0; i < nu; i++) { int e = U[i] & 0xFFFFF; if (e < 512) bm[e >> 5] |= (1u << (e & 31)); }
  int nud = 0;                             // the post's distinct experts must fit the pool (the loader aborts above C_m)
  for (int i = 0; i < 16; i++) nud += popcount(bm[i]);
  int room = min(cap, max(0, cm - nud));   // cap stays the output's size: every entry past k is padded
  int k = 0, over = 0;
  for (int i = 0; i < nc; i++) {
    if (!res[i]) continue;
    int e = cand[i] & 0xFFFFF;
    if (e >= 512 || (bm[e >> 5] & (1u << (e & 31)))) continue;
    bm[e >> 5] |= (1u << (e & 31));        // seen: a repeat is neither added nor counted again
    if (k < room) { extra[k] = e; k++; } else { over++; }
  }
  int pad = U[0] & 0xFFFFF;
  for (int i = k; i < cap; i++) extra[i] = pad;
  cnt[0] = k; cnt[1] = over;
}
"""
QMAXK = 92       # S1q's base post: U (<= 40) + residents (<= 52); the routing row's limit (gated.py)


def qom0_moe(mx, TG, x2, t, gate, L, S, ids, p20, K, W20, norm_mass, plan):
    """(Tue 10 Nov 10:2x) S1q: QOM0 at cf 0 with the bonus row exact, for one wide call of S rows. U = every row's ranks 0-1
    + the bonus row's ranks 2-9, as S1u. A probe kernel reads the residency table the loader published for this (pass,
    layer) (the one post_ids reads) and adds the drafted rows' ranks 2-9 that are resident and not in U (unique, at most
    QMAXK - |U|) to the SAME post: posted, they are in the loader's 'need' set, so make_room cannot evict them during the
    call, and as hits they cost no reads. Each row computes its ranks 0-9 that are posted, at REF2-T's tiers and the model's
    weights; the others get weight 0 on rank 0's slot (never read). Needs BIGMOE_MAXK=92."""
    import mlx.core as mx_
    if "k" not in _QK:
        _QK["k"] = mx_.fast.metal_kernel(name="f12_qom0_probe", input_names=["tables", "cand", "U", "meta"],
                                         output_names=["extra", "cnt"], source=_QSRC)
    r01 = ids[:, :2].T.reshape(-1)                                   # [2S] every row's rank 0, then rank 1
    U = mx.concatenate([r01, ids[S - 1, 2:K]])                       # [2S + K - 2]
    nU = 2 * S + K - 2
    cap = QMAXK - nU
    cand = ids[:S - 1, 2:K].reshape(-1)                              # the drafted rows' ranks 2-9
    off = ((gate.seq * gate.nl + L) % gate.ntab) * 17 * 32           # tab_offset(L, seq) (bigmoe_ext.cpp; TLINES 17)
    cm = int(os.environ.get("F12_CMCL", "47,32").split(",")[0])        # the base pool's slots per layer (C_m)
    meta = mx.array([off, gate.seq, int(cand.size), nU, cap, cm], mx.int32)
    extra, cnt = _QK["k"](inputs=[gate.tables, cand, U, meta], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[(cap,), (2,)], output_dtypes=[mx.int32, mx.int32])
    P = mx.concatenate([U, extra])                                   # [QMAXK] the base post
    dep_m, n_m = gate.post(L, P | W20, wk=255 | (QMAXK << 8))
    dep_l, n_l = gate.post(t["side"], r01 | W20, wk=255 | ((2 * S) << 8))
    su = gate.spin(dep_m, L, n_m)[:QMAXK]
    sl = gate.spin(dep_l, t["side"], n_l).reshape(2, S).T
    eq = ids[:, :K, None] == P[None, None, :]                        # [S, K, QMAXK]
    inP = eq.any(axis=-1)
    s10 = mx.take(su, mx.argmax(eq.astype(mx.int32), axis=-1))
    s10 = mx.where(inP, s10, s10[:, :1])
    w = mx.where(inP, p20[:, :K], 0.0)
    w = w / p20[:, :K].sum(axis=-1, keepdims=True) if norm_mass else w / w.sum(axis=-1, keepdims=True)
    K1U_ACC.append((mx.stack([inP.astype(mx.int32).sum(), mx.array(0, mx.int32)]), S))   # kept pairs (generate counts them)
    # self-check of the probe: post_ids read the same table for the posted ids (dep = [ok, slot or -1 per id]); every resident
    # the probe added must show a slot there too, else the probe read the wrong (pass, layer) region or entry
    dx = dep_m[1 + nU:1 + nU + cap]
    bad = ((dx < 0) & (mx.arange(cap) < cnt[0])).astype(mx.int32).sum()
    QOM0_ACC.append(mx.concatenate([cnt, bad.reshape(1)]))
    return TG(x2, t["base"], t["sideP"], s10, sl, plan=plan, scores=w)


def _diskread():
    """bytes this process has read from disk (rusage_info_v4 ri_diskio_bytesread; the loader's F_NOCACHE reads included)"""
    import ctypes

    class _RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]
    ru = _RU()
    return ru.u16 if ctypes.CDLL(None).proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else -1
_NARROW_K2 = ('        y = TG(x2, t["base"], t["sideP"], s10, l2, plan="".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel\n',
              '        if _p1 and _F12_K2[0]:                                        # [Fit12 T3] pass 1: 2 ranks in tier_glu\n'
              '            y = TG(x2, t["base"], t["sideP"], s10[:, :2], l2[:, :2], plan="".join(T.PLAN[:2]), scores=w_now[:, :2])\n'
              '        else:\n'
              '            y = TG(x2, t["base"], t["sideP"], s10, l2, plan="".join(T.PLAN), scores=w_now)   # [S, D] combined in-kernel\n')


def apply(B):
    """T1 + the wide pass-1 branch in bt_b12._install, the tap hook in bt_b12.b12_prefill; before B.load_b12"""
    if B.__dict__.get("_F12_T3"):
        return True
    assert "_F12_P1" not in B.__dict__ and "_F12_T1" not in B.__dict__, "another two-pass patch is applied here"
    src = inspect.getsource(B._install)
    for old, new in T2._REPL + [_WIDE, _NARROW_K2]:
        assert src.count(old) == 1, ("t3_twopass: bt_b12 changed", old[:80])
        src = src.replace(old, new)
    B.__dict__["_F12_T1"] = T2._P1
    B.__dict__["_F12_K2"] = _K2
    B.__dict__["_F12_K1U"] = _K1U
    B.__dict__["_F12_K1U_FN"] = k1u_moe
    B.__dict__["_F12_M3"] = _M3
    B.__dict__["_F12_TOPK"] = _TOPK
    B.__dict__["_F12_U01B"] = _U01B                    # (S1u)
    B.__dict__["_F12_U01B_FN"] = u01b_moe
    B.__dict__["_F12_QOM0"] = _QOM0                    # (S1q)
    B.__dict__["_F12_QOM0_FN"] = qom0_moe
    B.__dict__["_F12_PF2"] = _PF2
    B.__dict__["_F12_PF2_ROWS"] = PF2_ROWS
    B.__dict__["_F12_PF2_FN"] = pf2_extra
    exec(compile(src, B.__file__ + " [t3_twopass]", "exec"), B.__dict__)
    src = inspect.getsource(B.b12_prefill)
    assert src.count(_PFTAP[0]) == 1, "t3_twopass: b12_prefill changed"
    B.__dict__["_F12_PFTAP"] = _PFTAP_FN
    exec(compile(src.replace(*_PFTAP), B.__file__ + " [t3_twopass prefill taps]", "exec"), B.__dict__)
    B.__dict__["_F12_T3"] = True
    return True


def _mean4(h):
    import mlx.core as mx
    Bn, L, W = h.shape
    return h.reshape(Bn, L, 4, W // 4).astype(mx.float32).mean(axis=2)[0].astype(mx.float16)


def load_draft(B, model):
    """DFlash2 L1-20 at 4 bits (D7b), bound to B12's model; tap hooks on the trunk's decoder layers 5 / 15 / 25 / 35 / 45"""
    import sys
    import mlx.core as mx
    for p in ("~/Leap/third_party", "~/Leap/scripts"):
        p = os.path.expanduser(p)
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")       # no downloads: the config must already be local
    import dflash_model_mlx as DF
    import d7b_live
    import mtplx.models.qwen4_exp as Q
    draft = DF.load_draft("incoai/Qwen3.8-Flash-Next-DFlash2")
    draft.load_weights(os.path.expanduser(os.environ.get("TL_DRAFT", "~/Leap/data/l120_ft.safetensors")), strict=False)
    d7b_live.apply(draft)
    draft.bind(model)
    mx.eval(draft.parameters())
    tm = model.language_model.model
    of = {id(tm.layers[L]): i for i, L in enumerate(TAPS)}
    call0 = Q.DecoderLayer.__call__                    # B12's layer_call after load_b12

    def _layer(self, *a, **kw):
        out = call0(self, *a, **kw)
        i = of.get(id(self))
        if TAP["on"] and i is not None:
            TAP["x"][i].append(_mean4(out if isinstance(out, mx.array) else out[0]))
        return out
    Q.DecoderLayer.__call__ = _layer
    return draft, DF


def tap_reset():
    TAP["x"] = [[] for _ in TAPS]


def taps():
    """[rows, 5 * 2560] fp16 of the forwards since the reset"""
    import mlx.core as mx
    f = [mx.concatenate(x) if len(x) > 1 else x[0] for x in TAP["x"]]
    return mx.stack(f, axis=1).reshape(f[0].shape[0], -1)


def _proposer(draft):
    import mlx.core as mx
    R, K = 16, 16

    def head2s(hid):
        lg = draft.compute_logits(hid)
        Bk, s = 128, lg.shape
        blk = lg.reshape(*s[:-1], s[-1] // Bk, Bk)
        tb = mx.argpartition(blk.max(axis=-1), -K, axis=-1)[..., -K:]
        sel = mx.take_along_axis(blk, mx.broadcast_to(tb[..., None], (*s[:-1], K, Bk)), axis=-2).reshape(*s[:-1], K * Bk)
        j = mx.argpartition(sel, -K, axis=-1)[..., -K:]
        c = mx.take_along_axis(tb, j // Bk, axis=-1) * Bk + j % Bk
        return c, mx.take_along_axis(lg, c, axis=-1)

    def chain(blk, unary, cand, hp):
        sel = draft.candidate_selector
        pred, path, conf = blk[:, 0], [], []
        for p in range(hp.shape[1]):
            sc = unary[:, p] + mx.sum(sel.predecessor_codebook(pred)[:, None] * hp[:, p, None]
                                      * sel.successor_codebook(cand[:, p]), axis=-1)
            s_ = mx.argmax(sc, axis=-1)
            pred = mx.take_along_axis(cand[:, p], s_[:, None], axis=-1)[:, 0]
            path.append(pred)
            conf.append(mx.take_along_axis(mx.softmax(sc.astype(mx.float32), axis=-1), s_[:, None], axis=-1)[:, 0])
        return mx.stack(path, axis=1), mx.stack(conf, axis=1)

    def propose(blk, ctx, dcache):
        hid = draft.hidden_states(blk, ctx, dcache, 1)
        cand, unary = head2s(hid)
        return chain(blk, unary, cand, draft.candidate_selector.hidden_projection(hid))
    return propose, R


def generate(B, model, draft, DF, prompt_ids, max_tokens, theta, cfmin=None, skip=None, stop_ids=(248044, 248046),
             log=None, stats=None, p2_all=False):
    """two-pass 16-row checks (theta float) or the exact chunked check (theta None); see the module docstring.
    p2_all: pass 2 covers every row from n on (the exact check of toks[n:]), the G1 gate's form (theta 1.01 = theta None)"""
    import numpy as np
    import mlx.core as mx
    from mtplx.cache_state import rollback_after_verify, snapshot_untrimmable_cache_lazy
    from mtplx.graphbank import ensure_eager_window_capacity
    from bigmoe_v3f.fuse import reset_generation_state
    propose, R = _proposer(draft)
    MASK, SW = int(draft.config.mask_token_id), int(draft.config.sliding_window)
    st = stats if stats is not None else {}
    st.update(checks=0, pass2=0, exact=0, rows1=0, rows2=0, calls2=0, refused=0, skipped=0, cfgate=0,
              t_draft=0.0, t_p1=0.0, t_p2=0.0, t_commit=0.0, miss1=0, miss2=0,   # timers: at the loop's own eval points
              k1u_kept=0, k1u_over=0, k1u_rl=0,                # P1-K1: kept pairs, row-layers over KU, row-layers
              pf2_ids=0, pf2_calls=0,                          # P2-PREFETCH: late ids posted, wide pass-1 calls
              w1_calls={})                                     # (T2) the exact checks' first calls, by width
    import random
    _rng = random.Random(int(os.environ.get("F12_W1SEED", "1")))   # (T2) "rand": the same width sequence per prompt

    def w1_pick(cf0):                                  # (T2) the exact check's first call width (W1's policy)
        p_ = W1[0]
        if p_ == "rand":
            return _rng.randint(1, 4)
        if p_.startswith("rand:"):                     # (stage 1b) a random width from a set, e.g. "rand:2,3,4"
            return _rng.choice([int(v_) for v_ in p_[5:].split(",")])
        if p_.startswith("cf:"):
            v_ = p_.split(":")[1:]
            for i_ in range(0, len(v_) - 1, 2):
                if cf0 is not None and cf0 < float(v_[i_]):
                    return int(v_[i_ + 1])
            return 4
        return int(p_)
    K1U_ACC.clear()
    PF2_ACC.clear()
    from bigmoe_v3f.gated import ext as _ext
    _miss = lambda: _ext.stats()["misses"]
    _PL = PHLOG[0]
    _gate = None if _PL is None else next((m_._bm["gate"] for m_ in model.modules()
                                           if isinstance(getattr(m_, "_bm", None), dict) and "gate" in m_._bm), None)

    def _mk():                                         # (T3P) a phase mark: uptime, gate seq, disk bytes, misses
        return None if _PL is None else (time.clock_gettime(time.CLOCK_UPTIME_RAW), getattr(_gate, "seq", -1),
                                         _diskread(), _miss())

    def _ph(name, k0):
        if _PL is not None:
            _PL.append((name, st["checks"]) + k0 + _mk())

    _now = lambda: time.clock_gettime(time.CLOCK_UPTIME_RAW)

    def _gs():                                         # (T2 PROF) the loader thread's own clocks and counts
        s_ = _ext.stats()
        return (s_["proc_s"], s_["io_s"], s_["cpu_wait_s"], s_["misses"], s_["stalled"], s_["uses"])
    reset_generation_state(model)
    B.STASH.clear(); B.PENDH.clear()
    ids = [int(t) for t in prompt_ids]
    P = len(ids)
    cache = model.make_cache()
    pf = []

    def pftap(L, hs):                                  # the drafter's taps, evaluated at once (no lazy hold on hs)
        if L in TAPS:
            t_ = [_mean4(h) for h in hs]
            mx.eval(*t_)
            pf.append((L, t_))
    _PFTAP_FN[0] = pftap
    t0 = time.perf_counter()
    pfd0 = _diskread()                                 # (Sun 8 Nov) the prefill's own SSD bytes (rusage diskio)
    gq0 = dict(GQ); GQ["maxm"] = 0; npl = len(getattr(B, "PFR_LOG", []))   # (Sun 8 Nov) the prefill's own audit + PFR chunk
    try:
        B.b12_prefill(model, mx.array([ids[:-1]]), cache, 512, S_MOE[0])
    finally:
        _PFTAP_FN[0] = None
    per = {L: [] for L in TAPS}
    for L, chunks in pf:
        per[L] += chunks
    T = mx.stack([mx.concatenate(per[L]) for L in TAPS], axis=1).reshape(P - 1, -1)   # [P - 1, 5 * 2560]
    del pf, per
    mx.eval(T)
    t1 = time.perf_counter()
    st["pf_mb"] = (_diskread() - pfd0) / 1e6
    st["gq"] = {k_: (GQ[k_] if k_ == "maxm" else GQ[k_] - gq0[k_]) for k_ in GQ}
    st["pfr"] = list(getattr(B, "PFR_LOG", [])[npl:])                    # (P, s_moe) when PFR took this prefill
    dcache = draft.make_cache()
    start = max(0, (P - 1) - (SW - 1))
    for c in dcache:
        c.offset = start
    ctx, a = T[start:], P - 1
    del T
    aw0 = B.AW[0]
    if hasattr(B, "_fs") and B._fs("AW"):
        B.AW[0] = True
    h, y, out = hashlib.sha1(), ids[-1], []
    streak, skip_left = 0, 0

    def forward(toks_, p1):
        S_ = len(toks_)
        snap_ = None
        if S_ > 1:
            snap_ = snapshot_untrimmable_cache_lazy(cache)
            ensure_eager_window_capacity(cache, S_)
        else:
            from mtplx.cache_state import snapshot_untrimmable_cache
            snap_ = snapshot_untrimmable_cache(cache)
        scope = model.verify_capture_scope() if S_ > 1 else contextlib.nullcontext()
        T2._P1[0] = p1
        tap_reset(); TAP["on"] = True
        try:
            with scope:
                lg_ = model(mx.array([toks_]), cache=cache)
        finally:
            T2._P1[0] = False
            TAP["on"] = False
        lg_ = (lg_[0] if isinstance(lg_, tuple) else lg_)[0].astype(mx.float32)
        return lg_, taps(), snap_

    def commit(toks_, snap_, keep, p1):
        S_ = len(toks_)
        if S_ == 1:
            return None
        tp_ = None
        ok = model.commit_verified_window(cache, snap_.states, keep_tokens=keep, verified_tokens=S_)
        if not ok:
            st["refused"] += 1
            rollback_after_verify(cache, snap_, verified_tokens=S_)
            T2._P1[0] = p1
            tap_reset(); TAP["on"] = True
            try:
                model(mx.array([toks_[:keep]]), cache=cache)
            finally:
                T2._P1[0] = False
                TAP["on"] = False
            tp_ = taps()
            mx.eval(tp_)
        model.clear_verify_capture(cache)
        return tp_

    def exact(toks_, d_, w1=4):
        """the exact check of toks_ (= [x] + d_) in <= 4-row calls (the first one w1 rows), stopping at the first miss:
        (new, taps of kept rows)"""
        new_, tps_, pos = [], [], 0
        _t, _m = time.perf_counter(), _miss()
        _k = _mk()
        try:
            return _exact_loop(toks_, d_, new_, tps_, pos, w1)
        finally:
            st["t_p2"] += time.perf_counter() - _t
            st["miss2"] += _miss() - _m
            _ph("exact", _k)

    def _call(ci, w, kept, kc, ts=None):               # (T2) one exact call's row
        if CALLS[0] is not None and kc is not None:
            CALLS[0].append((st["checks"], ci, w, kept) + kc + _mk())
        if TS[0] is not None and ts is not None:       # (T2 PROF) stamps [t0, t1, t2, t3, t4] + gate stats [g0, g1, g2]
            TS[0].append((st["checks"], ci, w, kept) + tuple(ts[:5]) + (_now(),) + tuple(ts[5]) + tuple(ts[6]) + tuple(ts[7])
                         + tuple(ts[8]))                # (chw16) the main thread's CPU time at t0, t1, t2

    def _exact_loop(toks_, d_, new_, tps_, pos, w1=4):
        ci, w_ = 0, w1
        while True:
            wc = w_                                            # the chosen width (W1's pick for the first call, else 4)
            if W1TAIL[0]:                                      # (stage 1b) never leave a 1-row last chunk (decode-width
                w_ = min(w_, len(toks_) - pos)                 # arithmetic): 4 + 1 -> 3 + 2, 2 + 1 -> 3
                if len(toks_) - pos - w_ == 1:
                    w_ = w_ - 1 if w_ >= 3 else w_ + 1
            ch = toks_[pos:pos + w_]
            kc = _mk()
            ts_ = None if TS[0] is None else [_now()]        # (T2 PROF) t0
            cp_ = None if ts_ is None else [time.thread_time()]   # (chw16) the main thread's CPU at t0
            gs_ = None if ts_ is None else [_gs()]           # (T2 PROF) g0
            aw = W1AW[0] and len(ch) == 1 and (wc == 1 or W1TAIL[0]) and not B.AW[0]   # a chosen 1-row call; with the
            # tail rule also the cap's last token (no drafts), whose late-rule bits would otherwise depend on the pools
            if aw:
                B.AW[0] = True
            try:
                lg_, tp_, sn_ = forward(ch, False)
                if ts_ is not None:
                    ts_.append(_now()); gs_.append(_gs())    # t1, g1: the graph is built (any syncs inside forward included)
                    cp_.append(time.thread_time())
                g_ = mx.argmax(lg_, axis=-1)
                mx.eval(g_, tp_)
                if ts_ is not None:
                    ts_.append(_now()); gs_.append(_gs())    # t2, g2: the GPU + loader work is done
                    cp_.append(time.thread_time())
            finally:
                if aw:
                    B.AW[0] = False
            st["w1_calls"][len(ch)] = st["w1_calls"].get(len(ch), 0) + (ci == 0)
            g_ = g_.tolist()
            st["calls2"] += 1
            st["rows2"] += len(ch)
            m_ = 0
            while m_ < len(ch) and pos + m_ < len(d_) and d_[pos + m_] == g_[m_]:
                m_ += 1
            if ts_ is not None:
                ts_.append(_now())                           # t3: tolist + compare done
            if m_ == len(ch) and pos + len(ch) <= len(d_) and pos + len(ch) < len(toks_):   # all matched: next chunk
                rt_ = commit(ch, sn_, len(ch), False)
                if ts_ is not None:
                    ts_.append(_now()); ts_ += gs_ + [cp_]           # t4: commit done
                tps_.append(rt_ if rt_ is not None else tp_)
                h.update(np.array(lg_.view(mx.uint32)).tobytes())
                new_ += d_[pos:pos + len(ch)]
                pos += len(ch)
                _call(ci, len(ch), len(ch), kc, ts_)
                ci, w_ = ci + 1, WL[0]                  # (C2c) later calls: F12_WL rows (default 4)
                continue
            rt_ = commit(ch, sn_, m_ + 1, False)
            if ts_ is not None:
                ts_.append(_now()); ts_ += gs_ + [cp_]               # t4: commit done
            tps_.append((rt_ if rt_ is not None else tp_)[: m_ + 1])
            h.update(np.array(lg_[: m_ + 1].view(mx.uint32)).tobytes())
            new_ += d_[pos:pos + m_] + [g_[m_]]
            _call(ci, len(ch), m_ + 1, kc, ts_)
            return new_, mx.concatenate(tps_) if len(tps_) > 1 else tps_[0]

    _ML = MEMLOG[0]                                    # (Tue 11:3x, S1q memory) per check: MLX peak after each phase
    _mem = (lambda ph_: _ML.append((st["checks"], ph_, mx.get_active_memory(), mx.get_peak_memory(), mx.get_cache_memory()))) \
        if _ML is not None else (lambda ph_: None)
    try:
        while len(out) < max_tokens:
            if _ML is not None:
                mx.reset_peak_memory()
            _t = time.perf_counter()
            _k = _mk()
            pth, cf = propose(mx.array([[y] + [MASK] * (R - 1)]), ctx[None].astype(mx.bfloat16), dcache)
            mx.eval(pth, cf)
            st["t_draft"] += time.perf_counter() - _t
            _ph("draft", _k)
            _mem("draft")
            d = np.array(pth[0]).tolist()
            cf = np.array(cf[0]).tolist()
            calrec = {}
            if (trim := dcache[0].offset - a) > 0:
                DF._trim_recent_cache(dcache, trim)
            d = d[: max(0, max_tokens - len(out) - 1)]
            toks = [y] + d
            S = len(toks)
            use_p1 = theta is not None and skip_left == 0 and (cfmin is None or (cf and cf[0] >= cfmin))
            if _U01B[0] and S <= 2:                    # (S1u) the cap's last short check takes the exact check
                use_p1 = False
            if _QOM0[0] and S <= _SMAXQ():             # (S1q) at MAXK 92 calls of <= 4 rows are narrow: exact check
                use_p1 = False
            if theta is not None and skip_left > 0:
                skip_left -= 1
                st["skipped"] += 1
            elif theta is not None and not use_p1:
                st["cfgate"] += 1
            if use_p1:
                _t, _m = time.perf_counter(), _miss()
                _k = _mk()
                _p1 = None if TS1[0] is None else dict(t=[_now()], g=[_gs()], c=[time.thread_time()], b=[_diskread()])   # (S1u prof)
                if _PF2[0]:                            # P2-PREFETCH: the rows whose draft the drafter is unsure of
                    PF2_ROWS[0] = mx.array([c_ < PF2_CF for c_ in cf[:len(d)]] + [False])
                try:
                    lg1, tp1, sn1 = forward(toks, True)
                finally:
                    PF2_ROWS[0] = None                 # the pass-1 graph already holds the mask
                if _p1 is not None:                    # t1: the graph is built (any syncs inside forward included)
                    _p1["t"].append(_now()); _p1["g"].append(_gs()); _p1["c"].append(time.thread_time())
                p1 = mx.max(mx.softmax(lg1, axis=-1), axis=-1)
                g1 = mx.argmax(lg1, axis=-1)
                extra = []                             # P1-K1 / P2-PREFETCH counters ride on pass 1's own eval
                cal_t2 = []                            # calibration: pass 1's top-2 probabilities ride on the same eval
                if CAL[0] is not None:                 # (kept out of `extra`, which is read by index below)
                    cal_t2 = [mx.topk(mx.softmax(lg1.astype(mx.float32), axis=-1), 2, axis=-1)]
                if K1U_ACC:
                    extra.append(mx.stack([a for a, _ in K1U_ACC]).sum(axis=0))
                if PF2_ACC:
                    extra.append(mx.stack(PF2_ACC).sum().reshape(1))
                q0 = [mx.stack(QOM0_ACC).sum(axis=0)] if QOM0_ACC else []   # (S1q) residents added / dropped for room
                mx.eval(p1, g1, tp1, *extra, *cal_t2, *q0)
                if q0:
                    qa = q0[0].tolist()
                    st["q0_added"] = st.get("q0_added", 0) + qa[0]; st["q0_over"] = st.get("q0_over", 0) + qa[1]
                    st["q0_bad"] = st.get("q0_bad", 0) + qa[2]  # residents the post kernel did not see as resident
                    st["q0_calls"] = st.get("q0_calls", 0) + len(QOM0_ACC)
                    QOM0_ACC.clear()
                if _p1 is not None:                    # t2: the GPU + loader work of the call is done
                    _p1["t"].append(_now()); _p1["g"].append(_gs()); _p1["c"].append(time.thread_time())
                    _p1["b"].append(_diskread())
                if K1U_ACC:
                    ka = extra[0].tolist()
                    st["k1u_kept"] += ka[0]; st["k1u_over"] += ka[1]; st["k1u_rl"] += sum(s_ for _, s_ in K1U_ACC)
                    K1U_ACC.clear()
                if PF2_ACC:
                    st["pf2_ids"] += int(extra[-1].item()); st["pf2_calls"] += len(PF2_ACC)
                    PF2_ACC.clear()
                st["t_p1"] += time.perf_counter() - _t
                st["miss1"] += _miss() - _m
                _ph("p1", _k)
                _mem("p1")
                _t = time.perf_counter()
                _k = _mk()
                p1, g1 = p1.tolist(), g1.tolist()
                if cal_t2:
                    v2 = np.array(cal_t2[0])
                    calrec.update(p1=[round(x, 5) for x in p1], g1=g1,
                                  m1=[round(float(r.max() - r.min()), 5) for r in v2.reshape(-1, 2)])
                st["rows1"] += S
                n = 0
                while n < len(d) and p1[n] >= theta and d[n] == g1[n]:
                    n += 1
                if p1[n] >= theta:                     # pass 1 decides the whole check
                    new = d[:n] + [g1[n]]
                    rt = commit(toks, sn1, n + 1, True)
                    st["t_commit"] += time.perf_counter() - _t
                    _ph("commit", _k)
                    if _ML is not None:
                        mx.eval(rt) if rt is not None else None
                        _mem(f"commit{n + 1}")         # the kept count: the PLE layer replays over these rows
                    if _p1 is not None:                # t3: the commit is done; one TS1 row per wide call
                        _p1["t"].append(_now())
                        TS1[0].append((st["checks"], -1, S, n + 1) + tuple(_p1["t"]) + tuple(_p1["g"][0]) + tuple(_p1["g"][1])
                                      + tuple(_p1["g"][2]) + tuple(_p1["c"]) + tuple(_p1["b"]))
                    kt = (rt if rt is not None else tp1)[: n + 1]
                    h.update(np.array(lg1[: n + 1].view(mx.uint32)).tobytes())
                    if skip is not None:
                        streak = 0
                    path = 1
                else:                                  # keep rows 0..n-1, pass 2 on rows n..j
                    j = n
                    while j < len(d) and (p2_all or g1[j] == d[j]):
                        j += 1
                    if n == 0:
                        rollback_after_verify(cache, sn1, verified_tokens=S)
                        if S > 1:
                            model.clear_verify_capture(cache)
                        tkeep = None
                    else:
                        rt = commit(toks, sn1, n, True)
                        tkeep = (rt if rt is not None else tp1)[:n]
                        h.update(np.array(lg1[:n].view(mx.uint32)).tobytes())
                    st["t_commit"] += time.perf_counter() - _t
                    _ph("commit", _k)
                    new2, tp2 = exact(toks[n:j + 1], d[n:j])
                    new = d[:n] + new2
                    kt = tp2 if tkeep is None else mx.concatenate([tkeep, tp2])
                    st["pass2"] += 1
                    if skip is not None:
                        streak += 1
                        if streak >= skip[0]:
                            streak, skip_left = 0, skip[1]
                    path = 2
            else:                                      # the exact check (cf gate, the history rule, or theta None)
                new, kt = exact(toks, d, w1_pick(cf[0] if cf else None))
                st["exact"] += 1
                path = 0
            st["checks"] += 1
            if log is not None:
                log.append((S, path, len(new), round(cf[0], 4) if cf else None))
            if CAL[0] is not None:
                CAL[0].append(dict(path=path, d=d, cf=[round(c_, 5) for c_ in cf[:len(d)]], new=list(new), **calrec))
            c0 = len(out)
            out += new[: max_tokens - len(out)]
            ctx, a = kt[: len(new)], a + len(new)
            if stop_ids and any(t in stop_ids for t in out[c0:]):
                j_ = next(i for i in range(c0, len(out)) if out[i] in stop_ids)
                del out[j_ + 1:]
                break
            y = new[-1]
    finally:
        B.AW[0] = aw0
        T2._P1[0] = False
        TAP["on"] = False
    t2 = time.perf_counter()
    st.update(pf_s=round(t1 - t0, 3), dec_s=round(t2 - t1, 3), tok_s=len(out) / max(1e-9, t2 - t1),
              kept=len(out) / max(1, st["checks"]), lsha=h.hexdigest()[:10])
    return out, st
