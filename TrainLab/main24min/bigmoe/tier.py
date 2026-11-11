"""Rank-tier experts (BIGMOE_TIER=4433333333, DESIGN_3bit.md): gate ranks with 4 bits run the exact 4-bit expert, rank
with 3 bits its MSB-3 tier (m = q >> 1, scale 2s, bias bf16(b + s*mean(q & 1))). Decode and verify rows (S <= 8) only;
prefill stays exact 4-bit.

Storage: records3/ [MSB-3 | LSB-1] per expert. Per real layer L: a base pool (MSB-3 part, every routed expert, ext layer
L) and an LSB sidecar (ext layer side0 + L, only the experts at 4-bit ranks). 4-bit ranks rebuild q = 2m + l on the GPU
into per-call scratch rows (3b); MoonShots' fused 3/4-bit GLU (3c) replaces that later.
Tier-agnostic by construction: the plan is a list of (rank range, bits) segments; a segment's bits must be the base
pool's (3) or 4 (base + sidecar). Other tails (2-bit) need their record part first."""
from __future__ import annotations
import json, os
import numpy as np
import mlx.core as mx

PLAN = os.environ.get("BIGMOE_TIER", "")
ON = bool(PLAN)
_UNIFORM = ON and len(set(PLAN)) == 1          # 4444444444: exact 4-bit through MSB-3 + LSB-1 pools, no rank sort
STATS = {"calls": 0, "prefill_calls": 0}
_MSB = [("gate_proj", "w3"), ("gate_proj", "s3"), ("gate_proj", "b3"), ("up_proj", "w3"), ("up_proj", "s3"),
        ("up_proj", "b3"), ("down_proj", "w3"), ("down_proj", "s3"), ("down_proj", "b3")]
_LSB = [("gate_proj", "l1"), ("gate_proj", "b"), ("up_proj", "l1"), ("up_proj", "b"), ("down_proj", "l1"),
        ("down_proj", "b")]


def segments(plan):
    segs = []
    for r, c in enumerate(plan):
        n = int(c)
        if segs and segs[-1][2] == n:
            segs[-1][1] = r + 1
        else:
            segs.append([r, r + 1, n])
    return segs


# q = 2m + l: one thread per output word (8 weights) of the flat tensor; 3-bit codes are an MLX bits=3 little-endian
# bitstream (value i at bit 3i), low bits a 1-bit bitstream. Same bit math as the ext's CPU rebuild (spread8).
_RB_SRC = r"""
uint j = thread_position_in_grid.x;
uint bit = j * 24u, w = bit >> 5, sh = bit & 31u;
uint x = w3[w] >> sh;
if (sh > 8u) x |= w3[w + 1] << (32u - sh);
x &= 0xFFFFFFu;
uint l = (l1[j >> 2] >> ((j & 3u) * 8u)) & 0xFFu;
uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
a = (a & 0x07070707u) | ((a & 0x38383838u) << 1);
uint y = l; y = (y | (y << 12)) & 0x000F000Fu; y = (y | (y << 6)) & 0x03030303u; y = (y | (y << 3)) & 0x11111111u;
out[j] = (a << 1) | y;
"""
_rb = mx.fast.metal_kernel(name="bigmoe_tier_rebuild4", input_names=["w3", "l1"], output_names=["out"], source=_RB_SRC)


def rebuild4(w3, l1):
    """[..., K*3/32] 3-bit words + [..., K/32] low-bit words -> [..., K/8] 4-bit words (bit-exact)."""
    shape = (*w3.shape[:-1], w3.shape[-1] * 4 // 3)
    n = int(np.prod(shape))
    return _rb(inputs=[w3.reshape(-1), l1.reshape(-1)], output_shapes=[(n,)], output_dtypes=[mx.uint32],
               grid=(n, 1, 1), threadgroup=(256, 1, 1))[0].reshape(shape)


def _proj(h, pools, p, slots, bits):
    """One projection over gathered slots: bits 3 = the base pool as is; bits 4 = rebuilt from base + sidecar."""
    B, S = pools
    if bits == 3:
        return mx.gather_qmm(h, B[p]["w3"], B[p]["s3"], B[p]["b3"], rhs_indices=slots, transpose=True, group_size=64, bits=3)
    sm, sl = slots
    w = rebuild4(B[p]["w3"][sm], S[p]["l1"][sl])
    s = B[p]["s3"][sm] * mx.array(0.5, dtype=B[p]["s3"].dtype)     # exact: the converter checked s3 / 2 == s
    idx = mx.arange(sm.size, dtype=mx.uint32).reshape(sm.shape)
    return mx.gather_qmm(h, w, s, S[p]["b"][sl], rhs_indices=idx, transpose=True, group_size=64, bits=4)


_GLU = os.environ.get("BIGMOE_TIER_GLU", "3b")                # 3b: rebuild4 + gather_qmm; 3c: fused (MoonShots)
_COMBINE = os.environ.get("BIGMOE_TIER_COMBINE") == "1"        # 3c only: gate-weighted combine inside dispatch B
# BIGMOE_TIER_NEARTIE=1: count decode/verify rows (per layer) whose score gap across the 4-bit/3-bit rank boundary is
# below 2^-8 of the higher score (a bf16-scale near tie: the tier of those experts depends on rounding noise)
_NEARTIE = os.environ.get("BIGMOE_TIER_NEARTIE") == "1"
NEARTIE = {"acc": [], "near": 0, "rows": 0}


def neartie_summary():
    if NEARTIE["acc"]:
        NEARTIE["near"] += int(mx.stack(NEARTIE["acc"]).sum().item()); NEARTIE["acc"].clear()
    return NEARTIE["near"], NEARTIE["rows"]


def _tier_glu():
    import sys
    p = os.path.expanduser("~/MoonShots/patches/planes_engine")
    if p not in sys.path:
        sys.path.insert(0, p)
    from tierglu import tier_glu
    return tier_glu


_OOO = os.environ.get("BIGMOE_OOO") == "1"                   # MoEPrefetch: out-of-order experts (3c + combine)
OOO_STATS = {"calls": 0}


def _tier_glu_mod():
    _tier_glu()
    import tierglu
    return tierglu


def _ooo_call(x, ids, bm, t, post, sc, shared):
    """Resident pairs during the loader spin, then the rest, then the LY=8 combine: == 3c + fused combine bitwise
    (MoEPrefetch fork/apply_ooo.py, MoonShots OOO_PROPOSAL.md)."""
    TG = _tier_glu_mod()
    OOO_STATS["calls"] += 1
    K = ids.shape[-1]
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    lo4 = [s for s in t["segs"] if s[2] == 4]
    R4 = lo4[-1][1] if lo4 else 0
    assert not lo4 or lo4[0][0] == 0, t["segs"]                 # 4-bit ranks lead (tier_glu's layout)
    gate, L = bm["gate"], bm["L"]
    dep_m, n_m = post(L, ids)
    pre_m = dep_m[1:1 + n_m].reshape(M, K)
    ok = pre_m >= 0
    if R4:
        dep_l, n_l = post(t["side"], ids[..., :R4])
        pre_l = dep_l[1:1 + n_l].reshape(M, R4)
        ok = ok & mx.concatenate([pre_l >= 0, mx.ones((M, K - R4), mx.bool_)], axis=-1)
    phase = mx.where(ok, 1, 2).astype(mx.int32)
    y1 = TG.tier_glu_phase(x2, t["base"], t["sideP"], pre_m, mx.maximum(pre_l, 0) if R4 else None, phase, 1, plan=PLAN)
    after = [y1] + ([shared] if shared is not None else [])     # encode the spins after the hit work (and the shared
    sm = gate.spin(mx.depends(dep_m, after), L, n_m).reshape(M, K)   # expert): they then run beside it on the GPU
    sl = gate.spin(mx.depends(dep_l, after), t["side"], n_l).reshape(M, R4) if R4 else None
    y2 = TG.tier_glu_phase(x2, t["base"], t["sideP"], sm, sl, phase, 2, plan=PLAN)
    return TG.combine_phase(y1, y2, phase, sc.reshape(-1, K)).reshape(*x.shape[:-1], -1)


def tier_call(self, x, ids, bm, route, post=None):
    """Decode/verify: ids = expert ids in gate-rank order [..., K]. route(L, ids) -> slots (the streamed gate)."""
    STATS["calls"] += 1
    t = bm["tier"]
    if _OOO and post is not None and _GLU == "3c" and t.get("scores") is not None:
        return _ooo_call(x, ids, bm, t, post, t.pop("scores"), t.pop("shared", None))
    sm = route(bm["L"], ids)                                    # base pool: every rank
    lo4 = [s for s in t["segs"] if s[2] == 4]
    sl = route(t["side"], ids[..., lo4[0][0]:lo4[-1][1]]) if lo4 else None
    if _GLU == "3c":                                            # MoonShots' fused 3/4-bit GLU: bit-identical to 3b
        K = ids.shape[-1]
        sc = t.pop("scores", None)                              # BIGMOE_TIER_COMBINE: the block's combine in dispatch B
        y = _tier_glu()(x.reshape(-1, x.shape[-1]), t["base"], t["sideP"], sm.reshape(-1, K),
                        sl.reshape(-1, sl.shape[-1]) if sl is not None else None, plan=PLAN,
                        **({"scores": sc.reshape(-1, K)} if sc is not None else {}))
        return y.reshape(*x.shape[:-1], -1) if sc is not None else y.reshape(*x.shape[:-1], K, -1)
    xe = mx.expand_dims(x, (-2, -3))
    pools = (t["base"], t["sideP"])
    outs = []
    for a, b, n in t["segs"]:
        if n == 4:
            r0 = lo4[0][0]
            sl_ab = (sm[..., a:b], sl[..., a - r0:b - r0])
        else:
            sl_ab = sm[..., a:b]
        up = _proj(xe, pools, "up_proj", sl_ab, n)
        gt = _proj(xe, pools, "gate_proj", sl_ab, n)
        outs.append(_proj(self.activation(up, gt), pools, "down_proj", sl_ab, n).squeeze(-2))
    return mx.concatenate(outs, axis=-2) if len(outs) > 1 else outs[0]


# ---------------------------------------------------------------- install (ExpertStreamer.install with BIGMOE_TIER)
def load_index(model_dir):
    d = os.path.join(model_dir, "records3")
    return d, json.load(open(os.path.join(d, "index.json")))


def _shapes(pers4_shapes):
    """4-bit piece shapes [(out, K/8), (out, K/64), (out, K/64)] x 3 -> MSB-3 and LSB-1 pool piece shapes."""
    msb, lsb = [], []
    shp = [sh if isinstance(sh[0], int) else sh[0] for sh in pers4_shapes]   # shapes or (shape, dtype) pairs
    for p in range(3):
        (o, w8), (_, g), _ = shp[3 * p:3 * p + 3]
        msb += [((o, w8 * 3 // 4), mx.uint32), ((o, g), mx.bfloat16), ((o, g), mx.bfloat16)]
        lsb += [((o, w8 // 4), mx.uint32), ((o, g), mx.bfloat16)]
    return msb, lsb


def _nbytes(sh, dt): return int(np.prod(sh)) * (4 if dt == mx.uint32 else 2)


def make_pools(ext, L, E, C_m, shapes4, rec_dir, r):
    """Base pool for layer L as ext layer L (must be the next add_layer index)."""
    msb, _ = _shapes(shapes4)
    P = [mx.zeros((C_m, *sh), dt) for sh, dt in msb]
    mx.eval(*P)
    pers = [_nbytes(sh, dt) for sh, dt in msb]
    assert sum(pers) <= r["msb_sz"], (L, sum(pers), r["msb_sz"])
    assert ext.add_layer(C_m, E, [0] * 9, [0] * 9, pers, P) == L
    ext.set_records_part(L, os.path.join(rec_dir, r["file"]), r["recsz"], 0, r["msb_sz"])
    return {p: {f: P[i] for i, (pp, f) in enumerate(_MSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}, P


def make_sidecar(ext, E, C_l, shapes4, rec_dir, r):
    _, lsb = _shapes(shapes4)
    P = [mx.zeros((C_l, *sh), dt) for sh, dt in lsb]
    mx.eval(*P)
    pers = [_nbytes(sh, dt) for sh, dt in lsb]
    Ls = ext.add_layer(C_l, E, [0] * 6, [0] * 6, pers, P)
    ext.set_records_part(Ls, os.path.join(rec_dir, r["file"]), r["recsz"], r["msb_sz"], r["recsz"] - r["msb_sz"])
    return Ls, {p: {f: P[i] for i, (pp, f) in enumerate(_LSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}, P


# ---------------------------------------------------------------- exact prefill in plan mode


def prefill_call(cls, self, x, indices, bm, *a, **kw):
    """Exact 4-bit prefill, the reference's own call shape (stream._sync_call): a scratch of exactly the experts this call
    uses (sorted ids), filled from records3 with the CPU rebuild, and compact order-preserving ids into it. (A full-E
    scratch indexed by expert id changes MLX's gather kernel choice on short prompts: B7 prose sha.)"""
    from .stream import PIECES
    STATS["prefill_calls"] += 1
    t = bm["tier"]
    ids = np.array(indices)
    uniq = np.unique(ids)
    if _PF_GROUP and len(uniq) > _PF_GROUP and ids.size >= 16 and ids.size >= 4 * len(uniq):
        y = _prefill_grouped(cls, self, x, indices, ids, uniq, bm, *a, **kw)
        if y is not None:
            return y
    compact = np.searchsorted(uniq, ids).astype(np.int32)
    scr = [mx.zeros((len(uniq), *sh), dt) for sh, dt in t["shapes4"]]
    mx.eval(*scr)
    t["ext"].read_rebuilt(t["path"], t["recsz"], t["msb_sz"], t["pers4"], uniq.astype(np.int32).tolist(),
                          list(range(len(uniq))), scr)
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    for (proj, field), arr in zip(PIECES, scr):
        setattr(getattr(self, proj), field, arr)
    try:
        y = cls.__call__(self, x, mx.array(compact).astype(indices.dtype), *a, **kw)
        mx.eval(y)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    del scr
    mx.clear_cache()
    return y


def prefill_done():
    pass


# ---------------------------------------------------------------- grouped exact prefill (TrainLab D1i, 10-22)
# BIGMOE_TIER_PF_GROUP=G: the call's experts run in ascending-id groups of ~G, each group on its own (row, slot) pairs
# with a scratch of only its experts, read_rebuilt of group g+1 running while the GPU computes group g. Bit-identical to
# the single call: only when the reference call itself takes the sorted rhs gather (>= 16 pairs, >= 4 per expert), and
# every group keeps it too (>= 16 rows, >= 4 per expert; a group grows past G until it holds, a short tail joins the one
# before), rows padded to a multiple of 64 (BUG1); the pairs go back to their (row, slot) places before the model's own
# combine. (v3's non-tier PF_GROUP lacked these conditions and was not exact on prose, LEDGER M24d/M24e.)
_PF_GROUP = int(os.environ.get("BIGMOE_TIER_PF_GROUP", "0") or 0)


def _groups(uniq, flat, G):
    cnt = np.bincount(flat, minlength=int(uniq[-1]) + 1)[uniq]
    ok = lambda n, r: r >= 16 and r >= 4 * n
    groups, a, rows = [], 0, 0
    for i in range(len(uniq)):
        rows += int(cnt[i])
        n = i + 1 - a
        if n >= G and ok(n, rows):
            groups.append(uniq[a:i + 1]); a, rows = i + 1, 0
    if a < len(uniq):
        if ok(len(uniq) - a, rows) or not groups:
            groups.append(uniq[a:])
        else:
            groups[-1] = uniq[len(uniq) - len(groups[-1]) - (len(uniq) - a):]
    if max(len(g) for g in groups) > 2 * G:
        return None
    for g in groups:
        if not ok(len(g), int(cnt[np.searchsorted(uniq, g)].sum())):
            return None
    return groups


def _prefill_grouped(cls, self, x, indices, ids, uniq, bm, *a, **kw):
    from .stream import PIECES
    t = bm["tier"]
    k = ids.shape[-1]
    lead = x.shape[:-1]
    flat = ids.reshape(-1)                                       # pair p = row * k + slot
    groups = _groups(uniq, flat, _PF_GROUP)
    if groups is None:
        return None
    poss = [np.nonzero(np.isin(flat, grp))[0] for grp in groups]
    G = max(len(g) for g in groups)
    x2 = x.reshape(-1, x.shape[-1])
    nb = min(2, len(groups))
    bufs = [[mx.zeros((G, *sh), dt) for sh, dt in t["shapes4"]] for _ in range(nb)]
    mx.eval(*[arr for b_ in bufs for arr in b_])

    def fill(i):
        grp = groups[i]
        t["ext"].read_rebuilt(t["path"], t["recsz"], t["msb_sz"], t["pers4"], grp.astype(np.int32).tolist(),
                              list(range(len(grp))), bufs[i % nb])

    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    outs, prev = [], []
    fill(0)
    try:
        for i, (grp, pos) in enumerate(zip(groups, poss)):
            scr = bufs[i % nb] if len(grp) == G else [arr[:len(grp)] for arr in bufs[i % nb]]
            for (proj, field), arr in zip(PIECES, scr):
                setattr(getattr(self, proj), field, arr)
            xp = mx.take(x2, mx.array((pos // k).astype(np.int32)), axis=0)
            cpn = np.searchsorted(grp, flat[pos]).astype(np.int32)
            pad = (-len(pos)) % 64
            if pad:
                xp = mx.concatenate([xp, mx.zeros((pad, xp.shape[-1]), xp.dtype)])
                cpn = np.concatenate([cpn, np.zeros(pad, np.int32)])
            cp = mx.array(cpn).reshape(-1, 1).astype(indices.dtype)
            y = cls.__call__(self, xp, cp, *a, **kw).reshape(len(cpn), -1)[:len(pos)]
            mx.async_eval(y)
            if i + 1 < len(groups):
                if prev:
                    mx.eval(*prev)                               # the other buffer's last user is finished
                fill(i + 1)
            prev = [y]
            outs.append(y)
        mx.eval(*prev)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    order = np.argsort(np.concatenate(poss), kind="stable").astype(np.int32)
    y = mx.take(mx.concatenate(outs, axis=0), mx.array(order), axis=0).reshape(*lead, k, -1)
    mx.eval(y)
    del bufs, outs
    mx.clear_cache()
    return y


# ---------------------------------------------------------------- MoE block: rank-ordered top-k on streamed tier layers
def install_block():
    """Decode/verify rows (<= 8) on tier layers: the top-k sorted by score (descending), as planes125's evaluated
    variant, so ranks are positions; shared expert first with Scout's spin dependency edge when that patch is loaded."""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_context import current_attention_phase
    from mtplx.attention_math import attention_gate
    import sys
    SF = next((m for m in list(sys.modules.values())       # Scout bundle's sharedfirst, if loaded (by file, not name)
               if getattr(m, "__file__", "") and m.__file__.endswith("Scout/patches/sharedfirst.py")), None)
    orig = Q.SparseMoeBlock.__call__

    def _call(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        if (bm is None or "tier" not in bm or x.ndim < 2 or x.shape[-2] > 8 or Q._moe_prefill_combine_applies(self, x)
                or getattr(self, "sharding_group", None) is not None):
            return orig(self, x)
        if current_attention_phase() == "decode_verify":
            shared = attention_gate(self.shared_expert(x), self.shared_expert_gate(x))
        else:
            shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if not _UNIFORM:                                        # one tier for every rank (e.g. 4444444444 = exact via the
            order = mx.argsort(-scores, axis=-1)                # plane pools): keep the reference's argpartition order
            inds, scores = mx.take_along_axis(inds, order, axis=-1), mx.take_along_axis(scores, order, axis=-1)
        if _NEARTIE and not _UNIFORM:                                            # rows whose 4-bit/3-bit rank boundary is a near tie
            nb = int("".join(PLAN).rfind("4")) + 1              # first 3-bit rank
            s32 = scores.astype(mx.float32)
            NEARTIE["acc"].append(((s32[..., nb - 1] - s32[..., nb]) < 2.0 ** -8 * s32[..., nb - 1]).sum())
            NEARTIE["rows"] += int(np.prod(scores.shape[:-1]))
            if len(NEARTIE["acc"]) >= 480:
                NEARTIE["near"] += int(mx.stack(NEARTIE["acc"]).sum().item()); NEARTIE["acc"].clear()
        dep = SF is not None and SF.S["on"]
        if dep:
            SF.S["dep"] = shared
        fused = _COMBINE and _GLU == "3c"
        if fused:                                               # tier_call returns the combined [..., D] (bitwise)
            bm["tier"]["scores"] = scores
            if dep:
                bm["tier"]["shared"] = shared                   # OoO: the spins also wait behind the shared expert
        try:
            y = self.switch_mlp(x, inds)
        finally:
            if dep:
                SF.S["dep"] = None
            bm["tier"].pop("scores", None)
            bm["tier"].pop("shared", None)
        return (y if fused else (y * scores[..., None]).sum(axis=-2)) + shared

    Q.SparseMoeBlock.__call__ = _call
