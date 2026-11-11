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
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # [v3f] tierglu.py pinned beside the package
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
    _t0 = _time.perf_counter()
    ids = np.array(indices)                                      # the per-layer sync: waits for attention + router
    _t1 = _time.perf_counter()
    PF["t_sync"] = PF.get("t_sync", 0.0) + (_t1 - _t0)
    _mem_in()                                                    # D1i: where the prefill peak sits (before / in the MoE)
    try:
        if STREAM is not None and bm["L"] in STREAM.L0:            # D1i: fixed-order streamed experts
            PF["grouped"] += 1
            return _prefill_streamed(cls, self, x, indices, ids, bm, STREAM)
        uniq = np.unique(ids)
        if _RESCOUNT:                                            # D1k: how much of the union the decode pools already hold
            ul = uniq.astype(np.int32).tolist()
            base = np.array(t["ext"].lookup(bm["L"], ul)) >= 0         # MSB-3 part resident (base pool)
            side = np.array(t["ext"].lookup(t["side"], ul)) >= 0       # LSB-1 part resident (sidecar)
            PF["rc_calls"] = PF.get("rc_calls", 0) + 1
            PF["rc_union"] = PF.get("rc_union", 0) + len(ul)
            PF["rc_base"] = PF.get("rc_base", 0) + int(base.sum())
            PF["rc_both"] = PF.get("rc_both", 0) + int((base & side).sum())
        warm = _Warm(bm, ids) if (_WARM and (PF["last"] or _WARM_ALL)) else None   # D1i: warm the pools from the last chunk
        if _PF_GROUP and len(uniq) > _PF_GROUP and ids.size >= 16 and ids.size >= 4 * len(uniq):
            y = _prefill_grouped(cls, self, x, indices, ids, uniq, bm, warm, *a, **kw)   # D1i: experts in sub-batches
            if y is not None:
                return y
        PF["single"] += 1
        return _prefill_single(cls, self, x, indices, ids, uniq, bm, warm, *a, **kw)
    finally:
        _mem_out()
        PF["t_moe"] = PF.get("t_moe", 0.0) + (_time.perf_counter() - _t1)
        if LA is not None and STREAM is None:                  # D1k: layer L's reads are done: stage layer L + 1's
            _la_advise(bm, x)                                    # predicted union in the OS read-ahead


# D1k (Breakthrough 05:15 Thu): LOOK-AHEAD READS for the exact prefill. Once layer L's experts are read, layer L + 1's
# predicted union (its router + MoEPrefetch's d=1 correction on layer L's MoE input, top-K per token) goes to the OS
# read-ahead (F_RDADVISE on a cached fd), so layer L + 1's F_NOCACHE reads can find those pages staged while layer L
# computes and layer L + 1's attention runs. Same reads, same bytes, same outputs. Off unless the driver sets LA
# (bt_b12: B12_PF_LA=1): {"L0": {L: i}, "bms": [...], "routers": [...], "corr": [(A, B, mu, delta)] | None, "k": K}.
LA = None
_LA_FDS = {}
_LA_POOL = None
_F_RDADVISE = 44                                                 # <sys/fcntl.h>; struct radvisory {off_t; int}


def _la_advise(bm, x):
    import fcntl, struct
    la = LA
    t0 = _time.perf_counter()
    try:
        i = la["L0"].get(bm["L"])
        if i is None or i + 1 >= len(la["bms"]):
            return
        nxt = la["bms"][i + 1].get("tier")
        if nxt is None:
            return
        x2 = x.reshape(-1, x.shape[-1])
        lg = la["routers"][i + 1](x2).astype(mx.float32)
        if la["corr"] is not None and i < len(la["corr"]):
            A, B, mu, dl = la["corr"][i]
            lg = lg + ((x2.astype(mx.float32) - mu) @ A) @ B + dl
        k = la["k"]
        u = np.unique(np.array(mx.argpartition(-lg, kth=k - 1, axis=-1)[:, :k]))
        path, rs = nxt["path"], nxt["recsz"]
        fd = _LA_FDS.get(path)
        if fd is None:
            fd = _LA_FDS[path] = os.open(path, os.O_RDONLY)    # cached fd: the advisories fill the unified buffer cache
        global _LA_POOL
        if _LA_POOL is None:                                     # an advisory costs ~0.14 ms of the caller's time
            from concurrent.futures import ThreadPoolExecutor    # (05:45 test): issue them off the main thread
            _LA_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("B12_PF_LA_THREADS", "8")))
        ul, w = u.tolist(), _LA_POOL._max_workers

        def adv(part):
            for e in part:
                fcntl.fcntl(fd, _F_RDADVISE, struct.pack("qi4x", int(e) * rs, rs))
        for j in range(w):                                       # ascending ids per worker: near-sequential offsets
            if ul[j::w]:
                _LA_POOL.submit(adv, ul[j::w])
        la["n"] = la.get("n", 0) + len(u)
    except Exception as ex:                                      # never break the prefill for a hint
        la["err"] = repr(ex)
    finally:
        la["t"] = la.get("t", 0.0) + (_time.perf_counter() - t0)


def _prefill_single(cls, self, x, indices, ids, uniq, bm, warm, *a, **kw):
    from .stream import PIECES
    t = bm["tier"]
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
    if warm is not None:
        warm.take(scr, uniq); warm.adopt(bm)
    del scr
    mx.clear_cache()
    return y


def prefill_done():
    pass


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


# ---------------------------------------------------------------- D1i (TrainLab, 10-22): sub-batched exact prefill + warm
# NewIdeas / Breakthrough IDEAS D1i. BIGMOE_TIER_PF_GROUP=G: the exact 4-bit prefill of a layer call runs in sub-batches of
# <= G experts (ascending ids), each on its own (row, slot) pairs with a scratch of only its experts: G x 2.77 MB instead
# of up to ~380 x 2.77 MB at 2048 tokens. Same construction as stream._sync_grouped (BIGMOE_PF_GROUP, v3's 24 GB mode):
# per-pair outputs do not depend on the other pairs in the gather while every expert's pairs stay in one call and the
# call keeps the reference's kernel (sorted rhs gather: >= 16 rows, >= 4 rows per expert; row counts padded to 64 for
# BUG1); the pairs go back to their (row, slot) positions before the model's own combine. Two scratch buffers: group
# g + 1 is read from records3 while the GPU runs group g.
# BIGMOE_TIER_WARM=1 (=all: every chunk): during the call that runs with PF["last"] set (the driver sets it for the last chunk), each
# layer's most-used experts of that chunk (by pair count) are converted from the 4-bit scratch into the pools' MSB-3 and
# LSB-1 layouts (DESIGN_3bit.md plane math, bit-exact vs records3) and copied into the pools with ext.adopt: C_m base
# slots and C_l sidecar slots, no SSD reads.
import time as _time
_PF_GROUP = int(os.environ.get("BIGMOE_TIER_PF_GROUP", "0") or 0)
_RESCOUNT = os.environ.get("BIGMOE_TIER_PF_RESCOUNT") == "1"   # D1k: count the union's experts already resident (no effect)
_WARM = os.environ.get("BIGMOE_TIER_WARM", "0") in ("1", "all")
_WARM_ALL = os.environ.get("BIGMOE_TIER_WARM") == "all"       # every chunk (drivers that do not set PF["last"]:
                                                              # spec125); the last chunk's experts win
_WARM_BATCH = 8                                               # experts converted per MLX batch
PF = {"last": False, "grouped": 0, "single": 0, "fallback": 0, "groups": 0, "t_read": 0.0, "warm_base": 0,
      "warm_side": 0, "t_warm": 0.0, "act_in_gb": 0.0, "peak_pre_gb": 0.0, "peak_moe_gb": 0.0}


def _mem_in():
    """at a prefill MoE call's entry: MLX active memory now, and the peak since the last call ended (the layer's
    attention / hyper-connection part); maxima over the prefill"""
    PF["act_in_gb"] = max(PF["act_in_gb"], round(mx.get_active_memory() / 1e9, 2))
    PF["peak_pre_gb"] = max(PF["peak_pre_gb"], round(mx.get_peak_memory() / 1e9, 2))
    mx.reset_peak_memory()


def _mem_out():
    PF["peak_moe_gb"] = max(PF["peak_moe_gb"], round(mx.get_peak_memory() / 1e9, 2))
    mx.reset_peak_memory()
_SH0 = mx.array([3 * i for i in range(0, 11)], dtype=mx.uint32)
_SH1 = mx.array([3 * i - 32 for i in range(11, 22)], dtype=mx.uint32)
_SH2 = mx.array([3 * i - 64 for i in range(22, 32)], dtype=mx.uint32)
_SH4 = mx.array([4 * i for i in range(8)], dtype=mx.uint32)
_SHB = mx.arange(32, dtype=mx.uint32)


def planes(w4, s, b):
    """4-bit pieces of one projection [n, out, K/8] u32, [n, out, K/64] bf16 x 2 -> (w3, s3, b3), (l1, b) as records3
    stores them: m = q >> 1 as an MLX bits=3 bitstream (value i at bit 3i), s3 = 2s, b3 = bf16(fp32(b) + fp32(s) * mean
    of l over the group), l1 = l = q & 1 as a little-endian bitstream, b unchanged."""
    n, o, w8 = w4.shape
    K = w8 * 8
    q = ((w4[..., None] >> _SH4) & 15).reshape(n, o, K // 32, 32)
    m, l = q >> 1, q & 1
    w0 = (m[..., 0:11] << _SH0).sum(-1).astype(mx.uint32)
    w1 = ((m[..., 10] >> 2) + (m[..., 11:22] << _SH1).sum(-1)).astype(mx.uint32)
    w2 = ((m[..., 21] >> 1) + (m[..., 22:32] << _SH2).sum(-1)).astype(mx.uint32)
    w3 = mx.stack([w0, w1, w2], axis=-1).reshape(n, o, 3 * K // 32)
    l1 = (l << _SHB).sum(-1).astype(mx.uint32)
    off = l.reshape(n, o, K // 64, 64).sum(-1).astype(mx.float32) / 64.0
    b3 = (b.astype(mx.float32) + s.astype(mx.float32) * off).astype(mx.bfloat16)
    s3 = (s.astype(mx.float32) * 2.0).astype(mx.bfloat16)
    return (w3, s3, b3), (l1, b)


def planes_expert(scr, rows):
    """rows of a 4-bit scratch (PIECES order) -> (9 base pieces in _MSB order, 6 sidecar pieces in _LSB order)."""
    r = mx.array(np.asarray(rows, np.int32))
    msb, lsb = [], []
    for p in range(3):
        (w3, s3, b3), (l1, b) = planes(scr[3 * p][r], scr[3 * p + 1][r], scr[3 * p + 2][r])
        msb += [w3, s3, b3]; lsb += [l1, b]
    return msb, lsb


class _Warm:
    """The last chunk's most-used experts of one layer call, gathered group by group, then adopted into the pools."""

    def __init__(self, bm, ids):
        t = bm["tier"]
        u, c = np.unique(ids, return_counts=True)
        o = np.lexsort((u, -c))                                  # most pairs first, ties by id
        self.base = [int(e) for e in u[o][:bm["C"]]]
        self.side = [int(e) for e in u[o][:t["C_l"]]]
        self.want = set(self.base)                               # the sidecar's experts are a prefix of the base's
        self.msb, self.lsb, self.got, self.pend = [], [], [], []

    def take(self, scr, grp):
        rows = [i for i, e in enumerate(grp) if int(e) in self.want]
        for a in range(0, len(rows), _WARM_BATCH):
            rr = rows[a:a + _WARM_BATCH]
            msb, lsb = planes_expert(scr, rr)
            self.msb.append(msb); self.lsb.append(lsb); self.got += [int(grp[i]) for i in rr]
            self.pend += msb + lsb
            mx.eval(*self.pend); self.pend = []                  # small batches: the transient stays ~80 MB

    def adopt(self, bm):
        if not self.got:
            return
        t0 = _time.perf_counter()
        t, ext_ = bm["tier"], bm["tier"]["ext"]
        msb = [mx.concatenate([m[i] for m in self.msb]) for i in range(9)]
        lsb = [mx.concatenate([m[i] for m in self.lsb]) for i in range(6)]
        mx.eval(*msb, *lsb)
        where = {e: k for k, e in enumerate(self.got)}
        for L, arrs, cand in ((bm["L"], msb, self.base), (t["side"], lsb, self.side)):
            cand = [e for e in cand if e in where]
            ext_.stage_bufs(1, arrs, len(self.got))
            n = ext_.adopt(L, 1, [], cand, [where[e] for e in cand])
            PF["warm_base" if L == bm["L"] else "warm_side"] += n
        PF["t_warm"] += _time.perf_counter() - t0


def _groups(uniq, flat, G):
    """Consecutive (ascending-id) expert groups of ~G that each keep the sorted rhs gather (>= 16 rows and >= 4 rows
    per expert): a group grows past G until it holds, a short tail joins the group before it; None if a group would
    pass 2G (then the caller runs the single call)."""
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
    for g in groups:                                             # a merged tail could still be short: then no grouping
        r = int(cnt[np.searchsorted(uniq, g)].sum())
        if not ok(len(g), r):
            return None
    return groups


def _prefill_grouped(cls, self, x, indices, ids, uniq, bm, warm, *a, **kw):
    from .stream import PIECES
    t = bm["tier"]
    G = _PF_GROUP
    k = ids.shape[-1]
    lead = x.shape[:-1]
    flat = ids.reshape(-1)                                       # pair p = row * k + slot
    groups = _groups(uniq, flat, G)
    if groups is None:
        PF["fallback"] += 1                                      # no grouping keeps the reference's kernel
        return None
    poss = [np.nonzero(np.isin(flat, grp))[0] for grp in groups]  # ascending pair positions of each group
    PF["grouped"] += 1; PF["groups"] += len(groups)
    G = max(len(g) for g in groups)                              # buffer rows: the largest group
    if _PF_GPU:
        return _prefill_grouped_gpu(cls, self, x, indices, ids, uniq, bm, warm, groups, poss, *a, **kw)
    x2 = x.reshape(-1, x.shape[-1])
    nb = min(2, len(groups))
    bufs = [[mx.zeros((G, *sh), dt) for sh, dt in t["shapes4"]] for _ in range(nb)]
    mx.eval(*[arr for b_ in bufs for arr in b_])

    def fill(i):
        t0 = _time.perf_counter()
        grp = groups[i]
        t["ext"].read_rebuilt(t["path"], t["recsz"], t["msb_sz"], t["pers4"], grp.astype(np.int32).tolist(),
                              list(range(len(grp))), bufs[i % nb])
        PF["t_read"] += _time.perf_counter() - t0

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
            pad = (-len(pos)) % 64                               # BUG1: keep the routed row count a multiple of 64
            if pad:
                xp = mx.concatenate([xp, mx.zeros((pad, xp.shape[-1]), xp.dtype)])
                cpn = np.concatenate([cpn, np.zeros(pad, np.int32)])
            cp = mx.array(cpn).reshape(-1, 1).astype(indices.dtype)
            y = cls.__call__(self, xp, cp, *a, **kw).reshape(len(cpn), -1)[:len(pos)]
            mx.async_eval(y)
            if warm is not None:
                warm.take(scr, grp)                              # reads this group's scratch: done before it is refilled
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
    if warm is not None:
        warm.adopt(bm)
    mx.clear_cache()
    return y


# D1i (NewIdeas 03:55): BIGMOE_TIER_PF_GPU=1: the grouped prefill reads RAW records3 records (both parts, one pread per
# expert, F_NOCACHE, a thread pool of BIGMOE_TIER_PF_THREADS) straight into an MLX buffer and rebuilds the 4-bit pieces on
# the GPU (rebuild4: bit-exact, s = s3 / 2 exact, the original bias from the LSB part), instead of read_rebuilt's CPU
# rebuild. Group g + 1's reads run while the GPU rebuilds and computes group g.
from concurrent.futures import ThreadPoolExecutor as _TPE
_PF_GPU = os.environ.get("BIGMOE_TIER_PF_GPU") == "1"
_POOL = None
_FDS = {}
_HALF = mx.array(0.5, dtype=mx.bfloat16)


def _fd(path):
    if path not in _FDS:
        import fcntl
        fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        _FDS[path] = fd
    return _FDS[path]


def _read_raw(t, grp, raw):
    """submit one pread per expert of grp into rows of raw ([G, recsz] uint8); returns the futures"""
    global _POOL
    if _POOL is None:
        _POOL = _TPE(max_workers=int(os.environ.get("BIGMOE_TIER_PF_THREADS", "16")))
    fd, rs = _fd(t["path"]), t["recsz"]
    mv = memoryview(raw).cast("B")
    return [_POOL.submit(os.preadv, fd, [mv[k * rs:(k + 1) * rs]], int(e) * rs) for k, e in enumerate(grp)]


def _rebuild_gpu(t, raw, n):
    """[n] raw records3 records -> the 9 exact 4-bit pieces (PIECES order) on the GPU"""
    r = t["r"]
    mo, ms, lo, ls = r["msb_piece_off"], r["msb_piece_size"], r["lsb_piece_off"], r["lsb_piece_size"]
    rows = raw[:n]
    piece = lambda off, size, dt: rows[:, off:off + size].view(dt)
    out = []
    for p in range(3):
        (wsh, _), (ssh, _), (bsh, _) = t["shapes4"][3 * p:3 * p + 3]
        w3 = piece(mo[3 * p], ms[3 * p], mx.uint32).reshape(n, wsh[0], -1)
        s3 = piece(mo[3 * p + 1], ms[3 * p + 1], mx.bfloat16).reshape(n, *ssh)
        l1 = piece(lo[2 * p], ls[2 * p], mx.uint32).reshape(n, wsh[0], -1)
        b = piece(lo[2 * p + 1], ls[2 * p + 1], mx.bfloat16).reshape(n, *bsh)
        out += [rebuild4(w3, l1).reshape(n, *wsh), s3 * _HALF, b]
    return out


def _prefill_grouped_gpu(cls, self, x, indices, ids, uniq, bm, warm, groups, poss, *a, **kw):
    from .stream import PIECES
    t = bm["tier"]
    G = max(len(g) for g in groups)
    k = ids.shape[-1]
    lead = x.shape[:-1]
    flat = ids.reshape(-1)
    x2 = x.reshape(-1, x.shape[-1])
    nb = min(2, len(groups))
    raws = [mx.zeros((G, t["recsz"] // 4), mx.uint32) if _RB2 else mx.zeros((G, t["recsz"]), mx.uint8)   # D1k (NewIdeas
            for _ in range(nb)]                                  # 05:55): RB2 = TrainLab's copy-free rebuild kernel
    mx.eval(*raws)
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    outs, prev = [], []
    fut = _read_raw(t, groups[0], raws[0])
    try:
        for i, (grp, pos) in enumerate(zip(groups, poss)):
            t0 = _time.perf_counter()
            for f in fut:
                if f.result() != t["recsz"]:
                    raise IOError(f"records3 short read in {t['path']}")
            PF["t_read"] += _time.perf_counter() - t0                  # time spent WAITING for reads
            scr = _rebuild_gpu2(t, raws[i % nb], len(grp)) if _RB2 else _rebuild_gpu(t, raws[i % nb], len(grp))
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
            if warm is not None:
                warm.take(scr, grp)
            if i + 1 < len(groups):
                if prev:
                    mx.eval(*prev)                                       # the other raw buffer's last user is done
                fut = _read_raw(t, groups[i + 1], raws[(i + 1) % nb])
            prev = [y]
            outs.append(y)
        mx.eval(*prev)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    order = np.argsort(np.concatenate(poss), kind="stable").astype(np.int32)
    y = mx.take(mx.concatenate(outs, axis=0), mx.array(order), axis=0).reshape(*lead, k, -1)
    mx.eval(y)
    del raws, outs
    if warm is not None:
        warm.adopt(bm)
    mx.clear_cache()
    return y


# D1i (NewIdeas 04:03): FIXED-ORDER EXPERT STREAMING for the decoupled prefill. Every tier layer's experts are read in
# id-range groups of G (all of them: at >= ~2K tokens a layer's prompt touches ~85-90 % of its 512 experts), in layer
# order, through a ring of R scratch buffers, independent of the routing. The driver pumps reads while the GPU runs a
# layer's attention (start_stream / pump / stop_stream); the layer's MoE then takes its groups from the ring and pumps
# the next reads while the GPU computes. read_rebuilt holds the GIL, so the pumps run on the main thread between
# asynchronous GPU launches. Per-pair outputs as in the grouped path (the scratch is the whole id range).
STREAM = None


class _Stream:
    def __init__(self, bms, R=4, G=64):
        t0 = bms[0]["tier"]
        self.bms, self.G, self.E = bms, G, t0["E"]
        self.L0 = {bm["L"]: i for i, bm in enumerate(bms)}
        self.ng = -(-self.E // G)
        self.bufs = [[mx.zeros((G, *sh), dt) for sh, dt in t0["shapes4"]] for _ in range(R)]
        mx.eval(*[a for b in self.bufs for a in b])
        self.free, self.pending, self.ready = list(range(R)), [], {}
        self.nxt, self.total = 0, len(bms) * self.ng
        self.stats = {"reads": 0, "t_read": 0.0, "t_wait": 0.0, "groups_used": 0, "groups_empty": 0}

    def _reclaim(self):
        b, users = self.pending.pop(0)
        t0 = _time.perf_counter()
        if users:
            mx.eval(*users)                                      # the GPU is done with buffer b
        self.stats["t_wait"] += _time.perf_counter() - t0
        self.free.append(b)

    def pump(self, n=1, block=False):
        """read up to n next groups (fixed order) into free buffers; block=True may wait for the GPU to finish the
        oldest group still reading a buffer (in the MoE), block=False only uses free buffers (during attention)"""
        while n > 0 and self.nxt < self.total:
            if not self.free:
                if not (block and self.pending):
                    return
                self._reclaim()
            b = self.free.pop(0)
            i, g = divmod(self.nxt, self.ng)
            t = self.bms[i]["tier"]
            ids = list(range(g * self.G, min(self.E, (g + 1) * self.G)))
            t0 = _time.perf_counter()
            t["ext"].read_rebuilt(t["path"], t["recsz"], t["msb_sz"], t["pers4"], ids, list(range(len(ids))), self.bufs[b])
            self.stats["t_read"] += _time.perf_counter() - t0; self.stats["reads"] += 1
            self.ready[(i, g)] = b
            self.nxt += 1; n -= 1

    def get(self, L, g):
        """buffer holding layer L's group g, reading (and waiting for the GPU) as needed"""
        key = (self.L0[L], g)
        while key not in self.ready:
            if not self.free:
                self._reclaim()
            self.pump(1)
        return self.ready.pop(key)

    def release(self, b, users):
        self.pending.append((b, users))


def start_stream(bms, R=4, G=64):
    global STREAM
    STREAM = _Stream(bms, R, G)
    return STREAM


def stop_stream():
    global STREAM
    s, STREAM = STREAM, None
    if s is not None:
        while s.pending:
            s._reclaim()
        for b, futs, n in getattr(s, "inflight", {}).values():   # reads submitted past the last layer: let them land
            for f in futs:
                f.result()
        if hasattr(s, "pool"):
            s.pool.shutdown(wait=True)
        del s.bufs
    mx.clear_cache()
    return s.stats if s is not None else {}


def _prefill_streamed(cls, self, x, indices, ids, bm, st):
    from .stream import PIECES
    k = ids.shape[-1]
    lead = x.shape[:-1]
    flat = ids.reshape(-1)
    x2 = x.reshape(-1, x.shape[-1])
    G, E = st.G, st.E
    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    outs, poss = [], []
    try:
        for g in range(st.ng):
            lo, hi = g * G, min(E, (g + 1) * G)
            b = st.get(bm["L"], g)
            pos = np.nonzero((flat >= lo) & (flat < hi))[0]
            if len(pos) == 0:
                st.release(b, []); st.stats["groups_empty"] += 1
                continue
            st.stats["groups_used"] += 1
            if hasattr(st, "scratch"):                           # v3: raw records3 -> exact 4-bit on the GPU
                scr = st.scratch(bm, b, hi - lo)
            else:
                scr = st.bufs[b] if hi - lo == G else [arr[:hi - lo] for arr in st.bufs[b]]
            for (proj, field), arr in zip(PIECES, scr):
                setattr(getattr(self, proj), field, arr)
            xp = mx.take(x2, mx.array((pos // k).astype(np.int32)), axis=0)
            cpn = (flat[pos] - lo).astype(np.int32)
            pad = (-len(pos)) % 64
            if pad:
                xp = mx.concatenate([xp, mx.zeros((pad, xp.shape[-1]), xp.dtype)])
                cpn = np.concatenate([cpn, np.zeros(pad, np.int32)])
            cp = mx.array(cpn).reshape(-1, 1).astype(indices.dtype)
            y = cls.__call__(self, xp, cp).reshape(len(cpn), -1)[:len(pos)]
            mx.async_eval(y)
            st.release(b, [y])
            if isinstance(st, (_Stream2, _Stream3)):
                if not st.free and len(st.pending) > 1:
                    st._reclaim()                                # the oldest group is done by now: its buffer reads on
                st.pump()
            else:
                st.pump(1, block=True)                           # the SSD keeps reading while the GPU computes
            outs.append(y); poss.append(pos)
    finally:
        for (proj, field), arr in zip(PIECES, saved):
            setattr(getattr(self, proj), field, arr)
    order = np.argsort(np.concatenate(poss), kind="stable").astype(np.int32)
    y = mx.take(mx.concatenate(outs, axis=0), mx.array(order), axis=0).reshape(*lead, k, -1)
    mx.eval(y)
    while st.pending:                                            # every group's output is computed: buffers free
        st._reclaim()
    if isinstance(st, (_Stream2, _Stream3)):
        st.pump()                                                # the next layer's groups read during its attention
    return y


# D1i stream v2 (D1I_STREAM=2 in the driver): the same fixed-order groups, read from the model's own safetensors (the
# exact 4-bit experts, stacked per piece: a group of consecutive experts is ONE contiguous range per piece, 9 preads per
# group) by a thread pool with the GIL released (os.preadv), straight into the ring's scratch: no rebuild, reads run
# while the main thread builds and the GPU computes. Same bytes as read_rebuilt's rebuild (records3's converter checked
# rebuild == the original 4-bit words), so the same outputs.
_HDRS = {}


class _Stream2(_Stream):
    def __init__(self, bms, R=4, G=64, threads=8):
        super().__init__(bms, R, G)
        from .stream import read_headers, PIECES
        md = os.path.dirname(os.path.dirname(bms[0]["tier"]["path"]))
        if md not in _HDRS:
            _HDRS[md] = read_headers(md)
        hdr = _HDRS[md]
        self.src = []                                            # per layer: [(fd, offset, bytes per expert)] x 9
        for bm in bms:
            pre = f"language_model.model.layers.{bm['L']}.mlp.switch_mlp"
            row = []
            for (proj, field), (sh, dt) in zip(PIECES, bm["tier"]["shapes4"]):
                path, off, _, shape = hdr[f"{pre}.{proj}.{field}"]
                row.append((_fd(path), off, int(np.prod(shape[1:])) * (4 if dt == mx.uint32 else 2)))
            self.src.append(row)
        self.pool = _TPE(max_workers=threads)
        self.inflight = {}                                       # (i, g) -> (buffer, futures)

    def pump(self, n=99, block=False):
        """submit reads for the next groups into free buffers (returns at once)"""
        while n > 0 and self.nxt < self.total:
            if not self.free:
                if not (block and self.pending):
                    return
                self._reclaim()
            b = self.free.pop(0)
            i, g = divmod(self.nxt, self.ng)
            lo, hi = g * self.G, min(self.E, (g + 1) * self.G)
            futs = []
            for (fd, off, per), arr in zip(self.src[i], self.bufs[b]):
                mv = memoryview(arr).cast("B")[:(hi - lo) * per]
                futs.append(self.pool.submit(os.preadv, fd, [mv], off + lo * per))
            self.inflight[(i, g)] = (b, futs, (hi - lo))
            self.stats["reads"] += 1
            self.nxt += 1; n -= 1

    def get(self, L, g):
        key = (self.L0[L], g)
        while key not in self.inflight:
            if not self.free:
                self._reclaim()
            self.pump(1)
        b, futs, n = self.inflight.pop(key)
        t0 = _time.perf_counter()
        for f, (fd, off, per) in zip(futs, self.src[key[0]]):
            if f.result() != n * per:
                raise IOError("short safetensors read")
        self.stats["t_read"] += _time.perf_counter() - t0          # time spent WAITING for reads
        return b


def start_stream2(bms, R=4, G=64, threads=8):
    global STREAM
    STREAM = _Stream2(bms, R, G, threads)
    STREAM.pump()                                                # the first R groups start reading now
    return STREAM


# D1i stream v3 (D1I_STREAM=3): the fixed-order groups read as RAW records3 records (both parts, one pread per expert,
# GIL-free thread pool, F_NOCACHE) into the ring, then rebuilt to the exact 4-bit pieces on the GPU (rebuild4) as each
# group runs. Never touches the model's safetensors (v2 read those, and decode afterwards ran ~17 % slower).
class _Stream3(_Stream):
    def __init__(self, bms, R=4, G=64, threads=16):
        self.bms, self.G, self.E = bms, G, bms[0]["tier"]["E"]
        self.L0 = {bm["L"]: i for i, bm in enumerate(bms)}
        self.ng = -(-self.E // G)
        rs = bms[0]["tier"]["recsz"]
        self.bufs = [mx.zeros((G, rs // 4), mx.uint32) if _RB2 else mx.zeros((G, rs), mx.uint8) for _ in range(R)]
        mx.eval(*self.bufs)
        self.free, self.pending, self.ready = list(range(R)), [], {}
        self.nxt, self.total = 0, len(bms) * self.ng
        self.stats = {"reads": 0, "t_read": 0.0, "t_wait": 0.0, "groups_used": 0, "groups_empty": 0}
        self.pool = _TPE(max_workers=threads)
        self.inflight = {}

    def pump(self, n=99, block=False):
        while n > 0 and self.nxt < self.total:
            if not self.free:
                if not (block and self.pending):
                    return
                self._reclaim()
            b = self.free.pop(0)
            i, g = divmod(self.nxt, self.ng)
            t = self.bms[i]["tier"]
            fd, rs = _fd(t["path"]), t["recsz"]
            lo, hi = g * self.G, min(self.E, (g + 1) * self.G)
            mv = memoryview(self.bufs[b]).cast("B")
            futs = [self.pool.submit(os.preadv, fd, [mv[k * rs:(k + 1) * rs]], e * rs) for k, e in enumerate(range(lo, hi))]
            self.inflight[(i, g)] = (b, futs, hi - lo)
            self.stats["reads"] += 1
            self.nxt += 1; n -= 1

    def get(self, L, g):
        key = (self.L0[L], g)
        while key not in self.inflight:
            if not self.free:
                self._reclaim()
            self.pump(1)
        b, futs, n = self.inflight.pop(key)
        t0 = _time.perf_counter()
        rs = self.bms[key[0]]["tier"]["recsz"]
        for f in futs:
            if f.result() != rs:
                raise IOError("short records3 read")
        self.stats["t_read"] += _time.perf_counter() - t0
        return b

    def scratch(self, bm, b, n):
        return _rebuild_gpu2(bm["tier"], self.bufs[b], n) if _RB2 else _rebuild_gpu(bm["tier"], self.bufs[b], n)


def start_stream3(bms, R=4, G=64, threads=16):
    global STREAM
    STREAM = _Stream3(bms, R, G, threads)
    STREAM.pump()
    return STREAM


# D1i (05:00): the v3 stream's rebuild straight from the raw records (D1I_RB2=1). _rebuild_gpu sliced each piece out of
# the [G, recsz] byte buffer (a strided byte copy of ~177 MB per group) before rebuild4; this kernel reads the 3-bit and
# 1-bit words in place (record stride and piece offsets as template constants) with rebuild4's bit math, per expert.
_RB2_SRC = r"""
uint j = thread_position_in_grid.x;
uint e = j / WPE, jj = j - e * WPE;
const device uint* w3 = raw + e * RS4 + W3OFF;
const device uint* l1 = raw + e * RS4 + L1OFF;
uint bit = jj * 24u, w = bit >> 5, sh = bit & 31u;
uint x = w3[w] >> sh;
if (sh > 8u) x |= w3[w + 1] << (32u - sh);
x &= 0xFFFFFFu;
uint l = (l1[jj >> 2] >> ((jj & 3u) * 8u)) & 0xFFu;
uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
a = (a & 0x07070707u) | ((a & 0x38383838u) << 1);
uint y = l; y = (y | (y << 12)) & 0x000F000Fu; y = (y | (y << 6)) & 0x03030303u; y = (y | (y << 3)) & 0x11111111u;
out[j] = (a << 1) | y;
"""
_rb2 = mx.fast.metal_kernel(name="d1i_rebuild4_strided", input_names=["raw"], output_names=["out"], source=_RB2_SRC)
_RB2 = os.environ.get("D1I_RB2") == "1"


def _rebuild_gpu2(t, raw32, n):
    """[n] raw records3 records in a [G, recsz/4] uint32 buffer -> the 9 exact 4-bit pieces (PIECES order)"""
    r = t["r"]
    mo, ms, lo, ls = r["msb_piece_off"], r["msb_piece_size"], r["lsb_piece_off"], r["lsb_piece_size"]
    rs4 = t["recsz"] // 4
    rows = raw32[:n]
    out = []
    for p in range(3):
        (wsh, _), (ssh, _), (bsh, _) = t["shapes4"][3 * p:3 * p + 3]
        wpe = int(np.prod(wsh))                                 # 4-bit words per expert of this projection
        w4 = _rb2(inputs=[raw32], template=[("RS4", rs4), ("W3OFF", mo[3 * p] // 4), ("L1OFF", lo[2 * p] // 4), ("WPE", wpe)],
                  output_shapes=[(n * wpe,)], output_dtypes=[mx.uint32], grid=(n * wpe, 1, 1), threadgroup=(256, 1, 1))[0]
        s3 = rows[:, mo[3 * p + 1] // 4:(mo[3 * p + 1] + ms[3 * p + 1]) // 4].view(mx.bfloat16).reshape(n, *ssh)
        b = rows[:, lo[2 * p + 1] // 4:(lo[2 * p + 1] + ls[2 * p + 1]) // 4].view(mx.bfloat16).reshape(n, *bsh)
        out += [w4.reshape(n, *wsh), s3 * _HALF, b]
    return out
