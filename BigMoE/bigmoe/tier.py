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
# R-44 b (NewIdeas 10-29, non-exact, behind a switch): BIGMOE_POM=R2 = MoonShots' M-03 precision on miss (a base-pool miss
# used only at ranks >= R2 reads records3p's top two code planes; a later rank < R2 use upgrades it; needs BIGMOE_M01);
# BIGMOE_POM_LP1=1 = M-07's LP1 (a rank-1 use whose LSB sidecar is not resident runs at tier 3; that miss reads nothing)
_POM = int(os.environ.get("BIGMOE_POM", "0") or 0)
_LP1 = os.environ.get("BIGMOE_POM_LP1") == "1"
_POM_IX = {}
# records3 converted in place (docs/PLAN_records3_convert.md): index.json's per-layer "layout" (absent = row-major). A
# converted layer's tier calls need a kernel router that reads its layout (Leap's REF2 native_3c: 3c-N, + p11 dequant for
# native+p11); it sets LAYOUT_ROUTER and routes each call by _LAYOUT[0] (tier_call sets it) or t["layout"] (OOO).
LAYOUTS = {"row-major": 0, "native": 1, "native+p11": 2, "native+p11+pad": 2}   # pad: 4 KB-aligned pieces (zero-copy reads)
LAYOUT_ROUTER = False
_LAYOUT = [0]
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
    if _LP1:
        _lp1_patch()
    return tier_glu


def _lp1_patch():
    """BIGMOE_POM_LP1: a pair at a 4-bit rank whose sidecar slot is < 0 (the loader read no LSB: -2) runs at tier 3. A
    runtime test in every tierglu kernel source before its first compile; tier stays uniform per threadgroup (one pair)."""
    import tierglu as TG
    if getattr(TG, "_bigmoe_lp1", False):
        return
    old = "const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);"
    new = ("int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);\n"
           "    if (tier == 4 && lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))] < 0) tier = 3;   // LP1")
    for name in ("_A", "_B", "_BC", "_AS", "_BS", "_AP", "_BP"):
        src = getattr(TG, name)
        assert src.count(old) == 1, name
        setattr(TG, name, src.replace(old, new))
    for f in (TG._ka, TG._kb, TG._kbc, TG._kas, TG._kbs, TG._kap, TG._kbp):
        f.cache_clear()
    TG._bigmoe_lp1 = True


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
        dep_l, n_l = post(t["side"], ids[..., :R4], after=dep_m)   # JOINT: posted right after the base row
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
    _LAYOUT[0] = lay = t.get("layout", 0)
    if lay and (not LAYOUT_ROUTER or _GLU != "3c"):
        raise RuntimeError(f"tier layer {bm['L']}: records3 layout {lay} needs TIER_GLU 3c with a layout router (REF2 "
                           "native_3c); BigMoE's own kernels read row-major records only")
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
def load_index(model_dir, layout_ok=False):
    # BIGMOE_M01=1 (R-44 a): records3m/, the same records with the per-group fields re-packed (the ext unpacks them)
    d = os.path.join(model_dir, "records3m" if os.environ.get("BIGMOE_M01") else "records3")
    ix = json.load(open(os.path.join(d, "index.json")))
    lays = sorted({r.get("layout", "row-major") for r in ix["layers"].values()})
    if any(x not in LAYOUTS for x in lays):
        raise RuntimeError(f"{d}: unknown records3 layout in {lays}")
    if lays != ["row-major"] and not layout_ok:
        raise RuntimeError(f"{d} holds converted layers ({lays}); this reader takes row-major records only "
                           "(docs/PLAN_records3_convert.md)")
    return d, ix


def _shapes(pers4_shapes, r=None):
    """4-bit piece shapes [(out, K/8), (out, K/64), (out, K/64)] x 3 -> MSB-3 and LSB-1 pool piece shapes. A native+p11
    layer (index entry r) holds its s3 / b3 / b pieces as packed uint32 words of the index's piece sizes."""
    msb, lsb = [], []
    shp = [sh if isinstance(sh[0], int) else sh[0] for sh in pers4_shapes]   # shapes or (shape, dtype) pairs
    p11 = r is not None and r.get("layout") in ("native+p11", "native+p11+pad")
    for p in range(3):
        (o, w8), (_, g), _ = shp[3 * p:3 * p + 3]
        sm = [((r["msb_piece_size"][3 * p + i] // 4,), mx.uint32) for i in (1, 2)] if p11 else [((o, g), mx.bfloat16)] * 2
        sl = ((r["lsb_piece_size"][2 * p + 1] // 4,), mx.uint32) if p11 else ((o, g), mx.bfloat16)
        msb += [((o, w8 * 3 // 4), mx.uint32), *sm]
        lsb += [((o, w8 // 4), mx.uint32), sl]
    return msb, lsb


def _part_order(r, part, n):
    """the record order of a part's pieces (the index lists them per projection; native+p11+pad stores the LSB part's l1
    planes first): the ext's layer takes its pieces in this order, so its sequential copy / direct preadv matches the file"""
    off = r.get(f"{part}_piece_off")
    return sorted(range(n), key=lambda i: off[i]) if off is not None else list(range(n))


def _check_part(r, pers, part):
    """the ext copies a record part piece by piece into the slots: the pool pieces must be the index's, back to back"""
    off, size = r.get(f"{part}_piece_off"), r.get(f"{part}_piece_size")
    if off is None:
        return
    start = 0 if part == "msb" else r["msb_sz"]
    o = _part_order(r, part, len(off))
    assert list(size) == pers and off[o[0]] == start and all(off[o[i + 1]] == off[o[i]] + size[o[i]] for i in range(len(o) - 1)), \
        (part, pers, off, size)


def _nbytes(sh, dt): return int(np.prod(sh)) * (4 if dt == mx.uint32 else 2)


def make_pools(ext, L, E, C_m, shapes4, rec_dir, r):
    """Base pool for layer L as ext layer L (must be the next add_layer index)."""
    if os.environ.get("BIGMOE_M01"):   # an ext without the M-01 unpack would read records3m's packed fields as raw bf16
        assert getattr(ext, "m01_on", lambda: False)(), "BIGMOE_M01=1 needs an ext built from 6753d83 or later"
    msb, _ = _shapes(shapes4, r)
    P = [mx.zeros((C_m, *sh), dt) for sh, dt in msb]
    mx.eval(*P)
    pers = [_nbytes(sh, dt) for sh, dt in msb]
    assert sum(pers) <= r.get("r3_msb_sz", r["msb_sz"]), (L, sum(pers), r["msb_sz"])
    if r.get("layout", "row-major") != "row-major":
        assert not os.environ.get("BIGMOE_M01") and not _POM, "M-01 / POM read row-major records only"
        _check_part(r, pers, "msb")
    assert ext.add_layer(C_m, E, [0] * 9, [0] * 9, pers, P) == L
    ext.set_records_part(L, os.path.join(rec_dir, r["file"]), r["recsz"], 0, r["msb_sz"])
    if _POM:                                       # R-44 b: records3p's seg1 / seg2 beside the records3m part
        assert os.environ.get("BIGMOE_M01") and os.environ.get("BIGMOE_PF_JIT"), "BIGMOE_POM needs BIGMOE_M01 + BIGMOE_PF_JIT"
        dp = os.path.join(os.path.dirname(rec_dir), "records3p")
        if not _POM_IX:
            _POM_IX.update(json.load(open(os.path.join(dp, "index.json"))))
        rp = _POM_IX["layers"][str(L)]
        assert rp["verified"] and rp["r3_msb_sz"] == r.get("r3_msb_sz"), (L, rp)
        ext.set_pom(L, os.path.join(dp, rp["file"]), rp["recsz"], rp["seg1"], rp["seg2"], _POM, len(PLAN))
    return {p: {f: P[i] for i, (pp, f) in enumerate(_MSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}, P


def make_sidecar(ext, E, C_l, shapes4, rec_dir, r):
    _, lsb = _shapes(shapes4, r)
    P = [mx.zeros((C_l, *sh), dt) for sh, dt in lsb]
    mx.eval(*P)
    pers = [_nbytes(sh, dt) for sh, dt in lsb]
    if r.get("layout", "row-major") != "row-major":
        _check_part(r, pers, "lsb")
    o = _part_order(r, "lsb", 6)                     # the ext layer's pieces in record order; the name map below stays
    Ls = ext.add_layer(C_l, E, [0] * 6, [0] * 6, [pers[i] for i in o], [P[i] for i in o])
    ext.set_records_part(Ls, os.path.join(rec_dir, r["file"]), r["recsz"], r["msb_sz"], r["recsz"] - r["msb_sz"])
    if r.get("layout") == "native+p11+pad":           # BIGMOE_RECORDS_DIRECT: the 3 aligned l1 planes straight into their
        assert hasattr(ext, "set_direct_split"), "native+p11+pad needs an ext from the lever-1 build"   # slots, the b's via a bounce
        ext.set_direct_split(Ls, 3)
    if _LP1:                                       # R-44 b LP1: rows of the 4-bit ranks; rank 1 = position 1 of each row
        ext.set_lp1(Ls, PLAN.count("4"))
    return Ls, {p: {f: P[i] for i, (pp, f) in enumerate(_LSB) if pp == p} for p in ("gate_proj", "up_proj", "down_proj")}, P


# ---------------------------------------------------------------- exact prefill in plan mode
def _read_rebuilt(t, ids, rows, dst):
    if t.get("layout", 0):
        t["ext"].read_rebuilt_layout(t["path"], t["recsz"], t["msb_sz"], t["pers4"], ids, rows, dst, t["layout"],
                                     t["rows3"], t["offs"])
    else:
        t["ext"].read_rebuilt(t["path"], t["recsz"], t["msb_sz"], t["pers4"], ids, rows, dst)



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
    _reuse_drop()                       # R-52: this call's own scratch must not sit beside the kept group buffers
    scr = [mx.zeros((len(uniq), *sh), dt) for sh, dt in t["shapes4"]]
    mx.eval(*scr)
    _read_rebuilt(t, uniq.astype(np.int32).tolist(), list(range(len(uniq))), scr)
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


def prefill_done():                     # every decode call: drop R-52's kept prefill buffers once, before the decode peak
    if _REUSE:
        _reuse_drop()


# R-52 SCRATCH24 (BIGMOE_PF_REUSE=1; NewIdeas 10-29 after Leap's R-50: a fresh ~330 MB MLX scratch costs ~8 ms in
# allocation / page faults): the grouped prefill's two G-expert double buffers stay allocated across layer calls instead
# of mx.zeros + eval + clear_cache in every call; groups use [:n] views as they already do for short groups. =2 also
# skips the per-call clear_cache (the call's temporaries are reused from MLX's cache). Everything goes at prefill_done.
_PF_REUSE = int(os.environ.get("BIGMOE_PF_REUSE", "1") or 0)   # DEFAULT 1 since NewIdeas 11-01 (R-52P: official peak 22.97 <= 23.4 GB); =0 restores
_REUSE = {}


def _reuse_drop():
    if _REUSE:
        _REUSE.clear()
        mx.clear_cache()                                     # the dropped buffers leave MLX's cache too


def _group_bufs(t, G, nb):
    if not _PF_REUSE:
        bufs = [[mx.zeros((G, *sh), dt) for sh, dt in t["shapes4"]] for _ in range(nb)]
        mx.eval(*[arr for b_ in bufs for arr in b_])
        return bufs
    cur = _REUSE.get("bufs")
    if cur is None or cur[0][0].shape[0] < G or len(cur) < nb:
        cap = max(G, cur[0][0].shape[0] if cur is not None else 0)
        _reuse_drop()                                        # free the smaller set before the larger one exists
        cur = [[mx.zeros((cap, *sh), dt) for sh, dt in t["shapes4"]] for _ in range(max(nb, 2))]
        mx.eval(*[arr for b_ in cur for arr in b_])
        _REUSE["bufs"] = cur
        STATS["pf_reuse_alloc"] = STATS.get("pf_reuse_alloc", 0) + 1
    return cur[:nb]


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
    bufs = _group_bufs(t, G, nb)                                 # R-52: fresh, or kept across calls (BIGMOE_PF_REUSE)
    cap = bufs[0][0].shape[0]

    def fill(i):
        grp = groups[i]
        _read_rebuilt(t, grp.astype(np.int32).tolist(), list(range(len(grp))), bufs[i % nb])

    saved = [getattr(getattr(self, proj), field) for proj, field in PIECES]
    outs, prev = [], []
    fill(0)
    try:
        for i, (grp, pos) in enumerate(zip(groups, poss)):
            scr = bufs[i % nb] if len(grp) == cap else [arr[:len(grp)] for arr in bufs[i % nb]]
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
    if _PF_REUSE < 2:
        mx.clear_cache()
    return y


# ---------------------------------------------------------------- MoE block: rank-ordered top-k on streamed tier layers
def install_block():
    """Decode/verify rows on tier layers (<= 8; <= 16 on rank plans, the R-55 gate's 16-row checks): the top-k sorted by
    score (descending), as planes125's evaluated variant, so ranks are positions; shared expert first with Scout's spin
    dependency edge when that patch is loaded."""
    import mtplx.models.qwen4_exp as Q
    from mtplx.attention_context import current_attention_phase
    from mtplx.attention_math import attention_gate
    import sys
    SF = next((m for m in list(sys.modules.values())       # Scout bundle's sharedfirst, if loaded (by file, not name)
               if getattr(m, "__file__", "") and m.__file__.endswith("Scout/patches/sharedfirst.py")), None)
    orig = Q.SparseMoeBlock.__call__
    from .stream import GATE_N                              # the streamer's sync-free width (MAXK; a screen may lower it)

    def _call(self, x):
        bm = getattr(self.switch_mlp, "_bm", None)
        # rank plans need the sort at every gated width: MTPLX's stock block keeps argpartition order, so a 9-16-row
        # check would tier experts by position (Leap R-69D, 11-01); uniform plans keep the 8-row path (no sort there).
        # TIER_COMBINE expects tier_call's combined [..., D], but a call with more ids than the streamer's sync-free width
        # goes to the sync prefill path, which returns per-expert [..., K, D]: since 11-01 a 9-16-row call under the 24 GB
        # stack's MAXK 80 (a 12-token prompt) crashed here (W1B 24 GB, Wed 4 Nov 07:42). Such calls take the stock block,
        # as every 9-16-row call did before 11-01; calls that worked are unchanged.
        n_ids = x.size // x.shape[-1] * self.top_k if x.ndim >= 2 else 0
        if (bm is None or "tier" not in bm or x.ndim < 2 or x.shape[-2] > (8 if _UNIFORM else 16)
                or Q._moe_prefill_combine_applies(self, x)
                or getattr(self, "sharding_group", None) is not None
                or (_COMBINE and _GLU == "3c"
                    and not (n_ids <= GATE_N[0] and (n_ids <= 92 or n_ids <= bm.get("C", 0))))):
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
