"""DAY9 / M-18 arm on the decision map (Breakthrough, Mon 2026-11-02; orchestrator 11:0x, MoonShots' lead): the gated rows'
top-2 experts served from REF2-T's 3-bit part (records3's MSB-3 record: m = q >> 1, scale 2s, bias b + s*mean(q & 1);
BigMoE's tier.py, DESIGN_3bit.md), the rest of the window exact 4-bit.
How: the block is row-independent, so a gated row gets k = 0 in the stock combine (no routed experts) and this module adds
its own routed output: sum over ranks 0-1 of the block's own normalised gate weight x the expert computed from records3
(gather_qmm bits=3 on the MSB part, as tier._proj does for a 3-bit rank). bits=4 is the CONTROL: the same path with the
4-bit expert rebuilt bit-exactly from MSB + LSB (tier.rebuild4, s = s3 / 2, b from the LSB part), so its flips separate the
path's own numerics from the 3-bit effect (compare with gate_top2_cf099, the stock top-2).
Reads: per layer, the unique experts of the call's gated rows (pread of each record's MSB part, + LSB for bits=4), with
F_NOCACHE so the page cache is left alone. The ids need the routing, so each such call syncs once (as BigMoE's prefill).
Converted records3 layers (BigMoE's PLAN_records3_convert.md: Leap's native order, + 11-bit packed scales; Kamen-approved
Tue 17:2x): each record goes back to today's row-major record through BigMoE's ext (record_rowmajor, ext/build_nat,
761cb3d), then decodes as before with today's offsets (records3_index_rowmajor.json, kept beside this file)."""
import fcntl, json, os, sys
sys.path.insert(1, os.path.expanduser("~/BigMoE"))   # bigmoe.tier (read-only import)
import numpy as np
import mlx.core as mx

MD = os.path.expanduser("~/BigMoE/models/bare")
R3 = os.path.expanduser(os.environ.get("DM_R3", os.path.join(MD, "records3")))   # DM_R3: another store (tests)
RM_IX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records3_index_rowmajor.json")   # today's offsets
LAYOUT = {"row-major": 0, "rowmajor": 0, "row_major": 0, "native": 1, "native+p11": 2}
_IX, _FD, _RM = {}, {}, {}
STATS = {"calls": 0, "rows": 0, "experts": 0, "bytes": 0}
# MSB pieces (tier._MSB order): gate w3 s3 b3, up w3 s3 b3, down w3 s3 b3; LSB (tier._LSB): gate l1 b, up l1 b, down l1 b
_SHP = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}

def _index():
    if not _IX:
        _IX.update(json.load(open(os.path.join(R3, "index.json")))["layers"])
    return _IX

def _layout(L, r):
    """0 = row-major (today), 1 = native, 2 = native + p11, from the layer's index entry: a name ("native+p11", BigMoE's
    converter) or {"order", "scales"} (the plan's form); anything else refuses"""
    lay = r.get("layout")
    if lay is None: return 0
    if isinstance(lay, dict):
        o, sc = lay.get("order", "rowmajor"), lay.get("scales", "bf16")
        lay = ({("native", "p11"): "native+p11", ("native", "bf16"): "native"}.get((o, sc))
               or ("row-major" if LAYOUT.get(o) == 0 and sc == "bf16" else repr(lay)))
    if lay not in LAYOUT: raise RuntimeError(f"records3 layer {L}: unknown layout {lay!r}; dm_t3 refuses it")
    return LAYOUT[lay]

def _ext(lay):
    import bigmoe_ext as ext                                   # the engine's ext; BIGMOE_EXT_BUILD picks the build
    if not hasattr(ext, "record_rowmajor") or ext.layout_max() < lay:
        raise RuntimeError("a converted records3 layer needs BigMoE's layout-aware ext (BIGMOE_EXT_BUILD=build_nat)")
    return ext

def _fd(L):
    if L not in _FD:
        r = _index()[str(L)]
        fd = os.open(os.path.join(R3, r["file"]), os.O_RDONLY)
        fcntl.fcntl(fd, 48, 1)                               # F_NOCACHE: don't fill the page cache with records
        _FD[L] = fd
    return _FD[L]

def read(L, ids, lsb=False):
    """Records of experts ids (unique np.int) of layer L -> {proj: {w3, s3, b3[, l1, b]}} as mx arrays with a leading [U]."""
    r0 = _index()[str(L)]; lay = _layout(L, r0); U = len(ids)
    if not _RM: _RM.update(json.load(open(RM_IX))["layers"])
    r = r0 if lay == 0 else _RM[str(L)]                      # the row-major record's sizes and offsets
    end = r["msb_sz"] + (r["lsb_sz"] if lsb else 0)
    buf = np.empty((U, end), np.uint8)
    if lay == 0:
        fd = _fd(L)
        for i, e in enumerate(ids):
            n = os.preadv(fd, [memoryview(buf[i])], int(e) * r["recsz"])
            assert n == end, (L, e, n, end)
    else:                                                     # converted: back to the row-major record first
        ext = _ext(lay); out = mx.zeros((r["recsz"],), mx.uint8); mx.eval(out)
        path, offs = os.path.join(R3, r0["file"]), list(r0["msb_piece_off"]) + list(r0["lsb_piece_off"])
        pers4 = [n for o, k in _SHP.values() for n in (o * k // 2, o * k // 64 * 2, o * k // 64 * 2)]
        for i, e in enumerate(ids):
            n = ext.record_rowmajor(path, r0["recsz"], r0["msb_sz"], pers4, lay, [_SHP[p][0] for p in _SHP], offs, int(e), out)
            assert n == r["recsz"], (L, e, n, r["recsz"])
            buf[i] = np.array(out)[:end]
    STATS["experts"] += U; STATS["bytes"] += U * end
    out = {}
    for p, proj in enumerate(("gate_proj", "up_proj", "down_proj")):
        o, k = _SHP[proj]; d = {}
        for j, (f, shp, dt) in enumerate((("w3", (o, k * 3 // 32), np.uint32), ("s3", (o, k // 64), np.uint16),
                                           ("b3", (o, k // 64), np.uint16))):
            off, sz = r["msb_piece_off"][3 * p + j], r["msb_piece_size"][3 * p + j]
            a = buf[:, off:off + sz].copy().view(dt).reshape(U, *shp)
            d[f] = mx.array(a) if dt == np.uint32 else mx.array(a).view(mx.bfloat16)
        if lsb:
            for j, (f, shp, dt) in enumerate((("l1", (o, k // 32), np.uint32), ("b", (o, k // 64), np.uint16))):
                off, sz = r["lsb_piece_off"][2 * p + j], r["lsb_piece_size"][2 * p + j]
                a = buf[:, off:off + sz].copy().view(dt).reshape(U, *shp)
                d[f] = mx.array(a) if dt == np.uint32 else mx.array(a).view(mx.bfloat16)
        out[proj] = d
    return out

def _proj(h, W, idx, bits):
    if bits == 2:   # Wed 06:3x (NewIdeas' qom_0_live_bq_t2, MoonShots' M-34): the MSB part at 2 bits, w = s3 (2 (q3 >> 1) + 0.5) + b3
        import to2bit   # decision_map/to2bit.py = ~/MoonShots/scripts/to2bit.py at c5af2d3, byte for byte
        return to2bit.proj2(h, W, idx)
    if bits == 3:
        return mx.gather_qmm(h, W["w3"], W["s3"], W["b3"], rhs_indices=idx, transpose=True, group_size=64, bits=3)
    from bigmoe import tier                                   # read-only import: BigMoE's bit-exact 4-bit rebuild
    w = tier.rebuild4(W["w3"], W["l1"]); s = W["s3"] * mx.array(0.5, dtype=W["s3"].dtype)
    return mx.gather_qmm(h, w, s, W["b"], rhs_indices=idx, transpose=True, group_size=64, bits=4)

def routed_top2(block, L, xg, bits=3):
    return routed_topk(block, L, xg, bits, 2)

def routed_topk(block, L, xg, bits=3, K=2, keep=None):
    """xg [G, D]: the gated rows' MoE input -> [G, D] = sum over ranks 0 .. K-1 of the block's normalised top-10 weight x
    the expert at `bits` (3: records3's MSB part; 4: the control, rebuilt 4-bit). K = 10 with every row gated is t3_all
    (Mon 23:3x: 45 GB's plan 3333333333, the 1-bit sidecar dropped)."""
    gates = mx.softmax(block.gate(xg), axis=-1, precise=True)            # the block's own ops (qwen4_exp._prefill_call)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    sc = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob: sc = sc / sc.sum(axis=-1, keepdims=True)
    o = mx.argsort(-sc.astype(mx.float32), axis=-1)[:, :K]
    top_ids, top_sc = mx.take_along_axis(inds, o, axis=-1), mx.take_along_axis(sc, o, axis=-1)
    if keep is not None: top_sc = top_sc * mx.array(keep[:, :K]).astype(top_sc.dtype)   # a rank mask (P1-K1 at 3 bits)
    ids_np = np.array(top_ids).astype(np.int64)                           # sync: the records to read depend on routing
    uniq, inv = np.unique(ids_np.reshape(-1), return_inverse=True)
    W = read(L, uniq, lsb=(bits == 4))
    idx = mx.array(inv.astype(np.uint32))                                 # [KG] pair -> its expert in W
    h = mx.repeat(xg, K, axis=0)[:, None, :]                              # [KG, 1, D]: row g's pairs are Kg .. Kg + K - 1
    up, gt = _proj(h, W["up_proj"], idx, bits), _proj(h, W["gate_proj"], idx, bits)
    y = _proj(block.switch_mlp.activation(up, gt), W["down_proj"], idx, bits)[:, 0, :]   # [KG, D]
    STATS["calls"] += 1; STATS["rows"] += int(xg.shape[0])
    return (y.reshape(-1, K, y.shape[-1]) * top_sc[..., None].astype(y.dtype)).sum(axis=1)

# ---- Map screen 7 (Breakthrough, Wed 2026-11-04 14:4x; NewIdeas' QUEUE 14:1x + amendment 3de406e, the orchestrator 14:3x,
# Fit12's TRL view): a per-layer weight source for the gated rows' KEPT experts. routed_kept serves every kept (row, rank)
# pair of keep [G, 10] from `src` and reads only the kept experts: src = an int (records3 at those bits, read / _proj as
# above) or "trl:<form>" (~/Fit12/scripts/trl_view.py, read-only import: the decoded trellis W = Q R^T as an MLX 8-bit affine
# view, gather_qmm bits 8). The combine is routed_topk's with K = 10 (a dropped rank weighs 0).
import struct
LSRC = {"calls": 0, "pairs": 0, "experts": 0, "trl": 0, "fallback": 0}
LSRC_GROUP = int(os.environ.get("DM_LSRC_GROUP", "128"))   # experts per read: TRL 5.22 MB each, MSB-2's unpacking ~4 B / weight
TRL_OK = {}   # (layer, form) -> bool [512]: the experts whose three TRL files trl_gate found complete and consistent
_TV = {}

def trl_view():
    if not _TV:
        p = os.path.expanduser("~/Fit12/scripts")
        if p not in sys.path: sys.path.insert(1, p)
        import trl_view as tv
        _TV["m"] = tv
    return _TV["m"]

def _st_ok(path, L, e, p, k):
    """One TRL file is whole and the one asked for: safetensors size = 8 + header + data end; w U32 [o, i / 4], s / b
    [o, i / 64]; its metadata (when present) names this layer, expert, projection and k."""
    o, i = {"gate": (640, 2560), "up": (640, 2560), "down": (2560, 640)}[p]
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        md = h.get("__metadata__", {}); end = max(v["data_offsets"][1] for x, v in h.items() if x != "__metadata__")
        return (os.path.getsize(path) == 8 + n + end and h["w"]["dtype"] == "U32" and h["w"]["shape"] == [o, i // 4]
                and h["s"]["shape"] == [o, i // 64] and h["b"]["shape"] == [o, i // 64]
                and all(md.get(x, v) == v for x, v in (("layer", str(L)), ("expert", str(e)), ("proj", p), ("k", str(k)))))
    except Exception:
        return False

def trl_gate(layers, form):
    """Refuse (RuntimeError) unless every expert of each layer (all 512: the arm's own routing is known only at run time)
    has its three TRL files whole (_st_ok). DM_TRL_FALLBACK=1: a missing or partial expert is served from records3 MSB-3
    instead, counted (LSRC['fallback']). Sets TRL_OK, which routed_kept serves from. Returns {layer: experts not served}."""
    tv = trl_view(); miss, part = {}, {}
    for L in layers:
        ok = np.zeros(512, bool)
        for e in range(512):
            fs = [(tv._f(L, e, p, k), p, k) for p, k in tv.KS[form].items()]
            if not all(os.path.exists(f) for f, _, _ in fs): continue
            if all(_st_ok(f, L, e, p, k) for f, p, k in fs): ok[e] = True
            else: part.setdefault(L, []).append(e)
        TRL_OK[(L, form)] = ok
        if not ok.all(): miss[L] = int((~ok).sum())
    if miss and os.environ.get("DM_TRL_FALLBACK") != "1":
        raise RuntimeError(f"TRL store {tv.STORE} ({form}): experts missing or partial {miss} (layer: count; partial {part}); not run")
    return miss

def _mlp(h, W, idx, proj, act):
    up, gt = proj(h, W["up_proj"], idx), proj(h, W["gate_proj"], idx)
    return proj(act(up, gt), W["down_proj"], idx)[:, 0, :]

def _kept_y(L, h, inv, uniq, src, act):
    """[P, D]: pair p's expert uniq[inv[p]] from src on h [P, 1, D], in pair order; experts in groups of LSRC_GROUP."""
    trl = isinstance(src, str)
    if trl:
        tv = trl_view(); form = src.split(":", 1)[1]
        ok = TRL_OK.get((L, form))                              # set by trl_gate (the arm's setup); else the files' presence
        have = ok[uniq] if ok is not None else np.array([tv.has(L, int(e), form) for e in uniq], bool)
    else:
        have = np.ones(len(uniq), bool)
    groups = []
    for m, s in ((have, src), (~have, 3)):                    # a missing TRL expert: records3 MSB-3 (DM_TRL_FALLBACK only)
        ix = np.flatnonzero(m)
        groups += [(ix[i:i + LSRC_GROUP], s) for i in range(0, len(ix), LSRC_GROUP)]
    parts, order = [], []
    for g, s in groups:
        sel = np.flatnonzero(np.isin(inv, g)); loc = mx.array(np.searchsorted(g, inv[sel]).astype(np.uint32))
        hs = mx.take(h, mx.array(sel.astype(np.int32)), axis=0)
        if isinstance(s, str):
            W = tv.read(L, [int(e) for e in uniq[g]], form); y = _mlp(hs, W, loc, tv.proj, act); LSRC["trl"] += len(g)
        else:
            W = read(L, uniq[g], lsb=(s == 4)); y = _mlp(hs, W, loc, lambda a, w, i, b=s: _proj(a, w, i, b), act)
            if trl: LSRC["fallback"] += len(g)
        mx.eval(y); del W                                     # free this group's weights before the next read
        parts.append(y); order.append(sel)
    sel = np.concatenate(order); back = np.empty(len(sel), np.int32); back[sel] = np.arange(len(sel), dtype=np.int32)
    return mx.take(mx.concatenate(parts, axis=0), mx.array(back), axis=0)

def routed_kept(block, L, xg, src, keep):
    """xg [G, D] (the gated rows' MoE input), keep [G, 10] (rank-sorted, True = kept) -> [G, D] = sum over the kept ranks of
    the block's normalised top-10 weight x the expert from src (routed_topk's ops; only kept experts are read)."""
    gates = mx.softmax(block.gate(xg), axis=-1, precise=True)            # the block's own ops (as routed_topk)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    sc = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob: sc = sc / sc.sum(axis=-1, keepdims=True)
    o = mx.argsort(-sc.astype(mx.float32), axis=-1)[:, :k]
    top_ids, top_sc = mx.take_along_axis(inds, o, axis=-1), mx.take_along_axis(sc, o, axis=-1)
    kp = np.asarray(keep, bool)[:, :k]
    top_sc = top_sc * mx.array(kp).astype(top_sc.dtype)
    ids_np = np.array(top_ids).astype(np.int64)                           # sync: the experts to read depend on routing
    G = int(xg.shape[0]); pr = np.flatnonzero(kp.reshape(-1))            # the kept (row, rank) pairs, row-major
    if not pr.size: return mx.zeros(xg.shape, xg.dtype)
    uniq, inv = np.unique(ids_np.reshape(-1)[pr], return_inverse=True)
    h = mx.take(mx.repeat(xg, k, axis=0), mx.array(pr.astype(np.int32)), axis=0)[:, None, :]   # [P, 1, D]
    y = _kept_y(L, h, inv.reshape(-1), uniq, src, block.switch_mlp.activation)                  # [P, D]
    pos = np.full(G * k, len(pr), np.int32); pos[pr] = np.arange(len(pr), dtype=np.int32)      # dropped: an appended zero row
    full = mx.take(mx.concatenate([y, mx.zeros((1, y.shape[-1]), y.dtype)]), mx.array(pos), axis=0)
    LSRC["calls"] += 1; LSRC["pairs"] += int(pr.size); LSRC["experts"] += int(uniq.size)
    return (full.reshape(G, k, -1) * top_sc[..., None].astype(full.dtype)).sum(axis=1)
