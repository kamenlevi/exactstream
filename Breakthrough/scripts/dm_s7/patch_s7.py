"""Map screens 6 + 7 harness patch (Breakthrough, Wed 2026-11-04 14:4x; the orchestrator 14:3x ROUTE; NewIdeas' QUEUE.md
14:0x / 14:1x + amendment 3de406e; Fit12's TRL view ~/Fit12/scripts/trl_view.py).
Usage: python patch_s7.py <decision_map dir> <out dir>  -> writes patched dm_t3.py, dm_common.py, dm_ablations.py into
<out dir> (anchored replacements: each anchor must occur exactly once; nothing is written on any failure).
What it adds:
- dm_t3.routed_kept(block, L, xg, src, keep): the gated rows' KEPT (row, rank) pairs only, each expert from `src` = an int
  (records3 at those bits via read / _proj, as routed_topk) or "trl:<form>" (Fit12's decoded trellis view, gather_qmm bits 8),
  experts in groups of DM_LSRC_GROUP (default 128) so the transient memory stays bounded; the combine is routed_topk's with
  K = 10 (dropped ranks weigh 0). trl_gate(layers, form) refuses unless all 512 experts of each layer are in the store with
  uniform file sizes (DM_TRL_FALLBACK=1: a missing expert falls back to records3 MSB-3, counted in LSRC['fallback']).
- dm_common: PERT['t3L'] = {layer: src}: at those layers the gated rows post nothing on the stock path (_row_rm) and _t3_add
  serves every kept expert (ranks < t3k forced in, since the _t3 masks drop them, + the mask's kept ranks t3k-9) through
  routed_kept; other layers are unchanged (tbits_mask's t3mask form). measure() logs and stores per-window counts (res 'lsrc').
- dm_ablations: qom_0_live_ball_c147_t3 (screen 6) and s7_ref / s7_trl / s7_cal / s7_cal_all (screen 7's 7-ref / 7-trl /
  7-cal / 7-cal-all on qom_0_live_ball_c239_t3, S = {7, 18, 30, 41}, the TRL mix)."""
import os, sys

DM, OUTD = sys.argv[1], sys.argv[2]


def rep(s, old, new, name):
    n = s.count(old)
    if n != 1:
        sys.exit(f"patch_s7: {name}: anchor found {n} times, want 1:\n{old}")
    return s.replace(old, new)


# ---- dm_t3.py: append routed_kept and the TRL reader ----
t3 = open(os.path.join(DM, "dm_t3.py")).read()
if "def routed_kept" in t3:
    sys.exit("patch_s7: dm_t3.py already has routed_kept")
t3 = rep(t3, "    return (y.reshape(-1, K, y.shape[-1]) * top_sc[..., None].astype(y.dtype)).sum(axis=1)\n",
         "    return (y.reshape(-1, K, y.shape[-1]) * top_sc[..., None].astype(y.dtype)).sum(axis=1)\n" + '''
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
''', "dm_t3 tail")

# ---- dm_common.py ----
cm = open(os.path.join(DM, "dm_common.py")).read()
cm = rep(cm, '    if PERT.get("t3keep") and CUR.get("k") is not None:   # rows served from records3 (dm_t3) post nothing on the stock path\n',
         '    if (PERT.get("t3keep") or li in PERT.get("t3L", ())) and CUR.get("k") is not None:   # rows served from records3 (dm_t3) post nothing on the stock path; screen 7: the gated rows at PERT["t3L"]\'s layers too\n',
         "dm_common _row_rm")
cm = rep(cm, '''    keep = None
    if PERT.get("t3keep"):                                # the gated rows' own rank mask at this layer (rank-sorted)
        rm = CUR["rm"]; keep = np.ones((len(G), 10), bool); gi = p0 + G; ok_ = gi < rm.shape[0]
        keep[ok_] = rm[gi[ok_], li]
    yg = dm_t3.routed_topk(block, li, rows[mx.array(G.astype(np.int32))], bits=PERT["t3"], K=int(PERT.get("t3k", 2)), keep=keep)
''', '''    keep = None; src = PERT.get("t3L", {}).get(li)        # screen 7 (Wed 14:4x): this layer's own source for the gated rows
    if PERT.get("t3keep") or src is not None:             # the gated rows' own rank mask at this layer (rank-sorted)
        rm = CUR["rm"]; keep = np.ones((len(G), 10), bool); gi = p0 + G; ok_ = gi < rm.shape[0]
        keep[ok_] = rm[gi[ok_], li]
    if src is not None:                                   # every kept expert from src; ranks < t3k forced in (the _t3 masks drop them)
        keep[:, :int(PERT.get("t3k", 2))] = True
        yg = dm_t3.routed_kept(block, li, rows[mx.array(G.astype(np.int32))], src, keep)
    else:
        yg = dm_t3.routed_topk(block, li, rows[mx.array(G.astype(np.int32))], bits=PERT["t3"], K=int(PERT.get("t3k", 2)), keep=keep)
''', "dm_common _t3_add")
cm = rep(cm, '''def measure(model, windows, name, setup):
''', '''def _lsrc():
    import dm_t3
    return dict(dm_t3.LSRC)

def measure(model, windows, name, setup):
''', "dm_common _lsrc")
cm = rep(cm, '''        tk, lp = run_window(model, ids)
        res[nm] = dict(dom=dom, a=int(a), off=int(off), tk=tk[a:].tolist(), lp=lp[a:].tolist())
        extra = ""
''', '''        l0 = _lsrc() if PERT.get("t3L") else None         # screen 7: this window's per-layer source counts
        tk, lp = run_window(model, ids)
        res[nm] = dict(dom=dom, a=int(a), off=int(off), tk=tk[a:].tolist(), lp=lp[a:].tolist())
        extra = ""
        if l0 is not None:
            d = {k_: v - l0[k_] for k_, v in _lsrc().items()}; res[nm]["lsrc"] = d
            extra += (f"; per-layer src: {d['calls']} calls, {d['pairs']} pairs, {d['experts']} experts read"
                      f" (TRL {d['trl']}, fallback {d['fallback']})")
''', "dm_common measure")

# ---- dm_ablations.py: append ----
ab = open(os.path.join(DM, "dm_ablations.py")).read()
if 'ABLATIONS["s7_trl"]' in ab or 'ABLATIONS["qom_0_live_ball_c147_t3"]' in ab:
    sys.exit("patch_s7: dm_ablations.py already has the screen 6 / 7 names")
ab = ab.rstrip("\n") + "\n" + '''ABLATIONS["qom_0_live_ball_c147_t3"] = tbits_mask(os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c147_t3"), 3, 2)   # NewIdeas screen 6 (QUEUE.md 14:0x; the orchestrator 14:3x): DRAFT17's quality at the 24 GB pool, C with every last row gated
def lsrc_mask(path, srcs, bits=3, k=2):
    """Map screen 7 (NewIdeas QUEUE.md 14:1x + amendment 3de406e; the orchestrator 14:3x): tbits_mask(path, bits, k) on every
    layer, except that at the layers of srcs {layer: src} the gated rows take EVERY kept expert (ranks < k + the mask's kept
    ranks k-9) from src and post nothing on the stock path (dm_t3.routed_kept): src = "trl:<form>" (Fit12's TRL view; refused
    unless all 512 experts of each such layer are in the store, or DM_TRL_FALLBACK=1, counted) or an int (records3 at those
    bits through dm_t3)."""
    def setup():
        PERT.update(rmask=path, gate=("t3mask",), t3=bits, t3k=k, renorm=False, t3L=dict(srcs))
        for s in sorted({v for v in srcs.values() if isinstance(v, str)}):
            import dm_t3
            miss = dm_t3.trl_gate(sorted(L for L, v in srcs.items() if v == s), s.split(":", 1)[1])
            print(f"[dm] {s}: " + (f"FALLBACK to records3 MSB-3 for missing experts {miss}" if miss else
                                   f"all 512 experts present at layers {sorted(L for L, v in srcs.items() if v == s)}"), flush=True)
    return setup
_S7M = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live_ball_c239_t3"); _S7 = (7, 18, 30, 41)   # screen 7's masks and S
ABLATIONS["s7_ref"] = tbits_mask(_S7M, 3, 2)                          # 7-ref: C + DRAFT17 at the mix's 239-slot pool (C's form)
ABLATIONS["s7_trl"] = lsrc_mask(_S7M, {L: "trl:mix" for L in _S7})    # 7-trl: at S every kept expert from the TRL mix (2.40 b)
ABLATIONS["s7_cal"] = lsrc_mask(_S7M, {L: 2 for L in _S7})            # 7-cal: at S every kept expert at MSB-2 (to2bit)
ABLATIONS["s7_cal_all"] = lsrc_mask(_S7M, {L: 2 for L in range(48)})  # 7-cal-all: the same at all 48 layers
'''

os.makedirs(OUTD, exist_ok=True)
for name, s in (("dm_t3.py", t3), ("dm_common.py", cm), ("dm_ablations.py", ab)):
    open(os.path.join(OUTD, name), "w").write(s)
print(f"patch_s7: wrote dm_t3.py, dm_common.py, dm_ablations.py into {OUTD}")
