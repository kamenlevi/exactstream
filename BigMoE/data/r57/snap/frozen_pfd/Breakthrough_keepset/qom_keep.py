"""QOM + DRAFT17 keep-set rule: C's rule on the decision map, as a standalone module (Breakthrough, Wed 2026-11-04 15:xx; the
orchestrator 15:1x, for BigMoE's tier call). numpy only; no engine files. Ported from NewIdeas' map-mask builder
(~/NewIdeas/scripts/dm_oea_masks.py, arms qom_0_live_bq_c147_t3 = C and qom_0_live_ball_c147_t3 = C + DRAFT17), whose
masks it reproduces bit for bit (test_qom_keep.py). See README.md for the interface.

Per verify block (the rows the engine computes together: the drafted rows and the last row) and per MoE layer:
  1. Sure rows. A row is sure if it verifies a draft whose drafter confidence cf >= thr (C: thr 0, so any row that verifies
     a draft). The last row verifies the 17th draft under DRAFT17 (pass that draft's cf); without one it is exact (pass -1).
  2. Loads. Every row's ranks < k0 (k0 = 2 on a sure row, 10 on an exact row): the block's union, read on a pool miss.
  3. Keep. An exact row keeps all 10 ranks. A sure row keeps ranks 0-1, and each rank 2-9 expert that is in the block's loads
     or resident in the layer's pool after those loads; the others (the ones that would miss) are dropped. The kept experts
     keep the model's own routing weights (no renormalisation).
  4. Bits (C's form): a sure row's ranks 0-1 at 3 bits (records3's MSB-3 part); its kept ranks 2-9 and every expert of an
     exact row at 4 bits. tail_bits=3 gives s7_ref3's / B''s form (every kept expert of a sure row at 3 bits).
Ids are rank-sorted (rank 0 = the router's highest weight), [R, 10] per layer."""
import numpy as np

THR, KS, K = 0.0, 2, 10          # C: QOM at cf 0, sure rows keep top-2 from the SSD path, a block row has 10 routed experts


def k0_from_cf(cf, thr=THR, ks=KS):
    """cf [R]: the confidence of the draft each row verifies (NaN or < 0 = no draft) -> k0 [R]: ks on sure rows, 10 else."""
    c = np.nan_to_num(np.asarray(cf, dtype=np.float64), nan=-1.0)
    return np.where(c >= thr, ks, K).astype(np.int64)


def loads(ids, k0):
    """ids [R, 10] (one layer), k0 [R] -> the block's loads at this layer: sorted unique ids of every row's ranks < k0."""
    ids = np.asarray(ids)
    return np.unique(np.concatenate([ids[r, :int(k0[r])] for r in range(ids.shape[0])]))


def strict_rows(cf, k0, split=None, ks=KS):
    """The split threshold (Thu 2026-11-05; Fable C's audit test 1, SPLIT90): sure rows whose draft cf >= split are STRICT
    (ranks < ks + the block's loads only, no resident keep). split None = off (every sure row keeps its residents: QOM)."""
    k0 = np.asarray(k0)
    if split is None: return np.zeros(len(k0), bool)
    c = np.nan_to_num(np.asarray(cf, dtype=np.float64), nan=-1.0)
    return (k0 == ks) & (c >= split)

def keep(ids, k0, block_loads, resident, strict=None):
    """-> keep [R, 10] bool, rank-sorted. resident = the layer's pool after the block's loads (any int iterable or array).
    strict [R] bool (strict_rows): those rows keep no resident expert beyond the block's loads."""
    ids = np.asarray(ids); kp = np.ones(ids.shape, bool)
    res = np.asarray(list(resident) if not isinstance(resident, np.ndarray) else resident)
    for r in range(ids.shape[0]):
        k = int(k0[r])
        if k < K:
            tail = ids[r, k:]
            kp[r, k:] = np.isin(tail, block_loads) | (False if strict is not None and strict[r] else np.isin(tail, res))
    return kp


def bits(kp, k0, ks=KS, top_bits=3, tail_bits=4, exact_bits=4):
    """-> bits [R, 10] int8 (0 = dropped): a sure row's ranks < ks at top_bits and its kept ranks >= ks at tail_bits; an
    exact row's experts at exact_bits. C's form: 3 / 4 / 4."""
    k0 = np.asarray(k0); b = np.where(kp, exact_bits, 0).astype(np.int8)
    sure = k0 == ks
    b[sure, :ks] = top_bits
    t = b[sure, ks:]; t[t > 0] = tail_bits; b[sure, ks:] = t
    return b


def kept_sets(ids, kp, b):
    """-> per row, (expert ids, bits) of the kept experts in rank order: what the tier call computes for that row."""
    ids = np.asarray(ids)
    return [(ids[r][kp[r]], b[r][kp[r]]) for r in range(ids.shape[0])]


def t3_mask(kp, k0, ks=KS):
    """The map's _t3 mask convention: keep without a sure row's ranks < ks (the harness serves those at 3 bits itself)."""
    m = kp.copy(); m[np.asarray(k0) == ks, :ks] = False
    return m


def touch_order(ids, k0, resident, strict=None, block_loads=None):
    """The map replay's pool touches after the block (optional for the engine): every sure row's ranks >= k0 that are
    resident, row by row in rank order (duplicates kept; an LRU applies them in this order). A strict row touches only what
    it keeps: its resident tail experts that are among the block's loads."""
    ids = np.asarray(ids); res = set(int(e) for e in resident); out = []
    U = set(int(e) for e in block_loads) if block_loads is not None else None
    for r in range(ids.shape[0]):
        k = int(k0[r])
        if k < K:
            st = strict is not None and strict[r]
            out += [int(e) for e in ids[r, k:] if int(e) in res and (not st or int(e) in U)]
    return out


def block(ids, cf, resident_after, thr=THR, ks=KS, top_bits=3, tail_bits=4, split=None):
    """One call per (block, layer) once the pool holds the block's loads: -> (k0, loads, keep, bits). Use loads(ids,
    k0_from_cf(cf)) first if the engine needs the load list before it can say what is resident. split: see strict_rows."""
    k0 = k0_from_cf(cf, thr, ks); U = loads(ids, k0); st = strict_rows(cf, k0, split, ks)
    kp = keep(ids, k0, U, resident_after, st)
    return k0, U, kp, bits(kp, k0, ks, top_bits, tail_bits)


def qom_keep_mx(ids, scores, conf, resident, thr=THR, ks=KS, top_bits=3, tail_bits=4, exact_bits=4, valid=None, ranked=True,
                split=None):
    """The same rule on MLX arrays, no host sync (for the engine's tier call). Leading batch dims are allowed.
      ids      [..., R, 10] int   the block's routed experts per row (R = the block's rows, the last row included)
      scores   [..., R, 10]       the router's normalised top-10 weights (the combine weights before the rule)
      conf     [..., R] float     the confidence of the draft each row verifies; NaN or < thr = no draft (exact row)
      resident [..., R, 10] bool  the (row, rank) expert is in the layer's pool for this block (values on ranks < k0 and
                                  on exact rows are ignored)
      valid    [..., R] bool      optional: padded rows (False) load nothing and are left all-kept
      ranked   True: columns are in rank order (rank 0 first). False: any column order (e.g. argpartition's); ranks come
               from argsort(-scores as float32), as the map's harness ranks them, and the outputs keep the input order.
    split    optional (default off): sure rows with conf >= split keep NO resident expert (ranks < ks + the block's loads
             only; Fable C's audit test 1 / SPLIT90, T = 0.9 or 0.99); every other row unchanged
    -> keep [..., R, 10] bool, weights = scores x keep (dropped ranks zeroed; no renormalisation), bits [..., R, 10] int8
       (0 = dropped; C's form: a sure row's ranks < ks at top_bits, its kept others at tail_bits, exact rows exact_bits).
    keep = rank < k0 OR the expert is among the block's loads (every row's ranks < k0) OR (resident AND the row is not
    strict)."""
    import mlx.core as mx
    if not ranked:
        o = mx.argsort(-scores.astype(mx.float32), axis=-1); inv = mx.argsort(o, axis=-1)
        g = lambda a: mx.take_along_axis(a, o, axis=-1)
        kp, _, b = qom_keep_mx(g(ids), g(scores), conf, g(resident), thr, ks, top_bits, tail_bits, exact_bits, valid, True, split)
        kp, b = mx.take_along_axis(kp, inv, axis=-1), mx.take_along_axis(b, inv, axis=-1)
        return kp, scores * kp.astype(scores.dtype), b
    sure = conf >= thr                                         # NaN compares False: no draft = an exact row
    if valid is not None: sure = mx.logical_and(sure, valid)
    rank = mx.arange(K)
    lo = rank < mx.where(sure, ks, K)[..., None]               # [..., R, 10]: ranks < k0 (the block's loads)
    lu = lo if valid is None else mx.logical_and(lo, valid[..., None])
    inU = mx.logical_and(ids[..., :, :, None, None] == ids[..., None, None, :, :], lu[..., None, None, :, :]).any(axis=(-2, -1))
    res = resident if split is None else mx.logical_and(resident, mx.logical_not(mx.logical_and(sure, conf >= split))[..., None])
    kp = mx.logical_or(mx.logical_or(lo, inU), res)
    b = mx.where(kp, mx.where(sure[..., None], mx.where(rank < ks, top_bits, tail_bits), exact_bits), 0).astype(mx.int8)
    return kp, scores * kp.astype(scores.dtype), b
