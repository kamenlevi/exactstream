"""cand_comp.py (Breakthrough, Tue 17 Nov 2026 11:3x; MOE BLOCK): component arms for bench.py. Each times ONE piece of
today's block alone (no overlap with the rest), so the block's time can be split. Not candidates.
  c_route  router logits + router_fused + t1cf + the slot lookups
  c_shared the shared expert (3 MLX quantized matmuls + silu * mul) and its sigmoid gate
  c_exp    the routed one-call (knob tgl16eu), on routing precomputed outside the timed region
  c_comb   combine_phase + the shared add, on precomputed inputs
  c_t1     Leap's t1cf alone (the 1c tiers), on router outputs precomputed outside the timed region. Added Tue 12:4x:
           the 1c tiers were not adopted (EvalSuite V2 FAIL, 12:3x), so the engine's glue does not run it"""
import moeblk as MB

mx = MB.mx
_CACHE = {}


_RC = {}


def reset():
    _CACHE.clear()
    _RC.clear()


def _pre_r(lay, x):
    k = (id(lay), id(x))                                   # as _pre: bench.py holds every x alive and reset()s per pass
    if k not in _RC:
        ids, scores = MB.route(lay, x)
        mx.eval(ids, scores)
        _RC[k] = (ids, scores)
    return _RC[k]


def c_t1(lay, x):
    ids, scores = _pre_r(lay, x)
    return list(MB.glue(lay, ids, scores))


def _pre(lay, x):
    k = (id(lay), id(x))                                   # safe: bench.py holds every block's x alive and reset()s per pass
    if k not in _CACHE:
        ids, scores = MB.route(lay, x)
        sc, idn, nl = MB.glue(lay, ids, scores)
        pm, pl = MB.slots(lay, idn, nl)
        sh = MB.shared_expert(lay, x)
        import bigmoe.tier as BT
        c = lay.c
        y, _, _ = MB.CQ.tier_glu_nax6n_1cq_coh(x, lay.base, lay.side, pm, pl, idn.astype(mx.int32), c["routed_store"]["plan"],
                                               lay.region, lay.offs, lay.status)
        mx.eval(sc, idn, nl, pm, pl, sh, y)
        _CACHE[k] = (sc, idn, nl, pm, pl, sh, y)
    return _CACHE[k]


def c_route(lay, x):
    ids, scores = MB.route(lay, x)
    sc, idn, nl = MB.glue(lay, ids, scores)
    pm, pl = MB.slots(lay, idn, nl)
    return [sc, pm, pl]


def c_shared(lay, x):
    return MB.shared_expert(lay, x)


def c_exp(lay, x):
    sc, idn, nl, pm, pl, sh, y = _pre(lay, x)
    c = lay.c
    return MB.CQ.tier_glu_nax6n_1cq_coh(x, lay.base, lay.side, pm, pl, idn.astype(mx.int32), c["routed_store"]["plan"],
                                        lay.region, lay.offs, lay.status)[0]


def c_comb(lay, x):
    import bigmoe.tier as BT
    sc, idn, nl, pm, pl, sh, y = _pre(lay, x)
    M, K = x.shape[0], lay.c["top_k"]
    TG = BT._tier_glu_mod()
    yc = TG.combine_phase(y.reshape(M * K, -1), y.reshape(M * K, -1), mx.ones((M, K), mx.int32), sc.reshape(-1, K))
    return yc + sh
