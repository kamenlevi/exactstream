"""Bit-equality test of qom_keep.py against the stored map masks (Breakthrough, Wed 2026-11-04 15:xx; the orchestrator 15:1x).
The map's SIMULATION is ported here from NewIdeas' builder (~/NewIdeas/scripts/dm_oea_masks.py masks(), read-only): the
HPD-0 walks' checks cut live (t 0.3), the rows after a check's first miss entering the block as stand-ins (row p + 97 with
their own drafts' cf), the base pool per layer as a plain LRU of C slots warmed by the prompt rows and carried across a
domain's windows in walk order. The keep-set RULE (sure rows, loads, keep, bits, the _t3 convention, the touch order) comes
only from qom_keep. Only a check's realized rows are compared (the builder masks those; prompt rows and rows outside the
walk keep all 10).
Usage (BigMoE venv): python test_qom_keep.py [arm ...]   (default: C and screen 6's arm; arms qom_0_live_{bq|ball}_c<N>[_t3])
-> per arm: windows bit-equal / 36, and the kept-expert and bit-width mix per domain."""
import json, os, re, sys
from collections import OrderedDict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DM = os.path.expanduser("~/FableGod/decision_map"); sys.path.insert(1, DM)
import qom_keep as K
import dm_gate                                        # walks + the gate's cf (read-only)
OUT = os.path.join(DM, "out"); MASKS = os.path.expanduser("~/NewIdeas/data/dm_masks")
LIVE_T = 0.3


def live_segments_nxt(k, m, A, dc):                   # NewIdeas' dm_oea_masks.live_segments_nxt, verbatim logic
    segs, a0, kk, dd = [], A, k, [float(x) for x in dc[:m]]
    while True:
        L = int((np.cumprod(dd) >= LIVE_T).sum()) if dd else 0
        if L >= kk or L == len(dd):
            segs.append((kk, L, a0, dd, dd[L] if L < len(dd) else -1.0)); return segs
        segs.append((L, L, a0, dd[:L], dd[L]))
        a0, kk, dd = a0 + L + 1, kk - L - 1, dd[L + 1:]


def touch(c, e, C):
    if e in c: c.move_to_end(e)
    else:
        c[e] = 1
        if len(c) > C: c.popitem(last=False)


def replay(arm, tail_bits=4, vec=None):
    t3 = arm.endswith("_t3"); a = arm[:-3] if t3 else arm
    mc = re.search(r"_c(\d+)$", a)
    if mc: CL = [int(mc.group(1))] * 48; a = a[:mc.start()]          # a uniform pool of N slots per layer
    else: CL = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]   # the 45 GB pool, per layer
    ball = a.endswith("_ball"); a = a[:-5] + "_bq" if ball else a
    ms = re.fullmatch(r"qom_0_split([0-9.]+)_live_bq", a); split = float(ms.group(1)) if ms else None   # Thu 11-05: SPLIT
    assert a == "qom_0_live_bq" or ms, f"{arm}: this test covers C's family (qom_0[_split<T>]_live_{{bq|ball}}[_c<N>][_t3])"
    base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load()
    eq = tot = 0; first_bad = None; mix = {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf)); cache = [OrderedDict() for _ in range(48)]
        s = mix.setdefault(dom, np.zeros(5, np.int64))          # row-layers of realized sure rows, kept 3-bit, kept 4-bit, exact row-layers, dropped
        nwin = 0
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            nwin += 1
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids)
            g = np.nan_to_num(G[nm], nan=-1.0)
            kp_all = np.ones((N, 48, 10), bool)
            for p in range(base[nm]["a"]):                      # the prompt warms the pool
                for l in range(48):
                    for e in ids[p, l].tolist(): touch(cache[l], e, CL[l])
            for (k_, m_, A_), dc_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
                for (k, m, A, dc, nxt) in live_segments_nxt(k_, m_, A_, dc_):
                    last, fm = min(A + m, N - 1), min(A + k, N - 1)
                    rows, cfs = [], []
                    for p in range(A, last + 1):
                        if p <= fm: rows.append(p); c_ = -1.0 if (k == m and p == A + m) else g[p]
                        else: rows.append((p + 97) % N); c_ = dc[p - A] if p - A < m else -1.0
                        if p - A == m and (nxt >= 0 or ball): c_ = nxt if nxt >= 0 else 1.0   # the last row: next / 17th draft
                        cfs.append(c_)
                    if not rows: continue
                    k0 = K.k0_from_cf(cfs); nreal = fm + 1 - A; st = K.strict_rows(cfs, k0, split)
                    for l in range(48):
                        idsb = ids[rows, l, :]
                        U = K.loads(idsb, k0); c = cache[l]
                        for e in U.tolist(): touch(c, e, CL[l])
                        res = np.fromiter(c.keys(), np.int64, len(c))
                        kp = K.keep(idsb, k0, U, res, st); b = K.bits(kp, k0, tail_bits=tail_bits)
                        kp_all[A:fm + 1, l] = (K.t3_mask(kp, k0) if t3 else kp)[:nreal]
                        sr = k0[:nreal] == K.KS; bb = b[:nreal]
                        s += [int(sr.sum()), int((bb[sr] == 3).sum()), int((bb[sr] == 4).sum()), int((~sr).sum()), int((bb[sr] == 0).sum())]
                        if vec is not None and nwin == 1:           # test vectors: every block-layer of the first window per domain
                            vec.append((dom, nm, l, idsb.astype(np.int32), np.array(cfs, np.float32), np.isin(idsb, res), kp, b))
                        for e in K.touch_order(idsb, k0, res, st, U): c.move_to_end(e)
            ref = np.load(os.path.join(MASKS, arm, nm))["keep"]
            ok = ref.shape == kp_all.shape and np.array_equal(ref, kp_all); eq += ok; tot += 1
            if not ok and first_bad is None:
                d = np.argwhere(ref != kp_all) if ref.shape == kp_all.shape else None
                first_bad = (nm, ref.shape, kp_all.shape, None if d is None else d[:3].tolist(), None if d is None else len(d))
    return eq, tot, first_bad, mix


VEC = os.path.join(HERE, "qom_vectors.npz"); RP = 17


def save_vectors(vec, arm):
    V = len(vec); rng = np.random.default_rng(0)
    ids = np.full((V, RP, 10), -1, np.int32); conf = np.full((V, RP), np.nan, np.float32); res = np.zeros((V, RP, 10), bool)
    valid = np.zeros((V, RP), bool); kp = np.ones((V, RP, 10), bool); bt = np.zeros((V, RP, 10), np.int8)
    lay = np.zeros(V, np.int16); dom = np.array([v[0] for v in vec]); win = np.array([v[1] for v in vec])
    for v, (_, _, l, i_, c_, r_, k_, b_) in enumerate(vec):
        R = len(c_); assert R <= RP, R
        ids[v, :R], conf[v, :R], res[v, :R], valid[v, :R], kp[v, :R], bt[v, :R], lay[v] = i_, c_, r_, True, k_, b_, l
    w = -np.sort(-rng.dirichlet(np.ones(10), size=(V, RP)), axis=-1).astype(np.float32)   # synthetic router weights, descending
    np.savez_compressed(VEC, arm=arm, domain=dom, window=win, layer=lay, ids=ids, scores=w, conf=conf, resident=res,
                        valid=valid, keep=kp, bits=bt)
    return V


def check_mx():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)                               # boolean rule: the CPU is exact and leaves the GPU alone
    z = np.load(VEC); a = {k: mx.array(z[k]) for k in ("ids", "scores", "conf", "resident", "valid")}
    kp, w, b = K.qom_keep_mx(a["ids"], a["scores"], a["conf"], a["resident"], valid=a["valid"])
    v = z["valid"][..., None]
    ok1 = np.array_equal(np.array(kp) | ~v, z["keep"] | ~v) and np.array_equal(np.where(v, np.array(b), 0), np.where(v, z["bits"], 0))
    ok2 = np.allclose(np.array(w), z["scores"] * np.array(kp), 0, 0)
    rng = np.random.default_rng(1); perm = np.argsort(rng.random(z["ids"].shape), axis=-1)   # any column order per row
    sh = lambda x: np.take_along_axis(x, perm, axis=-1); un = np.argsort(perm, axis=-1)
    kq, wq, bq = K.qom_keep_mx(mx.array(sh(z["ids"])), mx.array(sh(z["scores"])), a["conf"], mx.array(sh(z["resident"])), valid=a["valid"], ranked=False)
    ok3 = (np.array_equal(np.take_along_axis(np.array(kq), un, axis=-1) | ~v, z["keep"] | ~v)
           and np.array_equal(np.where(v, np.take_along_axis(np.array(bq), un, axis=-1), 0), np.where(v, z["bits"], 0)))
    print(f"qom_keep_mx on {z['ids'].shape[0]} test vectors (block-layers of the first window per domain, {z['arm']}): "
          f"keep + bits == the numpy rule {ok1}; weights = scores x keep {ok2}; argpartition column order (ranked=False) {ok3}", flush=True)
    ok4 = True                                                  # the split (Thu 11-05): split=T vs the numpy rule with strict rows
    for T in (0.9, 0.99):
        kq = np.array(K.qom_keep_mx(a["ids"], a["scores"], a["conf"], a["resident"], valid=a["valid"], split=T)[0]); bad_v = nst = 0
        for v in range(z["ids"].shape[0]):
            R = int(z["valid"][v].sum()); ids_ = z["ids"][v, :R].astype(np.int64); cf = z["conf"][v, :R]
            k0 = K.k0_from_cf(cf); st = K.strict_rows(cf, k0, T); nst += int(st.sum())
            exp = K.keep(ids_, k0, K.loads(ids_, k0), ids_[z["resident"][v, :R]], st)
            bad_v += not np.array_equal(kq[v, :R], exp)
        ok4 &= bad_v == 0
        print(f"  split {T}: qom_keep_mx(split) == the numpy rule with strict rows on {z['ids'].shape[0] - bad_v}/{z['ids'].shape[0]} "
              f"vectors ({nst} strict row-layers)", flush=True)
    return ok1 and ok2 and ok3 and ok4


if __name__ == "__main__":
    arms = sys.argv[1:] or ["qom_0_live_bq_c147_t3", "qom_0_live_ball_c147_t3"]
    bad = 0
    for arm in arms:
        vec = [] if arm == "qom_0_live_ball_c147_t3" else None
        eq, tot, fb, mix = replay(arm, vec=vec)
        if vec: print(f"  wrote {save_vectors(vec, arm)} test vectors to {VEC}", flush=True)
        print(f"{arm}: {eq}/{tot} windows bit-equal to ~/NewIdeas/data/dm_masks/{arm}" + ("" if eq == tot else f"; first difference {fb}"), flush=True)
        for dom, (n, b3, b4, ex, dr) in mix.items():
            print(f"  {dom}: sure row-layers {n}: kept 3-bit {b3 / max(n, 1):.2f} + 4-bit {b4 / max(n, 1):.2f} experts per row-layer, "
                  f"dropped {dr / max(n, 1):.2f}; exact row-layers {ex} (10 at 4 bits)", flush=True)
        bad += eq != tot or tot == 0
    if os.path.exists(VEC): bad += not check_mx()
    print("ALL BIT-EQUAL" if not bad else f"FAILED: {bad} check(s)", flush=True)
    sys.exit(1 if bad else 0)
