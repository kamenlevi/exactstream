"""w4a8u_check.py (Breakthrough, Wed 18 Nov 2026 07:4x; ALL_LEVERS_PLAN item 7, before BigMoE wires blob 2a): MODE
w4a8u through the FROZEN knob (default data/xrw/knob_v2, the release BigMoE wires; blob 2a = R57_XRW=w4a8u in place of
tgl16eu). Pre-registered in docs/EXPERT_REWRITE.md 'Wed 07:4x'. Every arm is the live COH one-call (ooo1cq) with the
knob ON / off, as xrw_robust.py calls it.
(1) real blocks, layers 20 / 44, 2 real 32-row blocks each (knob_check.py's blocks): ON off == ooo1cq bitwise (y, rs_m,
    rs_l); w4a8u 3 repeats identical (y, rs_m, rs_l); w4a8u's rs_m / rs_l == ooo1cq's.
    Info: per-pair rel L2 of w4a8u's y vs tgl16eu's (the same tiers in both, so it is the int8 activations' error alone).
(2) the robust shapes (xrw_robust.py's cases, the same seed 1117): M = 1, 2, 3, 5, 8, 16, 17, 24, 32 x layers 20 / 44;
    nolsb share XRW_NOLSB_P (0.3, or 0 = the stack without the 1c tiers): w4a8u 2 repeats identical (y, rs_m, rs_l);
    rs_m / rs_l == ooo1cq's; QOM-dropped pairs exactly 0; y finite.
PASS = every line of (1) and (2) True.
Usage: [XRW_NOLSB_P=0.3] ~/BigMoE/.venv/bin/python -I -B scripts/xrw/w4a8u_check.py [knob dir]   (share lock, MEM ~4 GB)"""
import os
import sys

KD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "xrw", "knob_v2"))
os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, KD)
import xknob                                                # noqa: E402
import tgl16, tgl16v, w4a8, xgrp                            # noqa: E402,F401
for m_ in (xknob, tgl16, tgl16v, w4a8, xgrp):
    assert os.path.dirname(os.path.abspath(m_.__file__)) == KD, (m_.__name__, m_.__file__)
import numpy as np                                          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrw_check as XC                                      # noqa: E402  (helpers only; the knob modules stay KD's)
assert XC.XK is xknob

mx, BG, T, CQ = XC.mx, XC.BG, XC.T, XC.CQ
NOLSB_P = float(os.environ.get("XRW_NOLSB_P", "0.3"))


def eq3(a, b):
    return XC.bits_eq(a[0], b[0]) and XC.bits_eq(a[1], b[1]) and XC.bits_eq(a[2], b[2])


def rel_pairs(a, b, npairs):
    """per-pair rel L2 of a vs b (pairs where b is all zero are skipped)"""
    fa = np.array(a.astype(mx.float32)).reshape(npairs, -1).astype(np.float64)
    fb = np.array(b.astype(mx.float32)).reshape(npairs, -1).astype(np.float64)
    nb = np.linalg.norm(fb, axis=1)
    k = nb > 0
    return np.linalg.norm(fa - fb, axis=1)[k] / nb[k]


def main():
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    mx.eval(region, status, offs)
    xknob.install()
    rng = np.random.default_rng(1117)
    ok1, ok2, n2 = True, True, 0
    rels = []
    nxt = 0
    for Lm in (20, 44):
        r = ix["layers"][str(Lm)]
        base, _ = T.make_pools(ext, nxt, 512, 512, XC.SH4, d_, r)
        Ls, side, _ = T.make_sidecar(ext, 512, 512, XC.SH4, d_, r)
        Lb = nxt
        nxt = Ls + 1
        allids = np.arange(512, dtype=np.int32)
        for j in range(0, 512, 64):
            ext.ensure(Lb, mx.array(allids[j:j + 64]))
            ext.ensure(Ls, mx.array(allids[j:j + 64]))
        sm_of = np.array(ext.lookup(Lb, allids.tolist()), np.int32)
        sl_of = np.array(ext.lookup(Ls, allids.tolist()), np.int32)
        # (1) real blocks
        for (f, b0, xf, idk, nolsb) in XC.blocks_for(Lm, 2):
            x = mx.array(xf).astype(mx.bfloat16)

            def run(on, mode=None):
                xknob.ON[0] = on
                if mode:
                    xknob.MODE[0] = mode
                return XC.call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs)
            yo = run(False)
            yu = run(True, "tgl16eu")
            yw = [run(True, "w4a8u") for _ in range(3)]
            xknob.ON[0] = False
            yo2 = run(False)
            off_ok = eq3(yo2, yo)
            rep = all(eq3(yw[0], yy) for yy in yw[1:])
            rs = XC.bits_eq(yw[0][1], yo[1]) and XC.bits_eq(yw[0][2], yo[2])
            rp = rel_pairs(yw[0][0], yu[0], idk.size)
            rels.append(rp)
            ok1 &= off_ok and rep and rs
            print(f"  (1) L{Lm} {f} rows {b0}-{b0 + 31}: ON off == ooo1cq (y, rs_m, rs_l): {off_ok}; w4a8u 3 repeats "
                  f"identical (y, rs_m, rs_l): {rep}; w4a8u rs_m / rs_l == ooo1cq's: {rs}; info: rel L2 per pair vs "
                  f"tgl16eu mean {rp.mean():.4f} max {rp.max():.4f} ({rp.size} pairs)", flush=True)
        # (2) the robust shapes (xrw_robust.py's draws, in its order)
        for M in (1, 2, 3, 5, 8, 16, 17, 24, 32):
            pool = rng.choice(512, size=max(12, 3 * M), replace=False)
            ids = np.stack([rng.choice(pool, 10, replace=False) for _ in range(M)]).astype(np.int64)
            if M >= 8:
                ids[:, 0] = pool[0]
            nolsb = rng.random((M, 2)) < NOLSB_P
            qom = set(rng.choice(pool[1:], size=2, replace=False).tolist())
            pm = np.where(np.isin(ids, list(qom)), -2, sm_of[ids]).astype(np.int32)
            pl = np.where(nolsb | np.isin(ids[:, :2], list(qom)), -2, sl_of[ids[:, :2]]).astype(np.int32)
            x = mx.array((rng.standard_normal((M, 2560)) * 0.5).astype(np.float32)).astype(mx.bfloat16)

            def run2(on, mode=None):
                xknob.ON[0] = on
                if mode:
                    xknob.MODE[0] = mode
                y, rm, rl = CQ.tier_glu_nax6n_1cq_coh(x, base, side, mx.array(pm), mx.array(pl),
                                                      mx.array(ids.astype(np.int32)), XC.PLAN, region, offs, status)
                mx.eval(y, rm, rl)
                return y, rm, rl
            yo = run2(False)
            yw1 = run2(True, "w4a8u")
            yw2 = run2(True, "w4a8u")
            xknob.ON[0] = False
            q = np.isin(ids, list(qom)).reshape(-1)
            fw = np.array(yw1[0].astype(mx.float32)).reshape(M * 10, -1)
            rep = eq3(yw1, yw2)
            rs = XC.bits_eq(yw1[1], yo[1]) and XC.bits_eq(yw1[2], yo[2])
            z = float(np.abs(fw[q]).max(initial=0.0)) == 0.0
            fin = bool(np.isfinite(fw).all())
            good = rep and rs and z and fin
            ok2 &= good
            n2 += 1
            print(f"  (2) L{Lm} M {M:2d}: w4a8u 2 repeats identical (y, rs_m, rs_l): {rep}; rs_m / rs_l == ooo1cq's: {rs}; "
                  f"QOM pairs ({int(q.sum())}) zero: {z}; y finite: {fin}", flush=True)
    rr = np.concatenate(rels)
    ok = ok1 and ok2
    print(f"W4A8U CHECK ({os.path.basename(KD)}, nolsb share {NOLSB_P}): {'PASS' if ok else 'FAIL'} ((1) real blocks "
          f"{ok1}; (2) {n2} robust cases {ok2}); info: rel L2 per pair vs tgl16eu, real blocks: mean {rr.mean():.4f} "
          f"max {rr.max():.4f} ({rr.size} pairs); calls {xknob.COUNTS}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
