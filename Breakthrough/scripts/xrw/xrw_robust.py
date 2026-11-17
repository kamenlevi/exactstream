"""xrw_robust.py (Breakthrough, Tue 17 Nov 2026 11:2x; EXPERT REWRITE, before BigMoE wires the flag): the shapes and slot
states the live stack can send that the real-block checks did not, through the FROZEN knob (data/xrw/knob_v1):
row counts M = 1, 2, 3, 5, 8, 16, 17, 24, 32; random routing with a popular expert; 2-D nolsb marks; QOM drops (a whole
expert's base slot -2); layers 20 and 44. Per case, vs the live COH one-call (ooo1cq):
  - tgl16e: y, rs_m, rs_l bitwise;
  - tgl16eu: the pairs whose tier the upgrade did not change bitwise (xgrp's replica), 2 repeats identical;
  - QOM-dropped pairs: y exactly 0 in every arm.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xrw_robust.py   (share lock, MEM ~4 GB)"""
import os
import sys

KD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "xrw", "knob_v1"))
os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, KD)
import xknob                                                # noqa: E402
import tgl16, tgl16v, w4a8, xgrp                            # noqa: E402,F401
for m_ in (xknob, tgl16, tgl16v, w4a8, xgrp):
    assert os.path.dirname(os.path.abspath(m_.__file__)) == KD, (m_.__name__, m_.__file__)
import numpy as np                                          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrw_check as XC                                      # noqa: E402
assert XC.XK is xknob

mx, BG, T, CQ = XC.mx, XC.BG, XC.T, XC.CQ


def main():
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    mx.eval(region, status, offs)
    xknob.install()
    rng = np.random.default_rng(1117)
    ok, n = True, 0
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
        for M in (1, 2, 3, 5, 8, 16, 17, 24, 32):
            pool = rng.choice(512, size=max(12, 3 * M), replace=False)       # few experts: many shared keys / upgrades
            ids = np.stack([rng.choice(pool, 10, replace=False) for _ in range(M)]).astype(np.int64)
            if M >= 8:
                ids[:, 0] = pool[0]                                         # a popular expert at rank 0
            nolsb = rng.random((M, 2)) < 0.3
            qom = set(rng.choice(pool[1:], size=2, replace=False).tolist())
            pm = np.where(np.isin(ids, list(qom)), -2, sm_of[ids]).astype(np.int32)
            pl = np.where(nolsb | np.isin(ids[:, :2], list(qom)), -2, sl_of[ids[:, :2]]).astype(np.int32)
            x = mx.array((rng.standard_normal((M, 2560)) * 0.5).astype(np.float32)).astype(mx.bfloat16)

            def run(on, mode=None):
                xknob.ON[0] = on
                if mode:
                    xknob.MODE[0] = mode
                y, rm, rl = CQ.tier_glu_nax6n_1cq_coh(x, base, side, mx.array(pm), mx.array(pl),
                                                      mx.array(ids.astype(np.int32)), XC.PLAN, region, offs, status)
                mx.eval(y, rm, rl)
                return y, rm, rl
            yo = run(False)
            ye = run(True, "tgl16e")
            yu1 = run(True, "tgl16eu")
            yu2 = run(True, "tgl16eu")
            xknob.ON[0] = False
            chg = (xgrp.cpu_grp(pm, pl, ids, XC.PLAN, up=True)[2] != xgrp.cpu_grp(pm, pl, ids, XC.PLAN, up=False)[2])
            q = np.isin(ids, list(qom)).reshape(-1)
            e_ok = XC.bits_eq(ye[0], yo[0]) and XC.bits_eq(ye[1], yo[1]) and XC.bits_eq(ye[2], yo[2])
            rows_eq = np.array(mx.all(yu1[0].view(mx.uint16) == yo[0].view(mx.uint16), axis=-1)).reshape(-1)
            u_ok = bool(rows_eq[~chg].all()) and XC.bits_eq(yu1[0], yu2[0])
            z = all(float(np.abs(np.array(yy[0].astype(mx.float32)).reshape(M * 10, -1)[q]).max(initial=0.0)) == 0.0
                    for yy in (yo, ye, yu1))
            good = e_ok and u_ok and z
            ok &= good
            n += 1
            print(f"  L{Lm} M {M:2d}: tgl16e == ooo1cq (y, rs_m, rs_l): {e_ok}; tgl16eu unchanged pairs == ooo1cq + 2 repeats "
                  f"identical: {u_ok} ({int(chg.sum())} of {M * 10} upgraded); QOM pairs ({int(q.sum())}) zero in every arm: {z}",
                  flush=True)
    print(f"XRW ROBUST ({os.path.basename(KD)}): {'PASS' if ok else 'FAIL'} ({n} cases); calls {xknob.COUNTS}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
