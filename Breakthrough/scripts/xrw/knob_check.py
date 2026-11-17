"""knob_check.py (Breakthrough, Tue 17 Nov 2026 10:3x; EXPERT REWRITE): the FROZEN knob release (data/xrw/knob_v1) as
Leap's c25 arm will load it: its modules are imported from knob_v1 FIRST (asserted by __file__), then for layers 20 / 44
and 2 real 32-row blocks each: ON off == the original bitwise; MODE tgl16e == ooo1cq bitwise; MODE tgl16eu: the pairs
whose tier did not change == ooo1cq bitwise and 3 repeats identical; the knob's call counts.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/knob_check.py [knob_dir]   (share lock, MEM ~4 GB)"""
import os
import sys

KD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "xrw", "knob_v1"))
os.environ["BIGMOE_MAXK"] = "320"
sys.path.insert(0, KD)
import xknob                                                # noqa: E402
import tgl16, tgl16v, w4a8, xgrp                            # noqa: E402,F401
for m in (xknob, tgl16, tgl16v, w4a8, xgrp):
    assert os.path.dirname(os.path.abspath(m.__file__)) == KD, (m.__name__, m.__file__)
import numpy as np                                          # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrw_check as XC                                      # noqa: E402  (helpers only; the knob modules stay knob_v1's)
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
    ok = True
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
        for (f, b0, xf, idk, nolsb) in XC.blocks_for(Lm, 2):
            x = mx.array(xf).astype(mx.bfloat16)

            def run(on, mode=None):
                xknob.ON[0] = on
                if mode:
                    xknob.MODE[0] = mode
                return XC.call(CQ.tier_glu_nax6n_1cq_coh, x, base, side, sm_of, sl_of, idk, nolsb, region, status, offs)
            yo = run(False)
            ye = run(True, "tgl16e")
            yu = [run(True, "tgl16eu") for _ in range(3)]
            xknob.ON[0] = False
            pm = sm_of[idk].astype(np.int64)
            pl = np.where(nolsb, -2, sl_of[idk[:, :XC.R4]]).astype(np.int64)
            chg = (xgrp.cpu_grp(pm, pl, idk, XC.PLAN, up=True)[2] != xgrp.cpu_grp(pm, pl, idk, XC.PLAN, up=False)[2])
            eq_e = XC.bits_eq(ye[0], yo[0]) and XC.bits_eq(ye[1], yo[1]) and XC.bits_eq(ye[2], yo[2])
            rows_eq = np.array(mx.all(yu[0][0].view(mx.uint16) == yo[0].view(mx.uint16), axis=-1)).reshape(-1)
            eq_u = bool(rows_eq[~chg].all())
            rep_u = all(XC.bits_eq(yu[0][0], yy[0]) for yy in yu[1:])
            ok &= eq_e and eq_u and rep_u
            print(f"  L{Lm} {f} rows {b0}-{b0 + 31}: tgl16e == ooo1cq bitwise (y, rs_m, rs_l): {eq_e}; tgl16eu unchanged "
                  f"pairs == ooo1cq: {eq_u} ({int((~chg).sum())} pairs; {int(chg.sum())} upgraded); 3 repeats identical: "
                  f"{rep_u}", flush=True)
    print(f"KNOB CHECK {KD}: {'PASS' if ok else 'FAIL'}; calls {xknob.COUNTS}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
