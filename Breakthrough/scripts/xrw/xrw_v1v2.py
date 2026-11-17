"""xrw_v1v2.py (Breakthrough, Tue 17 Nov 2026 11:2x; EXPERT REWRITE): knob_v2's only change is the grouping kernel
(xgrp.gpu_grp_up2: the upgrade's lookup in O(1); its self-test == v1 on 240 random calls). Here the whole candidate is
compared: tgl16eu with v1's grouping vs v2's on real routing (the decision map's code blocks), layers 20 and 44,
32- and 16-row calls. y, rs_m and rs_l must be BITWISE identical, so Leap's A/B on knob_v1 carries over to v2's
numerics and only the speed changes.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/xrw_v1v2.py   (share lock, MEM ~4 GB)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrw_check as XC                                      # noqa: E402
import tgl16v                                               # noqa: E402
import xgrp                                                 # noqa: E402

mx, BG, T = XC.mx, XC.BG, XC.T


def main():
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    region = mx.zeros((128 * BG.PAIRS * 32 * (2 + BG.XLINES),), mx.int32)
    status = mx.zeros((4096,), mx.int32)
    offs = mx.zeros((3,), mx.int32)
    mx.eval(region, status, offs)
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
        for (f, b0, xf, idk, nolsb) in XC.blocks_for(Lm, 6):
            for M in (32, 16):
                x = mx.array(xf[:M]).astype(mx.bfloat16)
                pm = mx.array(sm_of[idk[:M]].astype(np.int32))
                pl = mx.array(np.where(nolsb[:M], -2, sl_of[idk[:M, :XC.R4]]).astype(np.int32))
                ids = mx.array(idk[:M].astype(np.int32))
                outs = []
                for g in (xgrp.gpu_grp_up, xgrp.gpu_grp_up2):
                    o = tgl16v.tier_glu_nax6n_1cq_tgl16eu(x, base, side, pm, pl, ids, XC.PLAN, region, offs, status,
                                                          grp=g(pm, pl, ids, XC.PLAN))
                    mx.eval(*o)
                    outs.append(o)
                same = all(XC.bits_eq(a, b) for a, b in zip(*outs))
                ok &= same
                n += 1
        print(f"  L{Lm}: tgl16eu v1 grouping == v2 grouping (y, rs_m, rs_l) on 6 blocks x (32, 16) rows: "
              f"{'all identical' if ok else 'DIFFERENT'}", flush=True)
    print(f"XRW V1 == V2: {'PASS' if ok else 'FAIL'} ({n} calls)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
