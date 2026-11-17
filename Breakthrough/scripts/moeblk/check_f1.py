"""check_f1.py (Breakthrough, Tue 17 Nov 2026 11:4x; MOE BLOCK candidate F1): correctness before any timing.
Per layer (20, 44), real blocks at 32 and 16 rows:
  1. the routed rows of msh's two kernels == tgl16eu's kernels on the same slots and grouping, BITWISE (the routed path
     must not change);
  2. the shared rows vs MLX's shared expert (down(silu(gate) * up), the same weights): rel L2 per row; and both vs an
     fp32 dequant reference of the MLX 4-bit weights (h in fp32);
  3. the whole block (block_f1) vs today's chain: rel L2 per row (FAST: the upgrade's pairs + the shared numerics);
  4. 3 repeats of block_f1 bitwise identical.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/moeblk/check_f1.py [--nb 2]   -> data/moeblk/check_f1.txt"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moeblk as MB                                         # noqa: E402
import msh                                                  # noqa: E402
import bench as BN                                          # noqa: E402

mx = MB.mx
OUTD = os.path.join(HERE, "..", "..", "data", "moeblk")


def deq(W, name, bits=4, gs=64):
    w = np.array(W[name + ".weight"]).astype(np.uint32)
    O, Wd = w.shape
    per = 32 // bits
    q = np.zeros((O, Wd * per), np.float32)
    for j in range(per):
        q[:, j::per] = ((w >> np.uint32(bits * j)) & np.uint32((1 << bits) - 1)).astype(np.float32)
    s = np.array(W[name + ".scales"].astype(mx.float32))
    b = np.array(W[name + ".biases"].astype(mx.float32))
    return q * np.repeat(s, gs, axis=1) + np.repeat(b, gs, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=2)
    a = ap.parse_args()
    cfg = MB.load_cfg()
    L_ = []

    def say(s=""):
        print(s, flush=True)
        L_.append(s)
    say(f"\nCHECK F1 {time.strftime('%a %d %b %H:%M:%S')}: msh (routed tgl16eu + shared segments) vs tgl16eu / MLX / fp32 / "
        f"today's chain")
    import tgl16v as V
    import xgrp
    ok = True
    slot0 = 0
    for L in (20, 44):
        lay = MB.Layer(cfg, L, slot0)
        slot0 = lay.next_slot
        gp, up, dp = cfg["shared"]["tensors"]
        Wg, Wu, Wd = (deq(lay.W, n) for n in (gp, up, dp))
        for rows in (32, 16):
            st = dict(rbit=True, sh_mlx=[], sh_f1=[], sh_mlx_ref=[], blk=[], rep=True)
            for (f, b0, xf, _) in BN.blocks(L, a.nb, rows):
                x = mx.array(xf).astype(mx.bfloat16)
                ids, scores = MB.route(lay, x)
                sc, idn, nl = MB.glue(lay, ids, scores)
                pm, pl = MB.slots(lay, idn, nl)
                ids32 = idn.astype(mx.int32)
                grp = xgrp.gpu_grp_up2(pm, pl, ids32, cfg["routed_store"]["plan"])
                y_f1, rm1, rl1 = msh.experts_and_shared(lay, x, pm, pl, idn, grp=grp)
                y_eu, rm2, rl2 = V.tier_glu_nax6n_1cq_tgl16eu(x, lay.base, lay.side, pm, pl, ids32,
                                                              cfg["routed_store"]["plan"], lay.region, lay.offs, lay.status,
                                                              grp=grp)
                mx.eval(y_f1, y_eu, rm1, rm2, rl1, rl2)
                M, K = rows, cfg["top_k"]
                st["rbit"] &= bool(mx.all(y_f1[:M * K].view(mx.uint16) == y_eu.reshape(M * K, -1).view(mx.uint16)).item()) \
                    and bool(mx.all(rm1 == rm2).item()) and bool(mx.all(rl1 == rl2).item())
                # shared rows vs MLX (without the gate) and vs fp32
                g = lay.qmm(x, gp, 4)
                u = lay.qmm(x, up, 4)
                ys = lay.qmm((g * mx.sigmoid(g)) * u, dp, 4)
                mx.eval(ys)
                xb = np.array(x.astype(mx.float32))
                gr = xb @ Wg.T
                yr = ((gr / (1 + np.exp(-gr))) * (xb @ Wu.T)) @ Wd.T
                yf = np.array(y_f1[M * K:].astype(mx.float32))
                ym = np.array(ys.astype(mx.float32))
                for m in range(M):
                    nr = np.linalg.norm(yr[m]) + 1e-30
                    st["sh_f1"].append(np.linalg.norm(yf[m] - yr[m]) / nr)
                    st["sh_mlx_ref"].append(np.linalg.norm(ym[m] - yr[m]) / nr)
                    st["sh_mlx"].append(np.linalg.norm(yf[m] - ym[m]) / (np.linalg.norm(ym[m]) + 1e-30))
                # whole block vs today's chain; repeats
                b1 = msh.block_f1(lay, x)
                bt = MB.chain_today(lay, x)
                rp = [msh.block_f1(lay, x) for _ in range(2)]
                mx.eval(b1, bt, *rp)
                st["rep"] &= all(bool(mx.all(b1.view(mx.uint16) == r_.view(mx.uint16)).item()) for r_ in rp)
                B1, BT_ = np.array(b1.astype(mx.float32)), np.array(bt.astype(mx.float32))
                st["blk"] += [float(np.linalg.norm(B1[m] - BT_[m]) / (np.linalg.norm(BT_[m]) + 1e-30)) for m in range(M)]
            ok &= st["rbit"] and st["rep"]
            q = lambda v: f"mean {np.mean(v):.4f} max {np.max(v):.4f}"
            say(f"  L{L} {rows} rows: routed rows == tgl16eu bitwise (y, rs_m, rs_l): {st['rbit']}; repeats bitwise: {st['rep']}")
            say(f"    shared rows rel L2: F1 vs fp32 {q(st['sh_f1'])} | MLX vs fp32 {q(st['sh_mlx_ref'])} | F1 vs MLX {q(st['sh_mlx'])}")
            say(f"    whole block (F1) vs today's chain, rel L2 per row: {q(st['blk'])} (FAST: the upgrade's pairs + shared)")
    say(f"CHECK F1: routed bitwise + repeats {'PASS' if ok else 'FAIL'}; numerics above")
    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, "check_f1.txt"), "a").write("\n".join(L_) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
