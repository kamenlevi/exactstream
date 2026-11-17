"""x4_tier_budget.py (Breakthrough, Tue 17 Nov 2026 10:3x; EXPERT REWRITE, context for the FAST error): how far apart are
an expert's outputs at tier 3 (MSB-3 + b3, what the lossy check runs for ranks 2-9) and tier 4 (MSB + LSB, ranks 0-1),
for the same row? That is the per-expert error the lossy path already accepts, against which w4a8's ~3.4 % (int8
activations) is read. fp32 dequant of the pools (xrw_check.Deq), real MoE inputs (the decision map's code windows),
layers 20 / 44, h kept in fp32. CPU only (no kernels run; the pools are loaded through BigMoE's loader like xrw_check).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x4_tier_budget.py [--pairs 400]  -> data/xrw/x4_tier_budget.txt"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xrw_check as XC                                      # noqa: E402

mx, BG, T = XC.mx, XC.BG, XC.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    a = ap.parse_args()
    ext = BG.ext
    d_, ix = T.load_index(XC.BARE, layout_ok=True)
    rng = np.random.default_rng(1117)
    lines = [f"X4 TIER BUDGET {time.strftime('%a %d %b %H:%M:%S')}: rel L2 between an expert's tier-3 and tier-4 outputs "
             f"(fp32 dequant, h in fp32) on real rows; {a.pairs} (row, expert) samples per layer"]
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
        dq = XC.Deq(base, side)
        BL = XC.blocks_for(Lm, 6)
        X = np.concatenate([b[2] for b in BL], 0).astype(np.float32)
        ID = np.concatenate([b[3] for b in BL], 0)
        X = np.array(mx.array(X).astype(mx.bfloat16).astype(mx.float32))
        rel, rel_r = [], {0: [], 1: [], 2: []}
        cache = {}
        for s in range(a.pairs):
            m = int(rng.integers(0, X.shape[0]))
            rk = int(rng.integers(0, 10))
            e = int(ID[m, rk])
            for tier in (3, 4):
                if (e, tier) not in cache:
                    cache[(e, tier)] = tuple(dq.proj(p, int(sm_of[e]), int(sl_of[e]), tier) for p in XC.PROJ)
            ys = {}
            for tier in (3, 4):
                Wg, Wu, Wd = cache[(e, tier)]
                g_ = Wg @ X[m]
                ys[tier] = Wd @ ((g_ / (1.0 + np.exp(-g_))) * (Wu @ X[m]))
            v = float(np.linalg.norm(ys[3] - ys[4]) / (np.linalg.norm(ys[4]) + 1e-30))
            rel.append(v)
            rel_r[min(rk, 2)].append(v)
            if len(cache) > 120:
                cache.clear()
        rel = np.array(rel)
        lines.append(f"  L{Lm}: tier 3 vs tier 4, rel L2: mean {rel.mean():.4f}, p50 {np.median(rel):.4f}, p90 "
                     f"{np.percentile(rel, 90):.4f}, max {rel.max():.4f}; by rank 0 / 1 / 2-9: "
                     + " / ".join(f"{np.mean(rel_r[k]):.4f}" for k in (0, 1, 2)))
        del base, side
    print("\n".join(lines), flush=True)
    open(os.path.join(XC.OUTD, "x4_tier_budget.txt"), "a").write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main()
