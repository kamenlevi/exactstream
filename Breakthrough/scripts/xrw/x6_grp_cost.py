"""x6_grp_cost.py (Breakthrough, Tue 17 Nov 2026 11:2x; EXPERT REWRITE): what does the grouping kernel cost on the call's
critical path? ooo1cq.gpu_grp_1cq (today) vs xgrp.gpu_grp_up (the upgrade's, an extra O(P) scan per candidate pair) at
P = 160 (16 rows) and 320 (32 rows) on real routing (the decision map's code blocks, layer 20). GPU interval of each
launch (libmtlprof; one launch per command buffer), min over ROUNDS x blocks, interleaved; MLX launch overhead excluded.
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/xrw/x6_grp_cost.py [knob_dir]"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Leap/third_party/mtlprof"))
import mtlprof as MP

MP.load(2)
KD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(KD))
import numpy as np                                          # noqa: E402
import mlx.core as mx                                       # noqa: E402
import xgrp                                                 # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xrw_check as XC                                      # noqa: E402
import ooo1cq as OQ                                         # noqa: E402


def gpu_us(f):
    mx.synchronize()
    c0 = MP.counts()["ncb"]
    mx.eval(f())
    mx.synchronize()
    cb = MP.cbs(c0, MP.counts()["ncb"])
    g = [1e6 * (e - s) for s, e, rd in zip(cb["start"], cb["end"], cb["ready"]) if rd and e > s]
    return max(g) if g else float("nan")


def main():
    rng = np.random.default_rng(1)
    BL = XC.blocks_for(20, 12)
    sm_of = np.arange(512, dtype=np.int32)                  # slots = ids (grouping only reads the slot numbers)
    for M in (16, 32):
        res = {"orig": [], "up": [], "up2": []}
        for rd in range(6):
            for (f, b0, xf, idk, nolsb) in BL:
                idm = idk[:M]
                pm = mx.array(sm_of[idm].astype(np.int32))
                pl = mx.array(np.where(nolsb[:M], -2, sm_of[idm[:, :2]]).astype(np.int32))
                ids = mx.array(idm.astype(np.int32))
                mx.eval(pm, pl, ids)
                for nm in (("orig", "up", "up2") if rd % 2 == 0 else ("up2", "up", "orig")):
                    if nm == "orig":
                        res[nm].append(gpu_us(lambda: OQ.gpu_grp_1cq(pm, pl, ids, XC.PLAN)[:3]))
                    elif nm == "up":
                        res[nm].append(gpu_us(lambda: xgrp.gpu_grp_up(pm, pl, ids, XC.PLAN)[:3]))
                    else:
                        res[nm].append(gpu_us(lambda: xgrp.gpu_grp_up2(pm, pl, ids, XC.PLAN)[:3]))
        print(f"  P = {10 * M}: grouping GPU us per launch, min / p25 / median: "
              + "; ".join(f"{k} {np.min(v):.1f} / {np.percentile(v, 25):.1f} / {np.median(v):.1f}" for k, v in res.items()),
              flush=True)


if __name__ == "__main__":
    main()
