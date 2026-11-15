"""diag_moe_dense.py (Breakthrough, CHECK55 lever 2): what bounds kernel A? ROUGH, shared Mac, NOT exact (diagnostic).
Throwaway copies of kernel A (moe_dense._LANE + _SRC_A, config rv:rr:rp:sgn) with one part changed:
- noX: x loads replaced by constants;
- noW: weight loads + dequantization replaced by the group's scale;
- noXW: both;
- fma2: every 8-FMA chain done twice (a second accumulator).
Chained calls (mx.depends), the 3 layers rotating, GPU busy from libmtlprof, rounds rotated; min / median per call.
usage: [BigMoE venv] python -I -B diag_moe_dense.py [out stem] [rounds] [calls] [rv:rr:rp:sgn]"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlprof as P
P.load()
import numpy as np
import mlx.core as mx
import test_moe_dense as T
MD = T.MD

STEM = sys.argv[1] if len(sys.argv) > 1 else "l2_diag_rough"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 96
RV, RR, RP, SGN = (int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "4:4:2:4").split(":"))
LOG = []

XL = "for (int i = 0; i < 8; i++) xf[i] = static_cast<float>(xc[i]);"
WL = "for (int r = 0; r < NR; r++) dequantize<float, 8, BITS>(wrow[r] + k0 * BITS / 8, scale[r], bias[r], w_dq[r]);"
FL = "for (int i = 0; i < 8; i++) acc += xf[i] * w_dq[r][i];\n                    res[r][v] += acc;"
VARIANTS = {
    "exact": [],
    "noX": [(XL, "for (int i = 0; i < 8; i++) xf[i] = 1.0f + 0.001f * (float)(i + v);")],
    "noW": [(WL, "for (int r = 0; r < NR; r++) for (int i = 0; i < 8; i++) w_dq[r][i] = scale[r] + 0.01f * (float)i;")],
    "noXW": [(XL, "for (int i = 0; i < 8; i++) xf[i] = 1.0f + 0.001f * (float)(i + v);"),
             (WL, "for (int r = 0; r < NR; r++) for (int i = 0; i < 8; i++) w_dq[r][i] = scale[r] + 0.01f * (float)i;")],
    "fma2": [("float acc = 0;", "float acc = 0, acc2 = 0;"),
             (FL, "for (int i = 0; i < 8; i++) { acc += xf[i] * w_dq[r][i]; acc2 += xf[i] * w_dq[r][i]; }\n"
                  "                    res[r][v] += acc + acc2 * 1e-30f;")],
}


def say(s):
    print(s, flush=True)
    LOG.append(s)


def kernel(name, edits):
    lane = MD._LANE
    for old, new in edits:
        assert lane.count(old) == 1, (name, old[:50])
        lane = lane.replace(old, new)
    return mx.fast.metal_kernel(name=f"b2_diag_{name}", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=MD.LK._q_header() + MD._SIG + lane,
                                source=MD._SRC_A)


def call(k, pk, xp):
    RM = xp.shape[0]
    (wr, sr, br), (wc, sc, bc), _ = pk
    vb = RM // RV
    slots = (512 // RR) * vb + (640 // RP) * vb + ((vb + 3) // 4) * 4
    tgs = (slots // 4 + SGN - 1) // SGN
    return k(inputs=[xp, wr, sr, br, wc, sc, bc],
             template=[("T", xp.dtype), ("RM", RM), ("RV", RV), ("RR", RR), ("RP", RP), ("SGN", SGN), ("LADDER", 0)],
             grid=(tgs * SGN * 32, 1, 1), threadgroup=(SGN * 32, 1, 1),
             output_shapes=[(RM, 512), (RM, 640), (RM, 1)], output_dtypes=[xp.dtype] * 3)


def timed(k, pks, x0):
    x = x0
    outs = []
    for i in range(NC):
        o = call(k, pks[i % len(pks)], x)
        outs += list(o)
        x = mx.depends(x0, list(o))
    outs.append(x)
    n0 = P.counts()["ncb"]
    t0 = P.now()
    mx.eval(outs)
    mx.synchronize()
    t1 = P.now()
    return P.busy(P.cbs(n0, P.counts()["ncb"]), t0, t1) / NC


def main():
    P.on(True)
    pks = [MD.packs(T.block(L)) for L in T.LAYERS]
    ks = {n: kernel(n, e) for n, e in VARIANTS.items()}
    say(f"LEVER2 DIAG (ROUGH, shared Mac, NOT exact) {time.strftime('%a %d %b %H:%M:%S')}: kernel A {RV}:{RR}:{RP}:{SGN}, "
        f"{NC} chained calls, {ROUNDS} rounds")
    for rows in (16, 32):
        x = T.draws(rows, 3)["normal x1.0"].reshape(rows, 2560)
        mx.eval(x)
        ref = call(ks["exact"], pks[0], x)
        mine = MD.dense_a(pks[0], x, rv=RV, rr=RR, rp=RP, sgn=SGN)
        assert all(T.ndiff(a, b) == 0 for a, b in zip(ref, mine)), "the exact copy must equal kernel A"
        names = list(ks)
        for n in names:
            timed(ks[n], pks, x)
        R = {n: [] for n in names}
        for r in range(ROUNDS):
            for n in names[r % len(names):] + names[:r % len(names)]:
                R[n].append(timed(ks[n], pks, x))
        say(f"  rows {rows}:")
        for n in names:
            v = np.array(R[n]) * 1e6
            say(f"    {n:6s} min {v.min():6.1f} us  median {np.median(v):6.1f} us per call")
    with open(os.path.join(T.OUT, STEM + ".txt"), "w") as fo:
        fo.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
