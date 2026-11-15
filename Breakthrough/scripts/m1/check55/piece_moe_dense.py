"""piece_moe_dense.py (Breakthrough, CHECK55 lever 2): GPU time of the dense side's pieces, ROUGH (shared Mac).
- A-side: MLX's two 16-row calls (router 8-bit N 512 + R-74's concat 4-bit N 1281) vs kernel A (configs).
- B-side: router_fused + swiglu + qmm16c + the g copy (K-73's) vs kernel B.
Calls are chained with mx.depends, so each waits for the one before (latency, as in a check). The layers rotate
(0 / 24 / 47), N calls per timed graph, rounds rotate the forms; GPU busy from libmtlprof; min and median per call.
usage: [BigMoE venv] python -I -B piece_moe_dense.py [out stem] [rounds] [calls] [A configs rv:rr:rp:sgn,...]"""
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
from qmm16 import qmm16c

STEM = sys.argv[1] if len(sys.argv) > 1 else "l2_pieces_rough"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 9
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 96
ACFG = (sys.argv[4] if len(sys.argv) > 4 else "4:4:2:4,4:2:1:4,4:1:1:2").split(",")
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def timed(step, pks, x0):
    x = x0
    outs = []
    for i in range(NC):
        o = step(pks[i % len(pks)], x)
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
    blocks = [T.block(L) for L in T.LAYERS]
    pks = [T.MD.packs(b) for b in blocks]
    say(f"LEVER2 PIECES (ROUGH, shared Mac) {time.strftime('%a %d %b %H:%M:%S')}: {NC} chained calls per graph, "
        f"{ROUNDS} rounds, layers {T.LAYERS} rotating")
    for rows in (16, 32):
        x = T.draws(rows, 3)["normal x1.0"].reshape(rows, 2560)
        mx.eval(x)
        def mlx2(pk, xx):
            (wr, sr, br), (wc, sc, bc), _ = pk
            o = []
            for h in ([xx[:16], xx[16:]] if rows == 32 else [xx]):
                o += [mx.quantized_matmul(h, wr, sr, br, transpose=True, group_size=64, bits=8),
                      mx.quantized_matmul(h, wc, sc, bc, transpose=True, group_size=64, bits=4)]
            return o

        forms = {"mlx 2 calls" + (" x2 halves" if rows == 32 else ""): mlx2}
        for c in ACFG:
            parts = c.split(":")
            fa = {"a2": T.MD.dense_a2, "a3": T.MD.dense_a3, "a4": T.MD.dense_a4, "a5": T.MD.dense_a5, "a2p": T.MD.dense_a2p}.get(parts[0], T.MD.dense_a)
            kw = dict(zip(("rv", "rr", "rp", "sgn"), (int(v) for v in parts if v.isdigit())))
            forms[f"A {c}"] = (lambda fa_, kw_: lambda pk, xx: fa_(pk, xx, **kw_))(fa, kw)
        # B-side inputs from A (bitwise the chain's: unit-checked)
        ins = [T.MD.dense_a(pk, x) for pk in pks]
        ys = [mx.quantized_matmul(x if rows == 16 else x, pk[1][0], pk[1][1], pk[1][2], transpose=True, group_size=64,
                                  bits=4) for pk in pks]
        mx.eval(ins, ys)
        bi = {id(pk): (i, y) for pk, i, y in zip(pks, ins, ys)}

        def chainB(pk, xx):
            (lg, xs, g), y = bi[id(pk)]
            lg, y = mx.depends([lg, y], [xx])
            inds, scores = T.router_fused(lg[:rows], 10)
            sw = T.swiglu(y[:, :640], y[:, 640:1280])
            s = [qmm16c(sw[i:i + 16], pk[2][0], pk[2][1], pk[2][2]) for i in range(0, rows, 16)]
            gc = mx.contiguous(y[:, 1280:1281])
            return [inds, scores, gc] + s

        def kB(pk, xx):
            (lg, xs, g), _ = bi[id(pk)]
            lg, xs = mx.depends([lg, xs], [xx])
            return list(T.MD.dense_b(pk, lg, xs, rows, 10))

        forms["K-73 B-side (4" + (" + halves" if rows == 32 else "") + ")"] = chainB
        forms["B"] = kB
        names = list(forms)
        for n in names:
            timed(forms[n], pks, x)
        R = {n: [] for n in names}
        for r in range(ROUNDS):
            for n in names[r % len(names):] + names[:r % len(names)]:
                R[n].append(timed(forms[n], pks, x))
        say(f"  rows {rows}:")
        for n in names:
            v = np.array(R[n]) * 1e6
            say(f"    {n:28s} min {v.min():6.1f} us  median {np.median(v):6.1f} us per call")
    with open(os.path.join(T.OUT, STEM + ".txt"), "w") as fo:
        fo.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
