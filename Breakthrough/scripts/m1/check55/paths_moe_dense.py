"""paths_moe_dense.py (Breakthrough, CHECK55 lever 2): which kernels each side dispatches, counted by libmtlprof.
- K-73's chain as the engine runs it (router_fused(blk.gate(x)), shared_cat(blk, x), then moe_tail(y, g, s)) under
  ref2 (+ v32 at 32 rows);
- vs moe_dense.dense_pre(blk, x) + moe_tail.
Layer 24, rows 16 and 32, after one warm call each (the per-block packs are built once). Also prints ref2's path
counters (qmm16c / mlx16) and v32's split counter for the chain.
usage: [BigMoE venv] python -I -B paths_moe_dense.py [out stem]   (MEM < 1 GB)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlprof as P
P.load()                                             # before mlx.core
import mlx.core as mx
import test_moe_dense as T                           # its loaders, block builder and imports (ref2, v32, Leap's chain)

STEM = sys.argv[1] if len(sys.argv) > 1 else "l2_paths"
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def counted(fn):
    a = P.psos()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    d = P.pso_delta(a, P.psos())
    return out, d


def main():
    P.on(True)
    T.ref2.apply()
    T.v32.install(T.ref2)
    blk = T.block(24)
    say(f"LEVER2 PATHS: layer 24; sources {T.MD.SOURCES}")
    for rows in (16, 32):
        x = T.draws(rows, 5)["normal x1.0"]
        y = (mx.random.normal((1, rows, 2560), key=mx.random.key(3)) * 0.3).astype(mx.bfloat16)
        mx.eval(x, y)

        def chain():
            with T.attention_phase("decode_verify"):
                inds, scores = T.router_fused(blk.gate(x), blk.top_k)
                g, s = T.shared_cat(blk, x)
                return inds, scores, T.moe_tail(y, g, s, exp="precise")

        def mine():
            inds, scores, g, s = T.MD.dense_pre(blk, x)
            return inds, scores, T.moe_tail(y, g, s, exp="precise")

        for name, fn in (("K-73 chain", chain), ("moe_dense", mine)):
            counted(fn)                              # warm (packs, compiles)
            c0, v0 = dict(T.ref2.counts()), T.v32.CNT["qmm"]
            out, d = counted(fn)
            c1 = T.ref2.counts()
            dc = {k: c1[k] - c0[k] for k in ("qmm16c", "mlx16") if c1[k] != c0[k]}
            say(f"  rows {rows} {name}: {sum(d.values())} dispatches: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())))
            if name == "K-73 chain":
                say(f"    ref2 paths {dc}; v32 2 x 16 splits {T.v32.CNT['qmm'] - v0}")
            if name == "moe_dense":
                same = all(T.ndiff(a, b) == 0 for a, b in zip(out, ref_out))
                say(f"    outputs bitwise == the chain's: {same}")
            else:
                ref_out = out
    with open(os.path.join(T.OUT, STEM + ".txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
