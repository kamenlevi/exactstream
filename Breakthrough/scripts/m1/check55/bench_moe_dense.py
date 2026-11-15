"""bench_moe_dense.py (Breakthrough, CHECK55 lever 2): GPU time per layer of the MoE dense side, K-73's chain vs
moe_dense (+ moe_tail on both), as a dependent chain over layers 0 / 24 / 47.
- Each layer's tail output is the next layer's x, so every dispatch waits for the one before, as in a check.
- All router outputs are evaluated too.
- NREP passes over the 3 layers per timed graph; rounds rotate the forms; medians.
- GPU busy comes from libmtlprof (the union of command buffers' GPU intervals); wall time is shown beside it.
- ROUGH on a shared Mac: the pre-registered timing line (docs/CHECK55_LEVER2.md section 4) is run in a quiet window
  only.
usage: [BigMoE venv] python -I -B bench_moe_dense.py [out stem] [rounds] [nrep] [forms: chain,mine,mine:rv:rr:rp:sgn,...]"""
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

STEM = sys.argv[1] if len(sys.argv) > 1 else "l2_bench_rough"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 16
FORMS = (sys.argv[4] if len(sys.argv) > 4 else "chain,mine").split(",")
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def step_fn(form, blk):
    if form == "chain":
        def f(x, y):
            inds, scores = T.router_fused(blk.gate(x), blk.top_k)
            g, s = T.shared_cat(blk, x)
            return inds, scores, T.moe_tail(y, g, s, exp="precise")
        return f
    kw = {}
    if form.startswith("mine:"):
        rv, rr, rp, sgn = (int(v) for v in form.split(":")[1:])
        kw = dict(rv=rv, rr=rr, rp=rp, sgn=sgn)

    def f(x, y):
        inds, scores, g, s = T.MD.dense_pre(blk, x, **kw)
        return inds, scores, T.moe_tail(y, g, s, exp="precise")
    return f


def run(form, blocks, x0, y):
    fns = [step_fn(form, b) for b in blocks]
    x = x0
    outs = []
    for _ in range(NREP):
        for f in fns:
            i, s, x = f(x, y)
            outs += [i, s]
    outs.append(x)
    n0 = P.counts()["ncb"]
    t0 = P.now()
    w0 = time.perf_counter()
    mx.eval(outs)
    mx.synchronize()
    w1 = time.perf_counter()
    t1 = P.now()
    cb = P.cbs(n0, P.counts()["ncb"])
    return P.busy(cb, t0, t1), w1 - w0, x


def main():
    P.on(True)
    T.ref2.apply()
    T.v32.install(T.ref2)
    blocks = [T.block(L) for L in T.LAYERS]
    say(f"LEVER2 BENCH (ROUGH, shared Mac) {time.strftime('%a %d %b %H:%M:%S')}: layers {T.LAYERS}, {NREP} passes per "
        f"graph, {ROUNDS} rounds; forms {FORMS}")
    with T.attention_phase("decode_verify"):
        for rows in (16, 32):
            x0 = T.draws(rows, 11)["normal x1.0"]
            y = (mx.random.normal((1, rows, 2560), key=mx.random.key(4)) * 0.3).astype(mx.bfloat16)
            mx.eval(x0, y)
            ends = {}
            for f in FORMS:                              # warm + the end state of each form (must agree bitwise)
                ends[f] = run(f, blocks, x0, y)[2]
            agree = all(T.ndiff(ends[f], ends[FORMS[0]]) == 0 for f in FORMS)
            G = {f: [] for f in FORMS}
            W = {f: [] for f in FORMS}
            for r in range(ROUNDS):
                order = FORMS[r % len(FORMS):] + FORMS[:r % len(FORMS)]
                for f in order:
                    g_, w_, _ = run(f, blocks, x0, y)
                    G[f].append(g_)
                    W[f].append(w_)
            per = NREP * len(blocks)
            med = {f: (1e3 * float(np.median(G[f])) / per, 1e3 * float(np.median(W[f])) / per) for f in FORMS}
            base = med[FORMS[0]][0]
            say(f"  rows {rows}: end states bitwise equal across forms: {agree}")
            for f in FORMS:
                say(f"    {f:18s} GPU {1e3 * med[f][0]:7.1f} us / layer (wall {1e3 * med[f][1]:7.1f}); vs {FORMS[0]}: "
                    f"{1e3 * (med[f][0] - base):+7.1f} us; rounds GPU us " + " ".join(f"{1e6 * v / per:.0f}" for v in G[f]))
    with open(os.path.join(T.OUT, STEM + ".txt"), "w") as fo:
        fo.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
