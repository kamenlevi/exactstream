"""THINK_1116b screen 2 (Fable FAR): ADAPTIVE BLOCK LENGTH on the real 127-node dumps.
Policy: pop the head's best-first tree to 32 nodes; if the first 32 pops reached depth >= DSURE (the head's spine stayed
sure: a near-chain), keep drafting to R2 rows (64 / 96 / 127) and check them all in one pass; else check 32.
Kept(rows) = deepest on_true node among the first `rows` pops + 1 (the bonus). Check cost: 32 rows = C32; each row past
32 costs c (0.28 ms per row at int8 dense: 0.15 dense + 0.08 expert-union growth (x1.19 for 64 rows) + 0.04 cores; band
0.19-0.45). Draft cost not included (same head either way; the extra levels are the fast head's problem).
Runs on all anchors and on walk-start anchors (the real decode's check starts)."""
import sys, numpy as np
D = "/Users/kamen/BigMoE/data/deepdraft/ceiling/"
f = sys.argv[1]
z = np.load(D + f); N = int(z["nodes"]); A = len(z["a_fi"])
fi_, an_, node, depth, on_true = (z[k] for k in ["fi","an","node","depth","on_true"])
o = np.lexsort((node, an_, fi_)); fi_, an_, node, depth, on_true = (x[o] for x in (fi_, an_, node, depth, on_true))
assert (node.reshape(A, N) == np.arange(N)).all()
depth = depth.reshape(A, N); on_true = on_true.reshape(A, N)
ws = z["a_walk_start"]
# anchor order in a_* equals the lexsorted (fi, an) order? check
ao = np.lexsort((z["a_an"], z["a_fi"])); ws = ws[ao]
def kept(rows): 
    d = np.where(on_true[:, :rows], depth[:, :rows], -1).max(1); return d + 1
def spine(rows): return depth[:, :rows].max(1)          # the head's own deepest pop (no truth)
K = {r: kept(r) for r in (32, 48, 64, 96, 127)}
C32 = 21.7
for name, m in (("all anchors", np.ones(A, bool)), ("walk starts", ws)):
    print(f"== {f} {name}: n {m.sum()}; kept 32/48/64/96/127 = " + " / ".join(f"{K[r][m].mean():.2f}" for r in (32, 48, 64, 96, 127)))
    sp = spine(32)[m]
    for dsure in (24, 28, 31):
        ext = sp >= dsure
        print(f"  spine depth in 32 pops >= {dsure}: {ext.mean():.3f} of blocks; their kept32 {K[32][m][ext].mean():.1f} vs others {K[32][m][~ext].mean():.1f}")
        for r2 in (64, 96, 127):
            kp = np.where(ext, K[r2][m], K[32][m])
            for c in (0.19, 0.28, 0.45):
                cost = C32 + ext * (r2 - 32) * c
                base = K[32][m].mean() / C32; fixed = K[r2][m].mean() / (C32 + (r2 - 32) * c)
                print(f"    extend to {r2} @ {c:.2f} ms/row: kept {kp.mean():.2f} (x{kp.mean()/K[32][m].mean():.2f}); tok/s x{(kp.mean()/cost.mean())/base:.3f} vs fixed-32, fixed-{r2} x{fixed/base:.3f}")
