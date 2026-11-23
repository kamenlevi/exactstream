"""Toy test of the tree-aware mask (CPU, tiny sizes, no model): sb/core.py's _kl_hit_rank against _kl_hit and a numpy
reference, and ttt_step's keep rule for k = 1 / 2 / 4. usage: python -I scripts/test_treemask.py (from ~/Sandbox98)"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
import sb.core as C

rng = np.random.default_rng(1116)
R, d, V, Kt = 64, 16, 300, 8
Wm = mx.array(rng.standard_normal((d, V)).astype(np.float32))
C.lm_logits_train = lambda hn: (hn @ Wm)            # stand-in for the 2B's lm_head: [1, R, d] -> [1, R, V]
hn = mx.array(rng.standard_normal((1, R, d)).astype(np.float32))
lg = np.array((hn @ Wm)[0])
tgt_np = np.where(rng.random(R) < 0.5, lg.argmax(-1), rng.integers(0, V, R))   # half the rows hit
tgt_np[:6] = np.argsort(-lg[:6], -1)[np.arange(6), [1, 1, 2, 3, 4, 9]]          # known ranks 1, 1, 2, 3, 4, 9
tgt = mx.array(tgt_np.astype(np.int32))
tk = mx.array(np.argsort(-lg, -1)[:, :Kt].astype(np.int32))
lp_np = rng.standard_normal((R, Kt)).astype(np.float32); lp_np -= np.log(np.exp(lp_np).sum(-1, keepdims=True))
lp = mx.array(lp_np)
mask = mx.array((rng.random(R) < 0.8).astype(np.float32))
(k1, h1), g1 = mx.value_and_grad(C._kl_hit)(hn, tk, lp, mask, tgt)
(k2, h2, rk), g2 = mx.value_and_grad(C._kl_hit_rank)(hn, tk, lp, mask, tgt)
mx.eval(k1, h1, g1, k2, h2, rk, g2)
ref = (lg > lg[np.arange(R), tgt_np][:, None]).sum(-1)
ok = []
ok.append(("kl equal", float(k1) == float(k2)))
ok.append(("hit equal", bool((np.array(h1) == np.array(h2)).all())))
ok.append(("grad equal", bool((np.array(g1) == np.array(g2)).all())))
ok.append(("rank == numpy reference", bool((np.array(rk) == ref).all())))
ok.append(("known ranks 1 1 2 3 4 9", np.array(rk)[:6].tolist() == [1, 1, 2, 3, 4, 9]))
ok.append(("rank 0 <=> hit (no ties here)", bool(((np.array(rk) == 0) == (np.array(h2) > 0)).all())))
# keep rule as written in ttt_step
hit_np, rkn = np.array(h2) > 0, np.array(rk)
def keep(tmk):
    near = (rkn < tmk) if tmk > 1 else np.zeros(R, bool)
    return np.flatnonzero(hit_np | near)
ok.append(("k1 keeps exactly the hits (ctl)", keep(1).tolist() == np.flatnonzero(hit_np).tolist()))
ok.append(("k2 adds the rank-1 misses only", keep(2).tolist() == np.flatnonzero(hit_np | (rkn == 1)).tolist() and 0 in keep(2) and 2 not in keep(2)))
ok.append(("k4 adds ranks 1-3, drops 4 and 9", set(keep(4).tolist()) >= {0, 1, 2, 3} and 4 not in keep(4) and 5 not in keep(4)))
for n, v in ok:
    print(f"{'PASS' if v else 'FAIL'}  {n}")
print("TOY", "PASS" if all(v for _, v in ok) else "FAIL")
sys.exit(0 if all(v for _, v in ok) else 1)
