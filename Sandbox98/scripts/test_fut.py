"""Toy test of the future-state input (BREAK98 Screen 1; sb/futstate.py + core.fut_in / depth1 / deeper), CPU, a tiny
random drafter, no model. Checks:
 (a) P = 0: ttt_step with the hook equals ttt_step without it: the loss bitwise, every shared gradient to rounding
     (<= 1e-6 relative); P gets a gradient
 (b) P random: ttt_step's two-pass gradient equals the one-graph autograd of ttt_loss for every parameter (rows forced
     to survive so depths past 8 run)
 (c) depth1: the true pairs' keys / values stay future-free (bitwise); the anchors' outputs change
 (d) the provider: depth j reads offset j (W[row, j - 1]); anchors without states and depths > 8 get no input
 (e) the read-out chain: P = 0 is bitwise the plain chain at every depth; the hook fires at depths 1..8 only
usage: python -I scripts/test_fut.py (from ~/Sandbox98)"""
import sys, os, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
import sb.core as C
from sb import futstate as F

rng = np.random.default_rng(1117)
d, V, S, a0, r, Dp, Kt = 64, 40, 40, 5, 12, 12, 6
E = mx.array(rng.standard_normal((V, d)).astype(np.float32))
Wm = mx.array((0.3 * rng.standard_normal((d, V))).astype(np.float32))
C.embed = lambda ids: E[ids]
C.lm_logits_train = lambda hn: hn @ Wm
C.lm_logits = lambda hn: hn.astype(mx.float32) @ Wm


def kl_hit_all(hn, tk, lp, mask, tgt):                  # every row 'hits', so chains run to Dp (the KL stays real)
    lg = C.lm_logits_train(hn)[0]
    lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
    return (C.kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), mx.stop_gradient(mx.ones(tgt.shape, mx.float32))


C._kl_hit = kl_hit_all
C._CK["fa"] = mx.checkpoint(lambda hn, tk, lp, mask, tgt: kl_hit_all(hn, mx.stop_gradient(tk), mx.stop_gradient(lp),
                                                                      mx.stop_gradient(mask), tgt))
tmp = tempfile.mkdtemp()
C.DATA = tmp
f = os.path.join(tmp, "toy", "w0.npz")
stored = np.array([a for a in range(a0, S - 1) if a not in (a0 + 3, a0 + 7)], np.int32)    # two anchors without states
Wf = rng.standard_normal((len(stored), F.K, d)).astype(np.float32)
os.makedirs(os.path.join(tmp, "fut_Z", "toy"))
mx.save_safetensors(F.fut_path("Z", f), {"anc": mx.array(stored), "W": mx.array(Wf).astype(mx.bfloat16),
                                         "top1": mx.zeros((len(stored), F.K), mx.int32)})
setup, standin, PREFIX, MLR = F.make("Z")


def drafter(seed):
    mx.random.seed(seed)
    D = C.Drafter(d=d, nh=2, nkv=1, hd=32, ffn=128, rope_dims=16)
    mx.eval(D.parameters())
    return D


ids = rng.integers(0, V, S).astype(np.int32)
tk = np.argsort(-rng.standard_normal((S - 1, V)), -1)[:, :Kt].astype(np.int32)
lp = rng.standard_normal((S - 1, Kt)).astype(np.float32); lp -= np.log(np.exp(lp).sum(-1, keepdims=True) + 0.5)
x = {"W": mx.array(rng.standard_normal((1, S, d)).astype(np.float32)).astype(mx.bfloat16), "ids": ids, "tk": tk, "lp": lp,
     "asst": np.ones(S, np.float32), "a0": a0, "r": r, "f": f}
ok = []


def flat(g):
    return dict(tree_flatten(g))


# (a) P = 0
D0 = drafter(1); D1 = drafter(1); setup(D1, None)
C.REPL["fn"] = None
l0, k0, _, g0 = C.ttt_step(D0, x, Dp)
C.REPL["fn"] = standin
l1, k1, _, g1 = C.ttt_step(D1, x, Dp)
C.REPL["fn"] = None
G0, G1 = flat(g0), flat(g1)
ok.append(("(a) P=0: loss bitwise equal", float(l0) == float(l1) and float(k0) == float(k1)))
rel0 = max(float(mx.abs(G0[k] - G1[k]).max() / (mx.abs(G0[k]).max() + 1e-12)) for k in G0)
ok.append((f"(a) P=0: every shared gradient equal to rounding (worst rel {rel0:.1e}; q now comes from its own projection call"
           f" while K / V come from the first, so sums regroup)", rel0 < 1e-6))
ok.append(("(a) P=0: P gets a gradient, norm_f none", float(mx.abs(G1["fut_fc.weight"]).max()) > 0
           and float(mx.abs(G1["fut_norm.weight"]).max()) == 0))
# (b) P random: two-pass == one-graph
D2 = drafter(2); setup(D2, None)
D2.fut_fc.weight = mx.array((0.05 * rng.standard_normal((d, d))).astype(np.float32))
D2.fut_norm.weight = mx.array((1 + 0.1 * rng.standard_normal(d)).astype(np.float32))
C.REPL["fn"] = standin; C.CUR["f"] = f
lg2, g2 = nn.value_and_grad(D2, lambda M: C.ttt_loss(M, x, Dp)[0])(D2)
mx.eval(lg2, g2)
ls, _, _, gs = C.ttt_step(D2, x, Dp)
C.REPL["fn"] = None
A_, B_ = flat(g2), flat(gs)
rel = {k: float(mx.abs(A_[k] - B_[k]).max() / (mx.abs(A_[k]).max() + 1e-12)) for k in A_}
worst = max(rel, key=rel.get)
ok.append((f"(b) two-pass == one-graph: loss {float(lg2):.6f} vs {float(ls):.6f}, worst grad rel {rel[worst]:.2e} ({worst})",
           abs(float(lg2) - float(ls)) <= 1e-5 * abs(float(lg2)) and rel[worst] < 1e-4 and set(A_) == set(B_)))
ok.append(("(b) P's gradient is nonzero", float(mx.abs(B_["fut_fc.weight"]).max()) > 0))
# (c) depth1: keys / values future-free
ida = mx.array(ids)
C.REPL["fn"] = None; hA, KA, VA = C.depth1(D2, x["W"], ida, a0)
C.REPL["fn"] = standin; C.CUR["f"] = f; hB, KB, VB = C.depth1(D2, x["W"], ida, a0)
C.REPL["fn"] = None
mx.eval(hA, KA, VA, hB, KB, VB)
dif = np.abs(np.array(hA) - np.array(hB)).max(-1)[0]          # per anchor row a0..S-2
miss = [a0 + 3 - a0, a0 + 7 - a0]
ok.append(("(c) depth1 K1 / V1 bitwise future-free", bool((np.array(KA) == np.array(KB)).all() and (np.array(VA) == np.array(VB)).all())))
ok.append(("(c) depth1 output changes at stored anchors, not at the two without states",
           bool((dif[[i for i in range(len(dif)) if i not in miss]] > 0).all()) and bool((dif[miss] == 0).all())))
# (d) the provider
C.CUR["f"] = f
anc = mx.array(np.array([stored[0], a0 + 3, stored[5]], np.int32))
o3 = standin._fut(D2, 3, anc)
ref = D2.fut_fc(D2.fut_norm(mx.array(Wf[[0, 5], 2]).astype(mx.bfloat16).astype(mx.float32)))
ok.append(("(d) depth 3 reads offset 3 (W[row, 2]); a missing anchor gets 0",
           bool(np.allclose(np.array(o3[0, [0, 2]]), np.array(ref), atol=1e-6)) and float(mx.abs(o3[0, 1]).max()) == 0))
ok.append(("(d) depth 9 gets no input (None)", standin._fut(D2, 9, anc) is None and standin._fut(D2, 8, anc) is not None))
# (e) the read-out chain
seen = []


def probe_fut(D, j, a):
    o = F_fut(D, j, a)
    if o is not None:
        seen.append(j)
    return o


F_fut = standin._fut
for Pz, tag in ((True, "P=0"), (False, "P random")):
    D3 = drafter(3); setup(D3, None)
    if not Pz:
        D3.fut_fc.weight = mx.array((0.05 * rng.standard_normal((d, d))).astype(np.float32))
    outs = {}
    for hook in (False, True):
        hs = []
        C.REPL["fn"] = None
        if hook:
            standin._fut = probe_fut
            C.REPL["fn"] = standin
        C.CUR["f"] = f
        C.chain(D3, x["W"], ida, Dp, lambda j, h, an: hs.append((j, np.array(h))), a0)
        standin._fut = F_fut; C.REPL["fn"] = None
        outs[hook] = hs
    same = [bool((a[1] == b[1]).all()) for a, b in zip(outs[False], outs[True])]
    if Pz:
        ok.append(("(e) chain with P=0 is bitwise the plain chain at every depth", all(same) and len(same) == len(outs[False])))
    else:
        ok.append(("(e) chain with P random differs from depth 1 on", not any(same)))
ok.append(("(e) the hook fired at depths 1..8 only", sorted(set(seen)) == list(range(1, 9))))
for n, v in ok:
    print(f"{'PASS' if v else 'FAIL'}  {n}")
print("TOY", "PASS" if all(v for _, v in ok) else "FAIL")
sys.exit(0 if all(v for _, v in ok) else 1)
