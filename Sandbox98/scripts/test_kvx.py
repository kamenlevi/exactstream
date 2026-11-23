"""Toy test of PREFIX EYES (BREAK98c; sb/kvx.py + core.kvx_in / block / depth1 / deeper; sb/kvxgen.py); (a)-(f) on the CPU. Checks:
 (a) o_x = 0: ttt_step with the hook equals ttt_step without it: the loss bitwise, every shared gradient to rounding;
     o_x gets a gradient, the block's q / norms none (o_x = 0 blocks them)
 (b) o_x random: ttt_step's two-pass gradient equals the one-graph autograd of ttt_loss for every parameter
 (c) depth1: the true pairs' keys / values are unchanged (bitwise); the anchors' outputs change
 (d) causality: a row with anchor s is bitwise unchanged when the trunk's K / V rows > s change, and changes when a row
     <= s does
 (e) the read-out chain: o_x = 0 is bitwise the plain chain at every depth; o_x random differs from depth 1 on; the hook
     fires at every depth
 (f) the new core with no hook is bitwise the committed one (git HEAD:sb/core.py): chain outputs and ttt_step's loss and
     gradients
 (g) REAL 2B (GPU, 64 tokens of an HE file): with the trunk's own layer input and its own q_proj / o_proj, the block
     reproduces the trunk's layer-15 and layer-19 attention outputs (RoPE, cache, gate and mask conventions), and
     kvxgen.capture's final state equals core.target_states' post (the capture path)
usage: python -I scripts/test_kvx.py [--no-real] (from ~/Sandbox98)"""
import sys, os, tempfile, subprocess, importlib.util
sys.path.insert(0, os.getcwd())
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
mx.set_default_device(mx.cpu)
import sb.core as C
from sb import kvx as X

rng = np.random.default_rng(1117)
d, V, S, a0, r, Dp, Kt, KV, hd = 64, 40, 40, 5, 12, 12, 6, 1, 32
E = mx.array(rng.standard_normal((V, d)).astype(np.float32))
Wm = mx.array((0.3 * rng.standard_normal((d, V))).astype(np.float32))


def kl_hit_all(hn, tk, lp, mask, tgt):                  # every row 'hits', so chains run to Dp (the KL stays real)
    lg = hn @ Wm
    lg = lg[0]
    lq = mx.take_along_axis(lg, tk, -1) - mx.logsumexp(lg, -1, keepdims=True)
    return (C.kl_top(lq, lp) * mask).sum() / mx.maximum(mask.sum(), 1.0), mx.stop_gradient(mx.ones(tgt.shape, mx.float32))


def patch(M):
    M.embed = lambda ids: E[ids]
    M.lm_logits_train = lambda hn: hn @ Wm
    M.lm_logits = lambda hn: hn.astype(mx.float32) @ Wm
    M._kl_hit = kl_hit_all
    M._CK["fa"] = mx.checkpoint(lambda hn, tk, lp, mask, tgt: kl_hit_all(hn, mx.stop_gradient(tk), mx.stop_gradient(lp),
                                                                          mx.stop_gradient(mask), tgt))


patch(C)
tmp = tempfile.mkdtemp()
C.DATA = tmp
f = os.path.join(tmp, "toy", "w0.npz")
Kf = (rng.standard_normal((len(X.LAYERS), KV, S, hd))).astype(np.float32)
Vf = (rng.standard_normal((len(X.LAYERS), KV, S, hd))).astype(np.float32)
os.makedirs(os.path.dirname(X.kv_path(f)))
mx.save_safetensors(X.kv_path(f), {"K": mx.array(Kf).astype(mx.bfloat16), "V": mx.array(Vf).astype(mx.bfloat16)})
ini = {}
for L in X.LAYERS:
    ini[f"n{L}"] = mx.array((1 + 0.1 * rng.standard_normal(d)).astype(np.float32))
    ini[f"q{L}"] = mx.array((0.2 * rng.standard_normal((2 * 2 * hd, d))).astype(np.float32))
    ini[f"qn{L}"] = mx.array((1 + 0.1 * rng.standard_normal(hd)).astype(np.float32))
mx.save_safetensors(X.init_path(), ini)
setup, standin, PREFIX, MLR = X.make(3e-5)


def drafter(seed):
    mx.random.seed(seed)
    D = C.Drafter(d=d, nh=2, nkv=KV, hd=hd, ffn=128, rope_dims=16)
    mx.eval(D.parameters())
    return D


def randomize_o(D, s=0.05):
    for L in X.LAYERS:
        getattr(D, f"kvx_o{L}").weight = mx.array((s * rng.standard_normal((d, 2 * hd))).astype(np.float32))


ids = rng.integers(0, V, S).astype(np.int32)
tk = np.argsort(-rng.standard_normal((S - 1, V)), -1)[:, :Kt].astype(np.int32)
lp = rng.standard_normal((S - 1, Kt)).astype(np.float32); lp -= np.log(np.exp(lp).sum(-1, keepdims=True) + 0.5)
x = {"W": mx.array(rng.standard_normal((1, S, d)).astype(np.float32)).astype(mx.bfloat16), "ids": ids, "tk": tk, "lp": lp,
     "asst": np.ones(S, np.float32), "a0": a0, "r": r, "f": f}
ok = []


def flat(g):
    return dict(tree_flatten(g))


# (a) o_x = 0
D0 = drafter(1); D1 = drafter(1); setup(D1, None)
C.REPL["fn"] = None
l0, k0, _, g0 = C.ttt_step(D0, x, Dp)
C.REPL["fn"] = standin
l1, k1, _, g1 = C.ttt_step(D1, x, Dp)
C.REPL["fn"] = None
G0, G1 = flat(g0), flat(g1)
ok.append(("(a) o_x=0: loss bitwise equal", float(l0) == float(l1) and float(k0) == float(k1)))
rel0 = max(float(mx.abs(G0[k] - G1[k]).max() / (mx.abs(G0[k]).max() + 1e-12)) for k in G0)
ok.append((f"(a) o_x=0: every shared gradient equal to rounding (worst rel {rel0:.1e})", rel0 < 1e-6))
zq = all(float(mx.abs(G1[k]).max()) == 0 for k in G1 if k.startswith("kvx_") and not k.startswith("kvx_o"))
ok.append(("(a) o_x=0: o_x gets a gradient; the block's q / q-norm / norm none",
           all(float(mx.abs(G1[f"kvx_o{L}.weight"]).max()) > 0 for L in X.LAYERS) and zq))
# (b) o_x random: two-pass == one-graph
D2 = drafter(2); setup(D2, None); randomize_o(D2)
C.REPL["fn"] = standin; C.CUR.update(f=f, x=x, train=True)
lg2, g2 = nn.value_and_grad(D2, lambda M: C.ttt_loss(M, x, Dp)[0])(D2)
mx.eval(lg2, g2)
ls, _, _, gs = C.ttt_step(D2, x, Dp)
C.REPL["fn"] = None
A_, B_ = flat(g2), flat(gs)
rel = {k: float(mx.abs(A_[k] - B_[k]).max() / (mx.abs(A_[k]).max() + 1e-12)) for k in A_}
worst = max(rel, key=rel.get)
ok.append((f"(b) two-pass == one-graph: loss {float(lg2):.6f} vs {float(ls):.6f}, worst grad rel {rel[worst]:.2e} ({worst})",
           abs(float(lg2) - float(ls)) <= 1e-5 * abs(float(lg2)) and rel[worst] < 1e-4 and set(A_) == set(B_)))
ok.append(("(b) every block parameter gets a nonzero gradient",
           all(float(mx.abs(B_[k]).max()) > 0 for k in B_ if k.startswith("kvx_"))))
# (c) depth1: keys / values unchanged
ida = mx.array(ids)
C.CUR.update(f=f, x=None, train=False)
C.REPL["fn"] = None; hA, KA, VA = C.depth1(D2, x["W"], ida, a0)
C.REPL["fn"] = standin; hB, KB, VB = C.depth1(D2, x["W"], ida, a0)
C.REPL["fn"] = None
mx.eval(hA, KA, VA, hB, KB, VB)
dif = np.abs(np.array(hA) - np.array(hB)).max(-1)[0]
ok.append(("(c) depth1 K1 / V1 bitwise unchanged", bool((np.array(KA) == np.array(KB)).all() and (np.array(VA) == np.array(VB)).all())))
ok.append(("(c) depth1 output changes at every anchor row", bool((dif > 0).all())))
# (d) causality
f2 = os.path.join(tmp, "toy", "w1.npz")
K2, V2 = Kf.copy(), Vf.copy(); cut = 21
K2[:, :, cut:] = rng.standard_normal(K2[:, :, cut:].shape); V2[:, :, cut:] = rng.standard_normal(V2[:, :, cut:].shape)
mx.save_safetensors(X.kv_path(f2), {"K": mx.array(K2).astype(mx.bfloat16), "V": mx.array(V2).astype(mx.bfloat16)})
anc = mx.array(np.array([10, 20, 21, 30], np.int32))
h = mx.array(rng.standard_normal((1, 4, d)).astype(np.float32))
C.CUR.update(f=f, x=None); o1 = np.array(standin._kvx(D2, 3, h, anc))
C.CUR.update(f=f2, x=None); o2 = np.array(standin._kvx(D2, 3, h, anc))
ok.append(("(d) rows with anchors 10 / 20 are bitwise unchanged when K / V rows >= 21 change; anchors 21 / 30 change",
           bool((o1[0, :2] == o2[0, :2]).all()) and bool((np.abs(o1[0, 2:] - o2[0, 2:]).max(-1) > 0).all())))
# (e) the read-out chain
seen = []
F_kvx = standin._kvx


def probe(D, j, hh, a):
    seen.append(j)
    return F_kvx(D, j, hh, a)


for oz, tag in ((True, "o_x=0"), (False, "o_x random")):
    D3 = drafter(3); setup(D3, None)
    if not oz:
        randomize_o(D3)
    outs = {}
    for hook in (False, True):
        hs = []
        C.REPL["fn"] = None
        if hook:
            standin._kvx = probe
            C.REPL["fn"] = standin
        C.CUR.update(f=f, x=None)
        C.chain(D3, x["W"], ida, Dp, lambda j, hh, an: hs.append((j, np.array(hh))), a0)
        standin._kvx = F_kvx; C.REPL["fn"] = None
        outs[hook] = hs
    same = [bool((p[1] == q[1]).all()) for p, q in zip(outs[False], outs[True])]
    if oz:
        ok.append(("(e) chain with o_x=0 is bitwise the plain chain at every depth", all(same) and len(same) == len(outs[False])))
    else:
        ok.append(("(e) chain with o_x random differs from depth 1 on", not any(same)))
ok.append((f"(e) the hook fired at every depth 1..{len(outs[False])}", sorted(set(seen)) == list(range(1, len(outs[False]) + 1))))
# (f) the new core with no hook == the committed core
src = subprocess.run(["git", "show", "HEAD:sb/core.py"], capture_output=True, text=True, check=True).stdout
pold = os.path.join(tmp, "core_old.py"); open(pold, "w").write(src)
spec = importlib.util.spec_from_file_location("core_old", pold); CO = importlib.util.module_from_spec(spec); spec.loader.exec_module(CO)
patch(CO); CO.DATA = tmp
DA, DB = drafter(4), drafter(4)
hs_new, hs_old = [], []
C.chain(DA, x["W"], ida, Dp, lambda j, hh, an: hs_new.append(np.array(hh)), a0)
CO.chain(DB, x["W"], ida, Dp, lambda j, hh, an: hs_old.append(np.array(hh)), a0)
lnew, _, _, gnew = C.ttt_step(DA, x, Dp); lold, _, _, gold = CO.ttt_step(DB, x, Dp)
GN, GO = flat(gnew), flat(gold)
ok.append(("(f) no hook: the new core is bitwise the committed one (chain at every depth, ttt_step loss + gradients)",
           len(hs_new) == len(hs_old) and all((p == q).all() for p, q in zip(hs_new, hs_old)) and float(lnew) == float(lold)
           and set(GN) == set(GO) and all(bool(mx.array_equal(GN[k], GO[k])) for k in GN)))
# (g) the real 2B
if "--no-real" not in sys.argv:
    # the 2B's 4-bit matmuls run a slow reference path on the CPU (> 10 min for 64 tokens, 11-17): part (g) runs on the
    # GPU (64 tokens, ~2 GB, seconds), both of its forwards on the same device
    mx.set_default_device(mx.gpu)
    from sb import kvxgen as KG
    importlib.reload(C)                                   # the real embedding / lm_head again (the toy patches undone)
    C.DATA = tmp
    m, _ = C.load_target(); t = C.tm()
    import glob
    fr = sorted(f_ for f_ in glob.glob(os.path.expanduser("~/Sandbox98/data/he/*.npz")) if not f_.endswith(".taps.npz"))[0]
    ids_r = np.asarray(np.load(fr)["ids"], np.int32)[:64]
    rec = {}
    for L in X.LAYERS:
        lay = t.layers[L]
        orig = lay.__call__

        def wrapped(xx, mask=None, cache=None, _L=L, _lay=lay):
            r_ = _lay.self_attn(_lay.input_layernorm(xx), mask, cache)
            rec[_L] = (xx, r_)
            hh = xx + r_
            return hh + _lay.mlp(_lay.post_attention_layernorm(hh))
        lay.__class__ = type("Rec" + str(L), (lay.__class__,), {"__call__": lambda self, xx, mask=None, cache=None, _w=wrapped: _w(xx, mask, cache)})
    Kr, Vr, post = KG.capture(t, ids_r)
    mx.eval(Kr, Vr, post, *[v for p in rec.values() for v in p])
    for L in X.LAYERS:
        t.layers[L].__class__ = t.layers[L].__class__.__mro__[1]
    _, post_ref, _, _ = C.target_states(ids_r, taps=(), logits=False)
    ok.append(("(g) kvxgen.capture's final state == core.target_states' post (bitwise)", bool(mx.array_equal(post, post_ref))))
    KG.write_init(t)
    Dr = C.Drafter()
    for i, L in enumerate(X.LAYERS):
        fL = os.path.join(tmp, f"real{L}", "w.npz")
        os.makedirs(os.path.dirname(X.kv_path(fL)), exist_ok=True)
        mx.save_safetensors(X.kv_path(fL), {"K": Kr[i:i + 1], "V": Vr[i:i + 1]})
        sL, stL, _, _ = X.make(3e-5, layers=(L,))
        sL(Dr, None)
        op = t.layers[L].self_attn.o_proj
        getattr(Dr, f"kvx_o{L}").weight = (mx.dequantize(op.weight, op.scales, op.biases, group_size=op.group_size, bits=op.bits)
                                           if hasattr(op, "scales") else op.weight).astype(mx.float32)
        xin, rout = rec[L]
        C.CUR.update(f=fL, x=None)
        n_ = len(ids_r)
        out = stL._kvx(Dr, 1, xin.astype(mx.float32), mx.arange(n_))
        a_, b_ = np.array(out[0]).astype(np.float64), np.array(rout[0].astype(mx.float32)).astype(np.float64)
        relerr = np.linalg.norm(a_ - b_) / np.linalg.norm(b_)
        cos = (a_ * b_).sum(-1) / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1))
        ok.append((f"(g) layer {L}: the block reproduces the trunk's attention output (rel err {relerr:.2e}, min row cos {cos.min():.5f})",
                   relerr < 0.03 and cos.min() > 0.999))
for n, v in ok:
    print(f"{'PASS' if v else 'FAIL'}  {n}")
print("TOY", "PASS" if all(v for _, v in ok) else "FAIL")
sys.exit(0 if all(v for _, v in ok) else 1)
