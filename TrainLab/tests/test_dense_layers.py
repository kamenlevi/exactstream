"""CPU tests for sticky/dense_layers.py (B10 v2 dense layer arms), tiny shapes:
1. V3Src + dense_experts == the suite's quantized path (gather_qmm on 4-bit codes, MSB-3 via evalsuite.model.tier_weights)
   per (row, rank): ranks 0-1 at 4 bits, ranks 2+ at MSB-3; and all4 = 4 bits everywhere.
2. TrSrc from a decoder in Fit12's format (both key styles) == int8 x group-scale dequantization; hi = full, lo = base.
3. install_block on a tiny real qwen4_exp.SparseMoeBlock: with a float source equal to its own SwitchGLU weights the dense
   block == the stock block; with a lo level of half the weights, ranks 2+ get the halved experts; the switch still runs.
usage: [BigMoE venv] python tests/test_dense_layers.py"""
import os, sys, types
sys.path.insert(0, os.path.expanduser("~/EvalSuite"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sticky"))
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)
from evalsuite import model as EM  # noqa: E402
import dense_layers as DL  # noqa: E402

E, I, H, K, R, GS = 8, 64, 128, 4, 3, 64
rng = np.random.default_rng(0)
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}")


# 1. V3Src vs the suite's quantized kernels
Wf = {p: mx.array(rng.normal(0, 0.05, (E,) + sh).astype(np.float32)).astype(mx.bfloat16)
      for p, sh in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I)))}
q = {p: mx.quantize(w, group_size=GS, bits=4) for p, w in Wf.items()}
src = DL.V3Src.__new__(DL.V3Src)
src.q, src.M, src.all4, src.gs = q, EM, False, GS
x = mx.array(rng.normal(0, 1, (1, R, H)).astype(np.float32)).astype(mx.bfloat16)
inds = mx.array(np.stack([rng.permutation(E)[:K] for _ in range(R)])[None].astype(np.int32))
got = DL.dense_experts(x, inds, src).astype(mx.float32)


def suite_expert(xr, e, bits):                     # the engine's math: gather_qmm on the (tiered) 4-bit codes
    out = {}
    for p, (w, s, b) in q.items():
        w_, s_, b_ = w[e:e + 1], s[e:e + 1], b[e:e + 1]
        if bits == 3:
            w_, b_ = EM.tier_weights(w_, s_, b_, GS, 3)
        out[p] = (w_, s_, b_)
    qmm = lambda h, p: mx.gather_qmm(h, *out[p], rhs_indices=mx.array([[0]], mx.uint32), transpose=True, group_size=GS, bits=4)
    xe = xr.astype(mx.float32).reshape(1, 1, 1, H)
    g, u = qmm(xe, "gate_proj"), qmm(xe, "up_proj")
    return qmm(g * mx.sigmoid(g) * u, "down_proj").reshape(H)


ref = mx.stack([mx.stack([suite_expert(x[0, r], int(inds[0, r, j]), 4 if j < 2 else 3) for j in range(K)]) for r in range(R)])
err = float(mx.abs(got[0] - ref).max() / mx.abs(ref).max())
check(f"V3Src: dense == the suite's quantized path per (row, rank), ranks 0-1 4-bit, 2+ MSB-3 (rel. err {err:.1e})", err < 2e-2)
ref3 = mx.stack([mx.stack([suite_expert(x[0, r], int(inds[0, r, j]), 3) for j in range(K)]) for r in range(R)])
d3 = float(mx.abs(got[0, :, 2:] - ref3[:, 2:]).max() / mx.abs(ref3).max())
d4 = float(mx.abs(got[0, :, :2] - ref3[:, :2]).max() / mx.abs(ref3).max())
check(f"V3Src: ranks 2+ are MSB-3 ({d3:.1e}) and ranks 0-1 are not ({d4:.1e})", d3 < 2e-2 and d4 > 5 * d3)
src.all4 = True
got4 = DL.dense_experts(x, inds, src).astype(mx.float32)
ref4 = mx.stack([mx.stack([suite_expert(x[0, r], int(inds[0, r, j]), 4) for j in range(K)]) for r in range(R)])
err4 = float(mx.abs(got4[0] - ref4).max() / mx.abs(ref4).max())
check(f"V3Src all4: every rank 4-bit ({err4:.1e})", err4 < 2e-2)

# 2. TrSrc from a decoder (Fit12's format: decode(L, e) -> {proj: (W_base, W_full)}), both key styles
def fake_decode(L, e, suffix=""):
    g = np.random.default_rng(1000 * L + e)
    return {p + suffix: tuple(g.normal(0, 0.05, sh).astype(np.float32) for _ in range(2))
            for p, sh in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I)))}


for suffix in ("", "_proj"):
    t = DL.TrSrc(5, lambda L, e: fake_decode(L, e, suffix), n_experts=E, gs=GS, workers=2)
    flat = mx.array([3, 1, 3, 6], mx.int32)
    good, worst = True, 0.0
    for hi, j in ((True, 1), (False, 0)):
        Wg = t.gather(flat, hi)
        for p in ("gate", "up", "down"):
            want = np.stack([fake_decode(5, e)[p][j] for e in (3, 1, 3, 6)])
            q, sc = zip(*[DL.int8_groups(w, GS) for w in want])
            deq = np.stack([(qq.reshape(ss.shape + (GS,)) * ss[..., None]).reshape(qq.shape) for qq, ss in zip(q, sc)])
            good &= np.allclose(np.array(Wg[p + "_proj"]), deq, rtol=0, atol=1e-7)
            worst = max(worst, float(np.square(deq - want).sum() / np.square(want).sum()))
    check(f"TrSrc ('{suffix or 'gate'}' keys): hi = W_full, lo = W_base, int8 x group scale exact; relative int8 error "
          f"{worst:.1e} (logged {t.err['base']:.1e} / {t.err['full']:.1e})",
          good and worst < 1e-4 and 0 < t.err["base"] < 1e-4 and 0 < t.err["full"] < 1e-4)

# 3. install_block on a tiny real SparseMoeBlock
from mtplx.models import qwen4_exp  # noqa: E402
args = types.SimpleNamespace(hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
                             norm_topk_prob=True, num_experts=E, num_experts_per_tok=K)
mx.random.seed(1)
blk = qwen4_exp.SparseMoeBlock(args)
called = {"n": 0}
sw_cls = type(blk.switch_mlp)
orig_sw = sw_cls.__call__


def sw_call(self, x_, i_):
    called["n"] += 1
    return orig_sw(self, x_, i_)


sw_cls.__call__ = sw_call
blk.switch_mlp._bm = {"L": 3}
fakeM = types.SimpleNamespace(STATE=types.SimpleNamespace(tiers_prefill=True))
xb = mx.array(rng.normal(0, 1, (1, R, H)).astype(np.float32))
stock = blk(xb)
DL.install_block(fakeM)


class FloatSrc:
    def __init__(self, lo_scale=1.0):
        self.lo = lo_scale

    def gather(self, flat, hi):                             # lo level: the down projection scaled (output x lo exactly)
        return {p: getattr(blk.switch_mlp, p).weight[flat].astype(mx.float32) * (1.0 if hi or p != "down_proj" else self.lo)
                for p in DL.PROJ}


DL.DENSE.update(on=True, src={3: FloatSrc()})
n0 = called["n"]
dense = blk(xb)
mx.eval(dense)
e1 = float(mx.abs(dense - stock).max() / mx.abs(stock).max())
check(f"install_block: dense block == stock block with the block's own weights ({e1:.1e}); switch still ran: {called['n'] > n0}",
      e1 < 1e-4 and called["n"] > n0)
DL.DENSE.update(src={3: FloatSrc(0.5)})                     # lo level = half the down projection: expert output x 0.5
half = blk(xb)
g = mx.softmax(blk.gate(xb), axis=-1, precise=True)
ii = mx.argsort(-g, axis=-1)[..., :K]
sc = mx.take_along_axis(g, ii, axis=-1)
sc = sc / sc.sum(-1, keepdims=True)
y_all = blk.switch_mlp(xb, ii)                              # [1, R, K, H] stock experts in rank order
fac = mx.array([1.0, 1.0] + [0.5] * (K - 2))[:, None]
want = (y_all * fac * sc[..., None]).sum(-2) + mx.sigmoid(blk.shared_expert_gate(xb)) * blk.shared_expert(xb)
e2 = float(mx.abs(half - want).max() / mx.abs(want).max())
check(f"install_block: ranks 2+ take the lo level, ranks 0-1 the hi level ({e2:.1e})", e2 < 1e-4)
DL.DENSE["on"] = False
off = blk(xb)
check("install_block: DENSE off -> the stock block", float(mx.abs(off - stock).max()) == 0.0)
fakeM.STATE.tiers_prefill = False
DL.DENSE.update(on=True, src={3: FloatSrc(0.5)})
check("install_block: prompt prefill (tiers_prefill False) -> the stock block", float(mx.abs(blk(xb) - stock).max()) == 0.0)
print("ALL OK" if ok_all else "SOME FAILED")
