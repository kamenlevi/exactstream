"""CPU self-test of the NEVER-WAIT v3 (b) patch (nw3_patch.py) on the decision map's harness (Breakthrough, Fri
2026-11-06 21:xx). Built on Fable FAR's test_nw.py (~/FableGod/far/build_1105/test_nw.py, read; its harness copied in
parts): a 48-layer model of the REAL classes the harness wraps (Q.DecoderLayer.__call__, Q.GatedResidual,
Q.SparseMoeBlock, Q._hyper_residual_write) at 32 dims, float64, on MLX's CPU device, through dm_common's own wrappers
(install()), in chunks of 40 / 33 / 7 rows, so both MoE paths run (>= 32 rows: _combine_p; < 32: the narrow path).
Stand-ins: the mixer and the routed experts (as FAR's), and dm_t3: a fake routed_topk with the real function's routing
ops whose experts are the block's own at "4 bits" and perturbed copies at "3 bits" (records3 needs the SSD store).
Checks (ALL must pass; exit 0):
  A. the old arms are unchanged: the baseline, the rmask twin, nw1, nw0 and a t3-gate arm (U3-like: sure rows' ranks 0-1
     via dm_t3) equal the UNPATCHED harness's outputs bit for bit (run first with --base <dir of the live files>);
  B. nw3 with the "4-bit" stand-in == nw1 to float64 rounding: the 3-bit path neither drops nor double-counts a term;
  C. nw3 with the "3-bit" stand-in == an independent reference forward (the gated rows' ranks 0-1 that are kept and not
     late at 3 bits on time; late ranks 4-bit, one layer late; the rest 4-bit), with power: the same reference with the
     late ranks 0-1 at 3 bits too, with the SURE rows gated instead, or with no 3-bit term is far from the harness;
  D. counts: the 3-bit terms served == the masks' count (gated rows only, prompt rows never); the late terms used ==
     the mask's; late writes == the chunk-layer pairs; nothing left stashed;
  E. guards: nw1 with t3 or with a gate is still refused; nw3 with topk, reuse, another gate or t3k 3 is refused;
  F. the real measure() end to end (FAR's fake prefill) over two windows with dm_ablations' own setups and a stubbed
     drafter gate: res['nw'] holds t3 == the masks' count and t3rows == the unsure answer rows, the nw3 note is printed,
     no WARNING; the 4-bit control's top-8 ids == nw1's and its log-probs within rounding; nw3 at 3 bits moves them;
  G. no state leaks: the twin after everything == before, bit for bit; clear() leaves CUR['rm3'] None;
  H. dm_ablations: the three nw3 arms resolve (alone and with +code) to the expected PERT; every name stands in quotes.
Usage (BigMoE venv; CPU, < 1 GB, about a minute):
  python -I -B test_nw3.py --base <dir with the live dm_common.py + dm_ablations.py>   # writes <scratch>/test3_base.npz
  python -I -B test_nw3.py <dir with the patched files>"""
import os, sys, types, inspect, json, io, contextlib
sys.dont_write_bytecode = True
BASE_MODE = len(sys.argv) > 2 and sys.argv[1] == "--base"
PD = os.path.abspath(sys.argv[2] if BASE_MODE else sys.argv[1])
SC = os.environ.get("NW3_TEST_SCRATCH", os.path.expanduser("~/Breakthrough/data/abl_scratch/nw3")); TO = os.path.join(SC, "test_out")
os.makedirs(TO, exist_ok=True)
os.environ["DM_OUT"] = TO                                       # the map's own out/ is never touched
os.environ["MTPLX_QWEN4_HC_PREFILL_READ"] = "0"                 # the eager read chain (the fused read is a Metal kernel)
os.environ["MLX_ENABLE_TF32"] = "0"
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)                                   # CPU only: nothing below may reach the GPU
import mlx.nn as nn
sys.path.insert(0, PD); sys.path.insert(1, os.path.expanduser("~/FableGod/decision_map"))   # the map's other modules, read only

FAKE = {"calls": 0, "rows": 0}
def fake_routed_topk(block, L, xg, bits=3, K=2, keep=None):
    """dm_t3.routed_topk's own routing ops (read from dm_t3.py) on the toy block; the experts at 'bits' (4: the block's
    own; 3: the perturbed copies)."""
    gates = mx.softmax(block.gate(xg), axis=-1, precise=True)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    sc = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob: sc = sc / sc.sum(axis=-1, keepdims=True)
    o = mx.argsort(-sc.astype(mx.float32), axis=-1)[:, :K]
    top_ids, top_sc = mx.take_along_axis(inds, o, axis=-1), mx.take_along_axis(sc, o, axis=-1)
    if keep is not None: top_sc = top_sc * mx.array(keep[:, :K]).astype(top_sc.dtype)
    y = block.switch_mlp.call_bits(xg, top_ids, bits)          # [G, K, D]
    FAKE["calls"] += 1; FAKE["rows"] += int(xg.shape[0])
    return (y * top_sc[..., None].astype(y.dtype)).sum(axis=1)
sys.modules["dm_t3"] = types.SimpleNamespace(routed_topk=fake_routed_topk, LSRC={})

import dm_common as C
import dm_ablations as A
assert os.path.dirname(os.path.abspath(C.__file__)) == PD and os.path.dirname(os.path.abspath(A.__file__)) == PD, (C.__file__, A.__file__)
assert mx.default_device() == mx.cpu, mx.default_device()
Q = C.Q
PATCHED = hasattr(C, "_nw3_setup")
assert PATCHED != BASE_MODE, "run --base on the live (unpatched) files and the plain form on the patched ones"

bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)

D, HC, NL, E, K, I = 32, 4, 48, 64, 10, 24
DT, NPDT = mx.float64, np.float64
PL = 0.9 / NL
args = Q.TextArgs(hidden_size=D, num_hidden_layers=NL, hc_count=HC, hc_lowrank=8, num_experts=E, num_experts_per_tok=K,
                  moe_intermediate_size=I, shared_expert_intermediate_size=I, norm_topk_prob=True,
                  layer_types=["linear_attention"] * NL)
def A_(a): return mx.array(a).astype(DT)

class StubGDN(Q.GatedDeltaNet):
    """Stand-in mixer (FAR's)."""
    def __init__(self, dim):
        nn.Module.__init__(self); self.lin = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, mask=None, cache=None):
        return mx.tanh(self.lin(x))

class SwitchSorted(nn.Module):
    """FAR's stand-in for the routed experts, plus 'call_bits': at 3 bits the perturbed copies (wg3 / wu3 / wd3)."""
    def __init__(self, dim, hid, n_exp):
        nn.Module.__init__(self); s = 1.0 / np.sqrt(dim)
        self.wg = mx.random.uniform(-s, s, (n_exp, hid, dim)); self.wu = mx.random.uniform(-s, s, (n_exp, hid, dim))
        self.wd = mx.random.uniform(-1.0 / np.sqrt(hid), 1.0 / np.sqrt(hid), (n_exp, dim, hid))
    def _mlp(self, x, indices, wg, wu, wd):
        xe = x[..., None, :, None]
        hg = (wg[indices] @ xe)[..., 0]; hu = (wu[indices] @ xe)[..., 0]
        return (wd[indices] @ (nn.silu(hg) * hu)[..., None])[..., 0]
    def __call__(self, x, indices):
        return self._mlp(x, indices, self.wg, self.wu, self.wd)
    def call_bits(self, x, indices, bits):
        return self(x, indices) if bits == 4 else self._mlp(x, indices, self.wg3, self.wu3, self.wd3)
    def sorted_experts(self, x, indices):
        y = self(x, indices); y2 = y.reshape(-1, y.shape[-1]); n = y2.shape[0]
        perm = np.random.RandomState(n).permutation(n).astype(np.uint32)
        return y2[mx.array(perm)], mx.array(np.argsort(perm).astype(np.uint32))

mx.random.seed(7)
def make_layer():
    L = Q.DecoderLayer.__new__(Q.DecoderLayer); nn.Module.__init__(L)
    L.layer_type = "linear_attention"; L.is_linear = True; L._hc = HC
    L.linear_attn = StubGDN(D)
    L.mlp = Q.SparseMoeBlock(args); L.mlp.switch_mlp = SwitchSorted(D, I, E)
    L.mlp.gate.weight = L.mlp.gate.weight * 6.0
    L.attn_hyper_connection = Q.GatedResidual(args); L.mlp_hyper_connection = Q.GatedResidual(args)
    return L
layers = [make_layer() for _ in range(NL)]
from mlx.utils import tree_map
for i, L in enumerate(layers):
    L.update(tree_map(lambda p_: p_.astype(DT), L.parameters())); mx.eval(L.parameters())
    sw = L.mlp.switch_mlp; r = np.random.RandomState(100 + i)
    pert = lambda w: w * A_(1.0 + 0.08 * r.uniform(-1, 1, w.shape))   # "3 bits": a fixed relative error per weight
    sw.wg3, sw.wu3, sw.wd3 = pert(sw.wg), pert(sw.wu), pert(sw.wd); mx.eval(sw.wg3, sw.wu3, sw.wd3)
model = types.SimpleNamespace(language_model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers)))
C.discover_layers(model); C.install()
Q._moe_prefill_combine_applies = lambda block, x: (x.size // x.shape[-1]) >= 32
assert type(layers[0]).__call__ is Q.DecoderLayer.__call__

def window_reset():
    C.PREV.clear(); C.CUR["pos"] = {}; C.CUR["kpos"] = {}
    C._nw_reset()

def fwd(x):
    for layer in layers: x = layer(x, input_ids=None, ssm_mask=None, cache=None)
    return x

def run_harness(h, chunks):
    window_reset(); outs = []; p = 0
    for S in chunks:
        x = fwd(mx.tile(A_(h[p:p + S])[None], (1, 1, HC)))
        mx.eval(x); outs.append(np.array(x[0])); p += S
    return np.concatenate(outs)

GR = C._ORIG[(Q.GatedResidual, "hc")]
def W(hyper, bo, inj):
    return hyper + (bo[..., None, :] * inj[..., :, None]).reshape(hyper.shape)

def run_ref3(h, chunks, keep, late, d, gated, bits3=True, late3=False):
    """The nw3 rule, written independently of the harness (experts in RANK order): a gated row's ranks 0-1 that are kept
    and not late run on the '3-bit' experts in layer L's own write; late ranks are written d layers late through layer
    L's inject gates at 4 bits (late3: a gated row's late ranks 0-1 at 3 bits, a wrong variant for power); the rest 4-bit."""
    outs = []; p = 0
    for S in chunks:
        x = mx.tile(A_(h[p:p + S])[None], (1, 1, HC)); pend = {}
        for M, layer in enumerate(layers):
            mixed, hyper, inj = GR(layer.attn_hyper_connection, x)
            bo = layer.linear_attn(mixed, None, None)
            mixed, hyper, inj = GR(layer.mlp_hyper_connection, hyper, pending=(bo, inj))
            blk = layer.mlp
            g = mx.softmax(blk.gate(mixed), axis=-1, precise=True)
            ids = mx.argsort(-g, axis=-1)[..., :K]
            sc = mx.take_along_axis(g, ids, axis=-1); sc = sc / sc.sum(axis=-1, keepdims=True)
            y4 = blk.switch_mlp(mixed, ids)                                             # [1, S, K, D], rank order
            y3 = blk.switch_mlp.call_bits(mixed, ids, 3) if bits3 else y4
            kp = np.ones((S, K), bool); lt = np.zeros((S, K), bool); g01 = np.zeros((S, K), bool)
            for j in range(S):
                if p + j < keep.shape[0]:
                    kp[j] = keep[p + j, M]; lt[j] = late[p + j, M]
                    if gated[p + j]: g01[j, :2] = True
            lt &= kp
            t3 = g01 & kp & ~lt                                                         # 3 bits, on time
            l3 = (g01 & lt) if late3 else np.zeros_like(lt)                             # (power) late and 3 bits
            sel = lambda m_: (sc * mx.array(m_)[None].astype(sc.dtype))[..., None]
            now = (y4 * sel(kp & ~lt & ~t3)).sum(axis=-2) + (y3 * sel(t3)).sum(axis=-2)
            dl = (y4 * sel(lt & ~l3)).sum(axis=-2) + (y3 * sel(l3)).sum(axis=-2)
            sh = mx.sigmoid(blk.shared_expert_gate(mixed)) * blk.shared_expert(mixed)
            x = W(hyper, now + sh, inj)
            if lt.any(): pend[M] = (dl, inj)
            if M - d in pend:
                dl_, inj_ = pend.pop(M - d); x = W(x, dl_, inj_)
        assert not pend, sorted(pend)
        mx.eval(x); outs.append(np.array(x[0])); p += S
    return np.concatenate(outs)

def rel(x, y): return float(np.abs(x - y).max() / np.abs(y).max())

def make_masks(N, a, d, seed):
    """FAR's: keep / late [N - 1, NL, K]; prompt rows keep all, no late; late flags on kept ranks of every rank."""
    r = np.random.RandomState(seed); keep = np.ones((N - 1, NL, K), bool); late = np.zeros((N - 1, NL, K), bool)
    for p in range(a, N - 1):
        if r.rand() < 0.8: keep[p, :, 2:] = r.rand(NL, K - 2) < 0.6
        if r.rand() < 0.85: late[p] = keep[p] & (r.rand(NL, K) < PL)
    late[45:70, 2] = False
    late[a + 2, NL - 1, 0] = keep[a + 2, NL - 1, 0]
    if d: late[:, NL - d:] = False
    return keep, late

def make_gate(N, a, seed):
    """Like dm_common._gate: [N] cf per row (prompt rows and no draft = -1, the last row -1); most answer rows sure."""
    r = np.random.RandomState(seed); g = np.full(N, -1.0)
    for p in range(a, N - 1):
        u = r.rand(); g[p] = -1.0 if u < 0.06 else (0.9 + 0.0999 * r.rand() if u < 0.72 else 0.89 * r.rand())
    return g

def pairs(lt, chunks):
    """(chunk, layer) pairs that hold a late flag (= the late writes), FAR's."""
    n = 0; p = 0
    for S in chunks:
        n += int(lt[p:min(p + S, lt.shape[0])].any(axis=(0, 2)).sum()); p += S
    return n

N, a, CH = 80, 20, [40, 33, 7]
h = np.random.RandomState(1).randn(N, D).astype(NPDT)
keep, late0 = make_masks(N, a, 0, 3); _, late1 = make_masks(N, a, 1, 3)
gt = make_gate(N, a, 9); U = np.zeros(N, bool); U[a:] = ~(gt[a:] >= 0.9)   # the unsure answer rows (the test's own count)
SURE = np.zeros(N, bool); SURE[a:N - 1] = gt[a:N - 1] >= 0.9
assert 8 <= U[:N - 1].sum() <= 40 and SURE.sum() >= 20, (U.sum(), SURE.sum())
assert (late1[np.flatnonzero(U[:N - 1]), :, :2]).any(), "the masks need a late rank 0-1 on a gated row (C's power check)"
def save(name, kp, lt=None):
    p = os.path.join(TO, name); np.savez(p, keep=kp, **({} if lt is None else {"late": lt})); return p + ".npz"
P_twin, P0, P1 = save("twin", keep), save("nw0", keep, late0), save("nw1", keep, late1)

def arm(path, **pert):
    C.clear(); C.PERT.update(rmask=path, renorm=False, **pert)
    C.CUR["rm"] = np.load(path)["keep"]
    if "nw" in pert: C.CUR["lm"] = C._nw_late(path, C.CUR["rm"])

def arm_t3gate():                                               # the old t3 gate (U3-like): sure rows' ranks 0-1 via dm_t3
    C.clear(); C.PERT.update(gate=("t3", 0.9), t3=3, t3k=2, renorm=False)
    kk = np.full(N, 10, np.int32); kk[gt >= 0.9] = 0; C.CUR["k"] = kk

def arm3(path, g, a_, bits=3, thr=0.9):                         # nw3 at the harness level (measure() does this per window)
    C.clear(); C.PERT.update(rmask=path, renorm=False, nw=1, nw3=True, gate=("t3u", thr), t3=bits, t3k=2)
    C.CUR["rm"] = np.load(path)["keep"]; C.CUR["lm"] = C._nw_late(path, C.CUR["rm"]); C._nw3_setup(g, a_)

print(f"mode: {'BASE (live files)' if BASE_MODE else 'PATCHED'} {PD}")
print("A. the old arms")
C.clear(); o_base = run_harness(h, CH)
arm(P_twin); o_twin = run_harness(h, CH)
arm(P1, nw=1); o_nw1 = run_harness(h, CH)
arm(P0, nw=0); o_nw0 = run_harness(h, CH)
arm_t3gate(); f0 = FAKE["calls"]; o_t3g = run_harness(h, CH); ok(FAKE["calls"] > f0, "the t3-gate arm reached dm_t3")
BP = os.path.join(SC, "test3_base.npz")
if BASE_MODE:
    np.savez(BP, base=o_base, twin=o_twin, nw1=o_nw1, nw0=o_nw0, t3g=o_t3g)
    print(f"stored the live harness's outputs: {BP}")
    print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True); sys.exit(1 if bad else 0)
B0 = np.load(BP)
for k_, o_ in (("base", o_base), ("twin", o_twin), ("nw1", o_nw1), ("nw0", o_nw0), ("t3g", o_t3g)):
    ok(np.array_equal(B0[k_], o_), f"{k_}: the patched harness == the live one, bit for bit")

print("B. nw3 at '4 bits' == nw1")
f1 = FAKE["calls"]; arm3(P1, gt, a, bits=4); o_c = run_harness(h, CH); n3_c, t_c = C.NW["t3"], C.NW["terms"]
ok(FAKE["calls"] > f1 and n3_c > 0, f"the nw3 path reached dm_t3 ({FAKE['calls'] - f1} calls) and served {n3_c} terms")
e_b = rel(o_c, o_nw1); ok(e_b < 1e-9, f"nw3 with the 4-bit stand-in vs nw1: relative error {e_b:.1e} (< 1e-9)")
print(f"  note: the 4-bit control equals nw1 bit for bit: {np.array_equal(o_c, o_nw1)} (expected False: two sums, one order each)")

print("C. nw3 at '3 bits' == the reference; power")
arm3(P1, gt, a, bits=3); o_3 = run_harness(h, CH)
n3, terms, writes, st = C.NW["t3"], C.NW["terms"], C.NW["writes"], dict(C.NW["stash"])
ref = run_ref3(h, CH, keep, late1, 1, U)
e_c = rel(o_3, ref); ok(e_c < 1e-9, f"nw3 vs the independent reference: relative error {e_c:.1e} (< 1e-9)")
SUREG = np.zeros(N, bool); SUREG[a:] = gt[a:] >= 0.9
for nm_, r_ in (("late ranks 0-1 at 3 bits too", run_ref3(h, CH, keep, late1, 1, U, late3=True)),
                ("the sure rows gated instead", run_ref3(h, CH, keep, late1, 1, SUREG)),
                ("no 3-bit term (nw1's rule)", run_ref3(h, CH, keep, late1, 1, U, bits3=False))):
    e_ = rel(o_3, r_); ok(e_ > 1e3 * max(e_c, 1e-16), f"power: the reference with {nm_} is {e_:.1e} away (> 1e3 x the match)")
e_m = rel(o_3, o_nw1); ok(e_m > 1e3 * max(e_c, 1e-16), f"nw3 at 3 bits moves the output against nw1 by {e_m:.1e}")

print("D. counts")
want3 = int((keep & ~late1)[np.flatnonzero(U[:N - 1]), :, :2].sum())
ok(n3 == want3 and n3_c == want3, f"3-bit terms served {n3} (4-bit control {n3_c}) == the masks' count on the gated answer rows {want3}")
ok(int((keep & ~late1)[:a, :, :2].sum()) > 0, "(prompt rows hold rank 0-1 terms, so serving them would show in the count)")
ok(terms == int(late1.sum()) and t_c == terms, f"late terms used {terms} == the mask's {int(late1.sum())} (and in the control)")
ok(writes == pairs(late1, CH) and not st, f"late writes {writes} == the chunk-layer pairs {pairs(late1, CH)}; nothing left stashed")
r2 = C.CUR["rm"]; gr = np.flatnonzero(U[:N - 1])
ok(np.array_equal(r2[gr, :, :2], keep[gr, :, :2] & late1[gr, :, :2]) and np.array_equal(np.delete(r2, gr, axis=0), np.delete(keep, gr, axis=0))
   and np.array_equal(r2[:, :, 2:], keep[:, :, 2:]), "the stock path's keep: a gated row's ranks 0-1 only where late; everything else as the mask")

print("E. guards")
def refuses(fn, what):
    try: fn()
    except AssertionError: ok(True, f"refused: {what}"); return
    ok(False, f"NOT refused: {what}")
def g1(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, t3=3, t3k=2); C._nw_late(P1, keep)
def g2(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, gate=("t3u", 0.9)); C._nw_late(P1, keep)
def g3(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, nw3=True, gate=("t3u", 0.9), t3=3, t3k=2, topk=4); C._nw_late(P1, keep)
def g4(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, nw3=True, gate=("t3u", 0.9), t3=3, t3k=2, reuse="all"); C._nw_late(P1, keep)
def g5(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, nw3=True, gate=("t3", 0.9), t3=3, t3k=2); C._nw_late(P1, keep)
def g6(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, nw3=True, gate=("t3u", 0.9), t3=3, t3k=3); C._nw_late(P1, keep)
for fn, what in ((g1, "nw1 with t3"), (g2, "nw1 with a gate"), (g3, "nw3 with topk"), (g4, "nw3 with reuse"),
                 (g5, "nw3 with the t3 gate instead of t3u"), (g6, "nw3 with t3k 3")):
    refuses(fn, what)

print("F. the real measure() and run_window(), end to end")
EMB = np.random.RandomState(5).randn(100, D).astype(NPDT); HEAD = A_(np.random.RandomState(6).randn(HC * D, 50).astype(NPDT) * 0.05)
def chunks_of(n): return [37] * (n // 37) + ([n % 37] if n % 37 else [])
def top8(x):
    lg = x.astype(HEAD.dtype) @ HEAD
    top = mx.argsort(-lg, axis=-1)[:, :8]
    return top.astype(mx.int32), mx.take_along_axis(lg, top, axis=-1) - mx.logsumexp(lg, axis=-1, keepdims=True)
def fake_prefill(rt, ids, **kw):
    hh = EMB[np.array(ids) % 100]; p = 0
    for S_ in chunks_of(len(ids)):
        x = fwd(mx.tile(A_(hh[p:p + S_])[None], (1, 1, HC)))
        if C.REC["on"]:
            t, l = top8(x[0]); C.REC["tk"].append(t); C.REC["lp"].append(l)
        p += S_
C.G.restore_or_prefill_prompt_state = fake_prefill
C.RT.update(rt=None, st=types.SimpleNamespace(gate=types.SimpleNamespace(check=lambda: None)), head=None)
WIN = []; MK = os.path.join(TO, "masks"); LT = {}; GT = {}
for wi, (n_, a_) in enumerate(((80, 20), (50, 12))):
    nm = f"w{wi}.npz"; ids = np.random.RandomState(40 + wi).randint(0, 100, n_).astype(np.int32)
    kp, l1 = make_masks(n_, a_, 1, 50 + wi)
    for sub, lt in (("twin", None), ("nw1", l1)):
        os.makedirs(os.path.join(MK, sub), exist_ok=True)
        np.savez(os.path.join(MK, sub, nm[:-4]), keep=kp, **({} if lt is None else {"late": lt}))
    WIN.append(("code", nm, ids, 0, a_ - 1)); LT[nm] = (kp, l1, a_); GT[nm] = make_gate(n_, a_, 60 + wi)
C._gate = lambda nm, n: GT[nm][:n]                              # the drafter gate (dm_gate's cf per row), stubbed
def meas(name, setup):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): r = C.measure(model, WIN, name, setup)
    return r, buf.getvalue()
T, out_t = meas("t3_twin", A.q1_union(os.path.join(MK, "twin")))
R1, out_1 = meas("t3_nw1", A.nw_mask(os.path.join(MK, "nw1"), 1))
R4, out_4 = meas("t3_nw3c", A.nw3_mask(os.path.join(MK, "nw1"), bits=4))
R3, out_3 = meas("t3_nw3", A.nw3_mask(os.path.join(MK, "nw1"), bits=3))
good = True
for dom, nm, ids, off, gen in WIN:
    kp, l1, a_ = LT[nm]; u = np.zeros(len(ids), bool); u[a_:] = ~(GT[nm][a_:] >= 0.9)
    w3 = int((kp & ~l1)[np.flatnonzero(u[:kp.shape[0]]), :, :2].sum()); wr = int(u[a_:len(ids) - 1].sum())
    for R_, b_ in ((R3, 3), (R4, 4)):
        nw = R_[nm].get("nw", {})
        good &= nw.get("t3") == w3 and nw.get("t3rows") == wr and nw.get("bits") == b_ and nw.get("terms") == int(l1.sum()) and "warn" not in R_[nm]
ok(good, "measure(): res['nw'] holds t3 == the masks' count, t3rows == the unsure answer rows, bits, terms == the mask's; no warn")
ok(out_3.count("nw3: ") == 2 and out_4.count("nw3: ") == 2 and "WARNING" not in out_1 + out_3 + out_4 + out_t,
   "measure(): the nw3 note is printed per window, no WARNING")
d4 = max(np.abs(np.array(R4[w]["lp"]) - np.array(R1[w]["lp"])).max() for w in R1)
LPT = 5e-6                                                      # FAR's: log-probs are stored as float32
ok(all(R4[w]["tk"] == R1[w]["tk"] for w in R1) and d4 < LPT, f"measure(): the 4-bit control's top-8 ids == nw1's, max |d log p| {d4:.1e} (< {LPT})")
d3 = max(np.abs(np.array(R3[w]["lp"]) - np.array(R1[w]["lp"])).max() for w in R1)
ok(d3 > 20 * max(d4, LPT), f"measure(): nw3 at 3 bits moves the log-probs by up to {d3:.1e} against nw1 (not a no-op)")

print("G. no state leaks")
arm(P_twin); o_t2 = run_harness(h, CH)
ok(np.array_equal(o_t2, o_twin), "the twin after the nw3 runs == before, bit for bit")
C.clear(); ok(C.CUR.get("rm3") is None and C.CUR.get("lm") is None and np.array_equal(run_harness(h, CH), o_base),
              "clear() leaves CUR['rm3'] / ['lm'] None; the baseline after them == before, bit for bit")
T2, _ = meas("t3_twin2", A.q1_union(os.path.join(MK, "twin")))
ok(all(T2[w]["tk"] == T[w]["tk"] and T2[w]["lp"] == T[w]["lp"] for w in T), "measure(): the twin after all of it == before, exactly")

print("H. dm_ablations arms")
S_ = "/Users/kamen/FableGod/decision_map/out/far/"
base3 = dict(renorm=False, nw=1, nw3=True, gate=("t3u", 0.9), t3k=2)
exp = {"qom_0_live_ball_c208_nw3": dict(base3, rmask=S_ + "qom_0_live_ball_c208_nw1", t3=3),
       "qom_0_live_ball_nw3": dict(base3, rmask=S_ + "qom_0_live_ball_nw1", t3=3),
       "qom_0_live_ball_c208_nw3c": dict(base3, rmask=S_ + "qom_0_live_ball_c208_nw1", t3=4)}
n_ok = 0
for name, want in exp.items():
    C.clear(); A.ablation(name)(); p1 = dict(C.PERT); C.clear(); A.ablation(name + "+code")(); p2 = dict(C.PERT); C.clear()
    n_ok += p1 == want and p2 == want
ok(n_ok == len(exp), f"{n_ok}/{len(exp)} nw3 arms resolve, alone and with +code, to the expected PERT")
src = open(os.path.join(PD, "dm_ablations.py")).read()
ok(all(f'"{k}"' in src for k in exp), "every nw3 arm name stands in dm_ablations.py in quotes (the s67 runner's installed() check)")
ok(all(os.path.isdir(v["rmask"]) for v in exp.values()), "their mask folders exist (nw1's snapshots in out/far)")
ok("_nw3_setup(" in inspect.getsource(C.measure) and "rm3" in inspect.getsource(C.clear), "measure() runs the nw3 setup; clear() resets rm3")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
