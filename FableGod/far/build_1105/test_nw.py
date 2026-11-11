"""CPU self-test of the NEVER-WAIT harness patch (Fable FAR, Thu 5 Nov 2026; NEVERWAIT_TEST.md 2.5's three self-checks
and more). No GPU, no 125B: a 48-layer model built from the REAL classes the harness wraps (Q.DecoderLayer's own
__call__, Q.GatedResidual, Q.SparseMoeBlock, Q._hyper_residual_write) at 32 dims, float64, on MLX's CPU device, run
through the patched dm_common's own wrappers (install()) in chunks of 40 / 33 / 7 rows, so both chunk widths run: the
>= 32-row path (the block's _prefill_call -> dm_common._combine_p -> the kernel module's reference combine) and the
narrow path of the MoE wrapper. Two stand-ins only, neither touched by never-wait: the mixer (a per-row linear map) and
the routed experts themselves (small gated MLPs; sorted_experts hands their outputs over in a shuffled order with the
inverse permutation, as the fused switch layer does). float64 because 48 random layers amplify float32 rounding noise
about 5,000 x, which would blur "harness == reference"; NW_TEST_DTYPE=float32 NW_TEST_LAYERS=6 runs the small form, and
NW_TEST_DTYPE=bfloat16 a smoke run in the 125B's own dtype (exact checks hard, reference comparisons as notes).

What is checked (ALL must pass; exit 0):
  A. the patch leaves the old arms alone: baseline, a topk arm and the rmask twin equal the UNPATCHED harness's output
     bit for bit (run this file first with --base <dir of the unpatched files>; it stores the outputs);
  B. self-check (i): the never-wait path with every late flag off == the twin, bit for bit;
  C. self-check (ii): lateness 0 == the twin up to the order of two adds (float64: relative error < 1e-9);
  D. nw1 and nw2 == an independent reference forward written from the spec (delta_L = the late ranks' weighted sum on
     layer L's input with layer L's scores, written after layer (L + d)'s MoE write through layer L's inject gates),
     and the test has power: the same reference with layer (L + d)'s gates, with the late rows shifted by one, or with
     another lateness is far from the harness;
  E. self-check (iii): every layer saw every row once (kpos), every late flag was used exactly once (NW terms == the
     mask's), the late writes == the (chunk, layer) pairs that hold a late flag, nothing is left stashed, and both
     chunk widths stashed late terms;
  F. guards: a late flag at a tail layer, a late flag outside keep, never-wait with topk, and a late term that reaches
     the last layer are all refused loudly;
  G. state does not leak: the twin after a never-wait run == the twin before it, bit for bit;
  H. dm_ablations: every new arm resolves (alone and with +code) to the expected PERT;
  I. the REAL measure() and run_window() end to end (only the prefill call is replaced by this model's forward): the
     twin, nwz, nw0 and nw1 arms over two windows through dm_ablations' own setups: nwz's JSON == the twin's, nw1's
     top-8 == a reference run, the "never-wait" note and res["nw"] are written, no WARNING; a prefill that skips rows
     trips both the row-count WARNING and the never-wait one.
Usage (BigMoE venv; CPU, < 1 GB, under a minute):
  python -I -B test_nw.py --base <dir with the unpatched dm_common.py>     # writes scratch/test_base.npz
  python -I -B test_nw.py [<dir with the patched files>]                  # default: ./patched"""
import os, sys, types, inspect, json, io, contextlib
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODE = len(sys.argv) > 2 and sys.argv[1] == "--base"
PD = os.path.abspath(sys.argv[2] if BASE_MODE else (sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "patched")))
SC = os.path.join(os.environ.get("NW_TEST_SCRATCH", os.path.join(HERE, "scratch"))); TO = os.path.join(SC, "test_out")
os.makedirs(TO, exist_ok=True)
os.environ["DM_OUT"] = TO                                       # the map's own out/ is never touched
os.environ["MTPLX_QWEN4_HC_PREFILL_READ"] = "0"                 # the eager read chain (the fused read is a Metal kernel)
os.environ["MLX_ENABLE_TF32"] = "0"
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu)                                   # CPU only: nothing below may reach the GPU
import mlx.nn as nn
sys.path.insert(0, PD)
import dm_common as C
import dm_ablations as A
assert os.path.dirname(os.path.abspath(C.__file__)) == PD and os.path.dirname(os.path.abspath(A.__file__)) == PD, (C.__file__, A.__file__)
assert mx.default_device() == mx.cpu, mx.default_device()
Q = C.Q

bad = []
def ok(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c: bad.append(m)

D, HC, NL, E, K, I = 32, 4, int(os.environ.get("NW_TEST_LAYERS", "48")), 64, 10, 24
DTN = os.environ.get("NW_TEST_DTYPE", "float64")               # float64 (MLX CPU only): 48 random layers amplify float32 rounding
HALF = DTN in ("bfloat16", "float16")                           # noise ~5,000 x, which would blur "harness == reference"
DT, NPDT = getattr(mx, DTN), (np.float32 if HALF else getattr(np, DTN))
# NW_TEST_DTYPE=bfloat16 (the 125B's dtype; the compiled residual write engages at >= 32 rows): a smoke run. The exact
# checks stay hard (bit-equalities, counts, guards, measure()); the comparisons with the reference are printed as notes,
# since 8-bit mantissas over 48 layers do not allow a tolerance.
def A_(a): return mx.array(a).astype(DT)
def N_(x): return np.array(x.astype(mx.float32)) if HALF else np.array(x)
PL = 0.9 / NL                                                   # late flag rate per kept rank (about 4 late terms per answer row)
args = Q.TextArgs(hidden_size=D, num_hidden_layers=NL, hc_count=HC, hc_lowrank=8, num_experts=E, num_experts_per_tok=K,
                  moe_intermediate_size=I, shared_expert_intermediate_size=I, norm_topk_prob=True,
                  layer_types=["linear_attention"] * NL)

class StubGDN(Q.GatedDeltaNet):
    """Stand-in mixer (a per-row map; the real mixers need caches and never-wait does not touch them)."""
    def __init__(self, dim):
        nn.Module.__init__(self); self.lin = nn.Linear(dim, dim, bias=False)
    def __call__(self, x, mask=None, cache=None):
        return mx.tanh(self.lin(x))

class SwitchSorted(nn.Module):
    """Stand-in for the routed experts (the real one is the fused, quantized switch layer; mlx_lm's SwitchGLU is float32
    only): E small gated MLPs, x [..., D] and indices [..., k] -> [..., k, D]; sorted_experts hands the same outputs over
    as _FusedGateUpSwitchGLU.sorted_experts does, [rows * k, D] in a shuffled order with the inverse permutation."""
    def __init__(self, dim, hid, n_exp):
        nn.Module.__init__(self); s = 1.0 / np.sqrt(dim)
        self.wg = mx.random.uniform(-s, s, (n_exp, hid, dim)); self.wu = mx.random.uniform(-s, s, (n_exp, hid, dim))
        self.wd = mx.random.uniform(-1.0 / np.sqrt(hid), 1.0 / np.sqrt(hid), (n_exp, dim, hid))
    def __call__(self, x, indices):
        xe = x[..., None, :, None]                                  # [..., 1, D, 1]
        hg = (self.wg[indices] @ xe)[..., 0]; hu = (self.wu[indices] @ xe)[..., 0]   # [..., k, hid]
        return (self.wd[indices] @ (nn.silu(hg) * hu)[..., None])[..., 0]
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
    L.mlp.gate.weight = L.mlp.gate.weight * 6.0                 # peaked routing: well separated ranks
    L.attn_hyper_connection = Q.GatedResidual(args); L.mlp_hyper_connection = Q.GatedResidual(args)
    return L
layers = [make_layer() for _ in range(NL)]
from mlx.utils import tree_map
for L in layers:
    L.update(tree_map(lambda p_: p_.astype(DT), L.parameters())); mx.eval(L.parameters())
model = types.SimpleNamespace(language_model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers)))
C.discover_layers(model); C.install()
Q._moe_prefill_combine_applies = lambda block, x: (x.size // x.shape[-1]) >= 32   # the >= 32-row path, without bf16 / the fused switch
assert type(layers[0]).__call__ is Q.DecoderLayer.__call__

def window_reset():                                             # what run_window() does at the start of each window
    C.PREV.clear(); C.CUR["pos"] = {}; C.CUR["kpos"] = {}
    if hasattr(C, "_nw_reset"): C._nw_reset()

def fwd(x):
    for layer in layers: x = layer(x, input_ids=None, ssm_mask=None, cache=None)   # Qwen4ExpTextModel._forward's eager loop
    return x

def run_harness(h, chunks):
    """The harness path: the real DecoderLayer.__call__ per layer, chunk by chunk, with dm_common's wrappers on."""
    window_reset(); outs = []; p = 0
    for S in chunks:
        x = fwd(mx.tile(A_(h[p:p + S])[None], (1, 1, HC)))
        mx.eval(x); outs.append(N_(x[0])); p += S
    return np.concatenate(outs)

GR = C._ORIG[(Q.GatedResidual, "hc")]                           # the read, unwrapped
def W(hyper, bo, inj):                                          # the residual write, spelled out (not Q's function)
    return hyper + (bo[..., None, :] * inj[..., :, None]).reshape(hyper.shape)

def run_ref(h, chunks, keep, late, d, wrong_gates=False, row_shift=0, strict=True):
    """NEVERWAIT_TEST.md 2.4, written independently of the harness: experts in RANK order (argsort), keep / late applied
    by rank, delta_L written after layer (L + d)'s MoE write through layer L's inject gates."""
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
            y = blk.switch_mlp(mixed, ids)                                           # [1, S, K, D], rank order
            kp = np.ones((S, K), bool); lt = np.zeros((S, K), bool)
            for j in range(S):
                if p + j < keep.shape[0]: kp[j] = keep[p + j, M]
                q = p + j + row_shift
                if late is not None and 0 <= q < late.shape[0]: lt[j] = late[q, M]
            lt &= kp
            now = (y * (sc * mx.array(kp & ~lt)[None].astype(sc.dtype))[..., None]).sum(axis=-2)
            dl = (y * (sc * mx.array(lt)[None].astype(sc.dtype))[..., None]).sum(axis=-2)
            sh = mx.sigmoid(blk.shared_expert_gate(mixed)) * blk.shared_expert(mixed)
            x = W(hyper, now + sh, inj)
            if lt.any(): pend[M] = (dl, inj)
            if M - d in pend:
                dl_, inj_ = pend.pop(M - d)
                x = W(x, dl_, inj if wrong_gates else inj_)
        if strict: assert not pend, sorted(pend)
        mx.eval(x); outs.append(N_(x[0])); p += S
    return np.concatenate(outs)

def rel(x, y): return float(np.abs(x - y).max() / np.abs(y).max())

def make_masks(N, a, d, seed):
    """keep / late [N - 1, NL, K] shaped like the map's: prompt rows exact; most answer rows 'sure' (ranks 0-1 + some of
    2-9), the rest keep all ten; late flags on kept ranks of every rank, rank 0 included; the tail resolved for d."""
    r = np.random.RandomState(seed); keep = np.ones((N - 1, NL, K), bool); late = np.zeros((N - 1, NL, K), bool)
    for p in range(a, N - 1):
        if r.rand() < 0.8: keep[p, :, 2:] = r.rand(NL, K - 2) < 0.6
        if r.rand() < 0.85: late[p] = keep[p] & (r.rand(NL, K) < PL)
    late[45:70, 2] = False                                      # one layer with no late flag in the middle chunk
    late[a + 2, NL - 1, 0] = keep[a + 2, NL - 1, 0]             # a late flag at the last layer (dropped below when d >= 1)
    if d: late[:, NL - d:] = False
    return keep, late

N, a, CH = 80, 20, [40, 33, 7]
h = np.random.RandomState(1).randn(N, D).astype(NPDT)
keep, late0 = make_masks(N, a, 0, 3); _, late1 = make_masks(N, a, 1, 3); _, late2 = make_masks(N, a, 2, 3)
assert np.array_equal(late1[:, :NL - 1], late0[:, :NL - 1]) and late0[:, NL - 1].any() and late1.any() and late2.any()
assert late1[:, :, 0].any() and (late1.any(axis=(1, 2)) & keep.all(axis=(1, 2))).any()   # a late rank 0; a late term on a row that keeps all ten
def save(name, kp, lt=None):
    p = os.path.join(TO, name); np.savez(p, keep=kp, **({} if lt is None else {"late": lt})); return p + ".npz"
P_twin, P0, P1, P2 = save("twin", keep), save("nw0", keep, late0), save("nw1", keep, late1), save("nw2", keep, late2)

def arm(path, **pert):
    """measure()'s per-arm / per-window setup for an rmask arm (section I runs measure() itself)."""
    C.clear(); C.PERT.update(rmask=path, renorm=False, **pert)
    C.CUR["rm"] = np.load(path)["keep"]
    if "nw" in pert: C.CUR["lm"] = C._nw_late(path, C.CUR["rm"])

# ---- the old arms (run in both modes) ------------------------------------------------------------------------------
C.clear(); o_base = run_harness(h, CH)
C.clear(); C.PERT.update(topk=4, renorm=False); o_topk = run_harness(h, CH)
arm(P_twin); o_twin = run_harness(h, CH); kpos_twin = dict(C.CUR["kpos"])
C.clear()
BF = os.path.join(SC, f"test_base_{NL}_{DTN}.npz")
if BASE_MODE:
    assert not hasattr(C, "NW"), "--base wants the UNPATCHED dm_common.py"
    np.savez(BF, base=o_base, topk=o_topk, twin=o_twin)
    print(f"[base] unpatched harness ({PD}): baseline / topk4 / twin outputs stored in {BF}; twin differs from the baseline "
          f"by {rel(o_twin, o_base):.3f} (relative), topk4 by {rel(o_topk, o_base):.3f}")
    sys.exit(0)

print(f"[t] patched harness from {PD}; MLX device {mx.default_device()}; {NL} layers x {D} dims, {E} experts, top {K}; window {N} rows "
      f"(answer from {a}) in chunks {CH}, {DTN}; late flags: {int(late1.sum())} (nw1), {int(late1[a:].sum()) / (N - 1 - a):.1f} per answer row", flush=True)
TOL = 1e-9 if DTN == "float64" else 1e-4; LPT = max(10 * TOL, 5e-6)   # log-probs are stored as float32
def soft(c, m):                                                 # a comparison with the reference: hard, but a note in half precision
    if HALF: print(("  note " if c else "  NOTE (not within the float tolerance, expected in half precision) ") + m, flush=True)
    else: ok(c, m)
print("A. the old arms are unchanged by the patch")
if os.path.exists(BF):
    z = np.load(BF)
    ok(np.array_equal(z["base"], o_base) and np.array_equal(z["topk"], o_topk) and np.array_equal(z["twin"], o_twin),
       "baseline, topk4 and the rmask twin == the unpatched harness's outputs, bit for bit")
else: ok(False, f"{BF} is missing: run this file with --base <unpatched dir> first")
e_ref = rel(o_twin, run_ref(h, CH, keep, None, 0))
soft(e_ref < TOL, f"the twin == the reference forward with no late flag (relative error {e_ref:.1e}): the reference copies the model's forward")

print("B. self-check (i): never-wait path, late flags off")
arm(P_twin, nw=1, nw_zero=True); o_z = run_harness(h, CH)
ok(np.array_equal(o_z, o_twin), "nw_zero == the twin, bit for bit")
ok(C.NW["writes"] == 0 and C.NW["terms"] == 0 and not C.NW["stash"] and not C.NW["inj"], "no late write, nothing stashed, no gates held")

print("C. self-check (ii): lateness 0")
arm(P0, nw=0); o_0 = run_harness(h, CH); n0 = dict(C.NW)
e0 = rel(o_0, o_twin)
soft(e0 < TOL, f"nw0 == the twin up to the order of two adds (relative error {e0:.1e}; {DTN} here, bf16 rounding on the 125B)")
ok(not np.array_equal(o_0, o_twin), "nw0 is not bit-equal to the twin (the late writes did run)")
ok(n0["terms"] == int(late0.sum()), f"every late flag used once ({n0['terms']} of {int(late0.sum())})")

print("D. nw1 / nw2 against the independent reference; the test's power")
shapes = []; stash0 = C._nw_stash
def stash_spy(li, delta): shapes.append(delta.ndim); stash0(li, delta)
C._nw_stash = stash_spy
arm(P1, nw=1); o_1 = run_harness(h, CH); n1 = dict(C.NW); kpos1 = dict(C.CUR["kpos"])
C._nw_stash = stash0
r_1 = run_ref(h, CH, keep, late1, 1); e1 = rel(o_1, r_1)
soft(e1 < TOL, f"nw1 == the reference (relative error {e1:.1e})")
eff = rel(o_1, o_twin)
soft(eff > 50 * max(e1, 1e-7), f"nw1 differs from the twin by {eff:.1e} (relative), {eff / max(e1, 1e-9):.0f} x the match: the late terms matter here")
for what, d_, kw in (("layer L + 1's gates instead of layer L's", 1, dict(wrong_gates=True)), ("the late rows shifted by one", 1, dict(row_shift=1)),
                     ("lateness 2 on nw1's flags", 2, {}), ("lateness 0 on nw1's flags", 0, {})):
    w = run_ref(h, CH, keep, late1, d_, strict=False, **kw)
    soft(rel(o_1, w) > 50 * max(e1, 1e-7), f"power: a reference with {what} is {rel(o_1, w):.1e} from the harness ({rel(o_1, w) / max(e1, 1e-9):.0f} x the match)")
arm(P2, nw=2); o_2 = run_harness(h, CH); n2 = dict(C.NW)
r_2 = run_ref(h, CH, keep, late2, 2); e2 = rel(o_2, r_2)
soft(e2 < TOL, f"nw2 == the reference (relative error {e2:.1e})")
w = rel(o_2, run_ref(h, CH, keep, late2, 1, strict=False))
soft(w > 50 * max(e2, 1e-7), f"power: nw2 is {w:.1e} from a lateness-1 reference ({w / max(e2, 1e-9):.0f} x the match)")

print("E. self-check (iii): rows, flags, writes")
def pairs(lt, chunks=CH):                                       # (chunk, layer) pairs that hold a late flag
    n = 0; p = 0
    for S in chunks: n += int(lt[p:p + S].any(axis=(0, 2)).sum()); p += S
    return n
ok(set(kpos1.values()) == {N} and len(kpos1) == NL and kpos1 == kpos_twin, f"every layer saw each of the {N} rows once (kpos == the twin's)")
ok(n1["terms"] == int(late1.sum()) and n2["terms"] == int(late2.sum()), f"every late flag used exactly once (nw1 {n1['terms']}, nw2 {n2['terms']})")
ok(n1["writes"] == pairs(late1) and n2["writes"] == pairs(late2) and n0["writes"] == pairs(late0),
   f"late writes == the (chunk, layer) pairs with a late flag (nw0 {n0['writes']}, nw1 {n1['writes']}, nw2 {n2['writes']})")
ok(not n1["stash"] and not n1["inj"] and not n2["stash"] and not n0["stash"], "nothing left stashed, no gates held after the last layer")
ok(2 in shapes and 3 in shapes, f"both chunk widths stashed late terms ({shapes.count(2)} from _combine_p, {shapes.count(3)} from the narrow path)")
for nm_, chunks_ in (("one wide chunk", [N]), ("narrow chunks only", [20, 20, 20, 20]), ("uneven chunks", [33, 5, 42])):
    arm(P1, nw=1); o_ = run_harness(h, chunks_)
    ok(C.NW["terms"] == int(late1.sum()) and not C.NW["stash"], f"nw1 with {nm_} {chunks_}: every late flag used once, nothing left stashed")
    soft(rel(o_, run_ref(h, chunks_, keep, late1, 1)) < TOL, f"nw1 with {nm_} {chunks_} == the reference")

print("F. guards")
def refuses(fn, what):
    try: fn(); ok(False, f"{what}: NOT refused")
    except AssertionError as e: ok(True, f"{what}: refused ({str(e).split(': ')[-1][:80]})")
refuses(lambda: arm(P0, nw=1), "a late flag at a tail layer under nw1")
lb = late1.copy(); lb[a + 1, 0, 5] = True; kb = keep.copy(); kb[a + 1, 0, 5] = False
refuses(lambda: arm(save("badsub", kb, lb), nw=1), "a late flag outside keep")
def with_topk(): C.clear(); C.PERT.update(rmask=P1, renorm=False, nw=1, topk=4); C._nw_late(P1, keep)
refuses(with_topk, "never-wait with topk")
def sneaks(): arm(P1, nw=1); C.CUR["lm"] = late0; run_harness(h, CH)
refuses(sneaks, "a late term that reaches the last layer")

print("G. no state leaks")
arm(P_twin); o_t2 = run_harness(h, CH)
ok(np.array_equal(o_t2, o_twin), "the twin after the never-wait runs == the twin before them, bit for bit")
C.clear(); ok(np.array_equal(run_harness(h, CH), o_base) and C.CUR.get("lm") is None, "the baseline after them == the baseline before, bit for bit")

print("H. dm_ablations arms")
S = os.environ.get("NW_SNAP", "/Users/kamen/FableGod/decision_map/out/far").rstrip("/") + "/"
exp = {"qom_0_live_ball_c208": dict(rmask=S + "qom_0_live_ball_c208", renorm=False),
       "qom_0_live_ball_c208_nw1": dict(rmask=S + "qom_0_live_ball_c208_nw1", renorm=False, nw=1),
       "qom_0_live_ball_c208_nw1r1": dict(rmask=S + "qom_0_live_ball_c208_nw1r1", renorm=False, nw=1),
       "qom_0_live_ball_nw1": dict(rmask=S + "qom_0_live_ball_nw1", renorm=False, nw=1),
       "qom_0_split0.99_live_ball_nw1": dict(rmask=S + "qom_0_split0.99_live_ball_nw1", renorm=False, nw=1),
       "qom_0_live_ball_c266": dict(rmask=S + "qom_0_live_ball_c266", renorm=False),
       "qom_0_live_ball_c266_nw1": dict(rmask=S + "qom_0_live_ball_c266_nw1", renorm=False, nw=1),
       "qom_0_live_ball_c208_nwz": dict(rmask=S + "qom_0_live_ball_c208", renorm=False, nw=1, nw_zero=True),
       "qom_0_live_ball_c208_nw0": dict(rmask=S + "qom_0_live_ball_c208_nw0", renorm=False, nw=0),
       "qom_0_live_ball_c208_nw2": dict(rmask=S + "qom_0_live_ball_c208_nw2", renorm=False, nw=2),
       "qom_0_live_ball_c208_nw1s": dict(rmask=S + "qom_0_live_ball_c208_nw1s", renorm=False, nw=1),
       "qom_0_split0.99_live_ball_c208": dict(rmask=S + "qom_0_split0.99_live_ball_c208", renorm=False),
       "qom_0_split0.99_live_ball_c208_nw1": dict(rmask=S + "qom_0_split0.99_live_ball_c208_nw1", renorm=False, nw=1),
       "qom_0_split0.9_live_ball_nw1": dict(rmask=S + "qom_0_split0.9_live_ball_nw1", renorm=False, nw=1)}
n_ok = 0
for name, want in exp.items():
    C.clear(); A.ablation(name)(); p1 = dict(C.PERT); C.clear(); A.ablation(name + "+code")(); p2 = dict(C.PERT); C.clear()
    n_ok += p1 == want and p2 == want and os.path.join(C.OUT, want["rmask"], "x") == os.path.join(want["rmask"], "x")
ok(n_ok == len(exp), f"{n_ok}/{len(exp)} arms resolve, alone and with +code, to the expected PERT and an absolute mask folder")
src = open(os.path.join(PD, "dm_ablations.py")).read()
ok(all(f'"{k}"' in src for k in exp), "every arm name stands in dm_ablations.py in quotes (the s67 runner's installed() check)")
ok("_nw_reset()" in inspect.getsource(C.run_window) and "_nw_reset()" in inspect.getsource(C.clear) and "_nw_late(" in inspect.getsource(C.measure),
   "run_window and clear reset the never-wait state; measure loads the late flags")

print("I. the real measure() and run_window(), end to end")
if NL != 48: print("  --   skipped (measure() counts 48 layers; run with NW_TEST_LAYERS=48)")
else:
    EMB = np.random.RandomState(5).randn(100, D).astype(NPDT); HEAD = A_(np.random.RandomState(6).randn(HC * D, 50).astype(NPDT) * 0.05)
    MODE = {"skip": False}
    def chunks_of(n): return [37] * (n // 37) + ([n % 37] if n % 37 else [])
    def top8(x):
        lg = x.astype(HEAD.dtype) @ HEAD; lg = lg.astype(mx.float32) if HALF else lg   # the real hook casts the logits to float32
        top = mx.argsort(-lg, axis=-1)[:, :8]
        return top.astype(mx.int32), mx.take_along_axis(lg, top, axis=-1) - mx.logsumexp(lg, axis=-1, keepdims=True)
    def fake_prefill(rt, ids, **kw):
        """Stands in for G.restore_or_prefill_prompt_state: this model's forward over the window in chunks of 37 rows
        (the last one narrow), the hook's top-8 capture into C.REC. MODE["skip"]: the last chunk never runs the layers."""
        hh = EMB[np.array(ids) % 100]; p = 0
        for i, S_ in enumerate(chunks_of(len(ids))):
            x0 = mx.tile(A_(hh[p:p + S_])[None], (1, 1, HC))
            x = x0 if (MODE["skip"] and p + S_ == len(ids)) else fwd(x0)
            if C.REC["on"]:
                t, l = top8(x[0]); C.REC["tk"].append(t); C.REC["lp"].append(l)
            p += S_
    C.G.restore_or_prefill_prompt_state = fake_prefill
    C.RT.update(rt=None, st=types.SimpleNamespace(gate=types.SimpleNamespace(check=lambda: None)), head=None)
    WIN = []; MK = os.path.join(TO, "masks"); LT = {}
    for wi, (n_, a_) in enumerate(((80, 20), (50, 12))):
        nm = f"w{wi}.npz"; ids = np.random.RandomState(40 + wi).randint(0, 100, n_).astype(np.int32)
        kp, l0 = make_masks(n_, a_, 0, 50 + wi); _, l1 = make_masks(n_, a_, 1, 50 + wi)
        for sub, lt in (("twin", None), ("nw0", l0), ("nw1", l1)):
            os.makedirs(os.path.join(MK, sub), exist_ok=True)
            np.savez(os.path.join(MK, sub, nm[:-4]), keep=kp, **({} if lt is None else {"late": lt}))
        WIN.append(("code", nm, ids, 0, a_ - 1)); LT[nm] = (kp, l0, l1, a_)
    def meas(name, setup):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): r = C.measure(model, WIN, name, setup)
        return r, buf.getvalue()
    T, out_t = meas("t_twin", A.q1_union(os.path.join(MK, "twin")))
    Z, out_z = meas("t_nwz", A.nw_mask(os.path.join(MK, "twin"), 1, zero=True))
    R0, out_0 = meas("t_nw0", A.nw_mask(os.path.join(MK, "nw0"), 0))
    R1, out_1 = meas("t_nw1", A.nw_mask(os.path.join(MK, "nw1"), 1))
    jt, jz = json.load(open(os.path.join(TO, "t_twin.json"))), json.load(open(os.path.join(TO, "t_nwz.json")))
    ok(all(jt[w]["tk"] == jz[w]["tk"] and jt[w]["lp"] == jz[w]["lp"] for w in jt) and "WARNING" not in out_t + out_z,
       "measure(): nwz's JSON (top-8 ids and log-probs) == the twin's, exactly, on both windows; no WARNING")
    d0 = max(np.abs(np.array(R0[w]["lp"]) - np.array(T[w]["lp"])).max() for w in T)
    ok("WARNING" not in out_0 and all("nw" in R0[w] and R0[w]["nw"]["d"] == 0 for w in R0), "measure(): nw0 runs with its record and no WARNING")
    soft(all(R0[w]["tk"] == T[w]["tk"] for w in T) and d0 < LPT, f"measure(): nw0's top-8 ids == the twin's, max |d log p| {d0:.1e}")
    good = True; gref = True; dref = 0.0
    for dom, nm, ids, off, gen in WIN:
        kp, l0, l1, a_ = LT[nm]; hh = EMB[ids % 100]; ch = chunks_of(len(ids))
        t, l = top8(mx.array(run_ref(hh, ch, kp, l1, 1))); t, l = np.array(t)[a_:len(ids) - 1], np.array(l)[a_:len(ids) - 1]
        gref &= np.array_equal(t[:, 0], np.array(R1[nm]["tk"])[:, 0]); good &= R1[nm]["a"] == a_
        dref = max(dref, float(np.abs(l - np.array(R1[nm]["lp"])).max()))
        nw = R1[nm].get("nw", {})
        good &= nw.get("d") == 1 and nw.get("terms") == int(l1.sum()) and nw.get("late") == int(l1[a_:].sum()) and nw.get("writes") == pairs(l1, ch) and "warn" not in R1[nm]
    ok(good, "measure(): nw1's res['nw'] holds d, late, terms (== the mask's count), writes (== the chunk-layer pairs); no warn")
    soft(gref and dref < LPT, f"measure(): nw1's top-1 == the reference run's on both windows (max |d log p| {dref:.1e})")
    ok(out_1.count("never-wait d 1: late terms per answer row") == 2 and "WARNING" not in out_1, "measure(): the 'never-wait' note is printed per window, no WARNING")
    dt = max(np.abs(np.array(R1[w]["lp"]) - np.array(T[w]["lp"])).max() for w in T)
    soft(dt > 20 * max(dref, LPT), f"measure(): nw1 moves the log-probs by up to {dt:.1e} against the twin (the arm is not a no-op)")
    MODE["skip"] = True
    RD, out_d = meas("t_skip", A.nw_mask(os.path.join(MK, "nw1"), 1))
    MODE["skip"] = False
    ok("WARNING gate row counts" in out_d and "WARNING never-wait used" in out_d and all("warn" in RD[w] for w in RD),
       "measure(): a prefill that skips rows trips both WARNINGs (the row count's and the never-wait one) and marks the windows")
    T2, _ = meas("t_twin2", A.q1_union(os.path.join(MK, "twin")))
    ok(all(T2[w]["tk"] == T[w]["tk"] and T2[w]["lp"] == T[w]["lp"] for w in T), "measure(): the twin after all of it == the twin before, exactly")
print("ALL OK" if not bad else f"FAILED: {len(bad)}", flush=True)
sys.exit(1 if bad else 0)
