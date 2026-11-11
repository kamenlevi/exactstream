"""gdnb_lazy_check.py (Fable B, ~/FableGod/fableb/build_1105; written Thu 2026-11-05, NOT RUN: it needs the GPU): the
exactness self-check and the timing of the lazy GDN commit (gdnbody_lazy.patch + ref2_lazy.patch), on Leap's
gdnb_wire_check.py harness (real layer 4, today's exact config: K-83's gdn_inproj_x, DENSE-N-RAM, the verify scope).
Copy to /Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts/ after applying the two patches.
 (1) the patch leaves today's path alone: with gdn_lazy_commit off, gdn_body on == off (y, conv state, recurrent state)
     bitwise at S 16 / 8 / 1 (gdnb_wire_check's own test, on the patched files).
 (2) the kernel's contract with nothing pending: gdn_body_lazy == gdn_body on y / q / k / v / conv state, its committed
     state == its input state, its pack == q | k | v | a | b, all bitwise.
 (3) the replay: for keep in 1, 2, 9, 15, 16, a second call with pending = the first call's pack[:keep] must write a
     committed state bitwise == MTPLX's commit replay (mlx_lm gated_delta_update, its kernel, on the kept rows from the
     same state), and its y must == gdn_body's on that replayed state. NSG 8 and 16, G3 off and on.
 (4) the protocol through ref2 on the real module: one random stream of checks (S 16 / 8 / 1 / others; partial and full
     accepts; rollback + the same rows again; an abandoned check; a 37-row prefill chunk in the middle), run twice from
     the same cache, lazy off (today) and on, with MTPLX's own snapshot, rollback and commit. After every event: y, conv
     state and the TRUE recurrent state (ref2.gdn_lazy_state) bitwise equal.
 (5) ms per cycle over 36 chained layers (forward S 16 + commit keep K, the commit's work pulled by the next forward),
     today vs lazy, K = 9 (partial) and K = 16 (full accept), median of 9, arms alternated.
Lines (pre-registered Thu 5 Nov ~19:0x, before any run):
  PASS    = (1)-(4) every comparison bitwise AND (5) saves >= 0.8 ms per partial-accept cycle AND costs <= 0.8 ms more per
            full-accept cycle.
  KILL    = (5) saves < 0.4 ms per partial-accept cycle (whatever the bits).
  BITS    = (3) or (4) not bitwise but (5) passes: an exact-track KILL; the FAST track may still take it (a new rounding
            of the committed state), after trying the fallback: gdnbody._lz_src keeping the output dot alive in B0.
  BETWEEN = anything else.
usage: [BigMoE venv] gdnb_lazy_check.py -> ~/Leap/data/gdnb_lazy_check.txt (GPU, MEM ~1.5 GB; share lock for the bits,
a quiet lock for (5))"""
import os, sys, json, time, types
os.environ.setdefault("MTPLX_FUSED_GDN_CONVNORM", "1"); os.environ.setdefault("MTPLX_FUSED_CONVNORM_VERIFY", "1")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
import gdnbody as GB
from mtplx import cache_state as CS
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import gated_delta_update
from mtplx.attention_context import attention_phase, model_forward_kind, exact_verify
from mlx.utils import tree_flatten

assert hasattr(GB, "gdn_body_lazy") and "gdn_lazy_commit" in ref2.CFG, "apply gdnbody_lazy.patch and ref2_lazy.patch first"
OUT = os.path.expanduser("~/Leap/data/gdnb_lazy_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nGDN lazy commit check {time.strftime('%a %H:%M:%S')}")
# ---- the harness: gdnb_wire_check.py's, verbatim (real layer 4, today's exact config)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
args = Q.TextArgs.from_dict(json.load(open(f"{D}/config.json"))["text_config"])
LAYER = 4
pre = f"language_model.model.layers.{LAYER}.linear_attn."
class Box(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Q.GatedDeltaNet(args)
box = Box()
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}"); w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
mod_in = {p_[:-len(".weight")]: int(a_.shape[-1]) for p_, a_ in tree_flatten(box.g.parameters()) if p_.endswith(".weight") and a_.ndim >= 2}
qmap = {}
for m_, in_f in mod_in.items():
    wq, sq = w.get(f"{m_}.weight"), w.get(f"{m_}.scales")
    if wq is not None and sq is not None and wq.dtype == mx.uint32:
        qmap[m_] = {"bits": int(wq.shape[-1]) * 32 // in_f, "group_size": in_f // int(sq.shape[-1])}
nn.quantize(box.g, group_size=64, bits=4, class_predicate=lambda p_, m: qmap.get(p_, False))
box.g.load_weights(list(w.items()), strict=False)
box.g.eval()
mx.eval(box.parameters())
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x"):
    ref2.CFG[k_] = True
n_conv, mb = ref2.dense_native_convert(box)
say(f"layer {LAYER} GDN, quantized modules {sorted(qmap)}; DENSE-N-RAM converted {n_conv} ({mb:.0f} MB)")
g = box.g
mx.random.seed(3)
def fresh():
    c = ArraysCache(size=2)
    x0 = (mx.random.normal((1, 37, 2560)) * 0.7).astype(mx.bfloat16)
    g(x0, None, c)                                        # prefill fills the conv / recurrent caches
    mx.eval(c[0], c[1])
    return c
def scope():
    import contextlib
    st = contextlib.ExitStack()
    st.enter_context(attention_phase("decode_verify")); st.enter_context(model_forward_kind("target_verify"))
    st.enter_context(exact_verify(True))
    return st
bits = lambda a: np.array(a.view(mx.uint16)) if a.dtype == mx.bfloat16 else np.array(a)
same = lambda a, b: a.shape == b.shape and np.array_equal(bits(a), bits(b))
rnd = lambda S: (mx.random.normal((1, S, 2560)) * 0.7).astype(mx.bfloat16)
ok = True
base_state = fresh()
snap0 = (base_state[0], base_state[1])

# ---- (1) lazy off: gdn_body on == off (the patched files leave today's path alone)
ref2.CFG["gdn_lazy_commit"] = False
for S in (16, 8, 1):
    x = rnd(S)
    outs = {}
    for arm in ("off", "nsg8", "nsg16"):
        ref2.CFG["gdn_body"] = arm != "off"
        GB.NSG_DEFAULT[0] = 16 if arm == "nsg16" else 8
        c = ArraysCache(size=2); c[0], c[1] = snap0
        with scope():
            y = g(x, None, c)
        mx.eval(y, c[0], c[1])
        outs[arm] = (y, c[0], c[1], len(c.state))
    e1 = all(all(same(a, b) for a, b in zip(outs[k][:3], outs["off"][:3])) and outs[k][3] == 2 for k in ("nsg8", "nsg16"))
    ok &= e1
    say(f"(1) S {S:2d}, lazy off: gdn_body NSG 8 / 16 == off (y, conv state, recurrent state) bitwise, 2-leaf state: {e1}")

# ---- (2) + (3) the kernel itself, on the body's own inputs
def body_inputs(x, conv, S):
    """ref2._gdn_body_call's inputs for one call (in_proj through the layer's own fused projection)"""
    fused_in = getattr(g, "in_proj_fused", None)
    if fused_in is not None:
        qkv, z, b, a = fused_in(x)
    else:
        qkv, z, b, a = g.in_proj_qkv(x), g.in_proj_z(x), g.in_proj_b(x), g.in_proj_a(x)
    z = z.reshape(1, S, g.num_v_heads, g.head_v_dim)
    return qkv, z, a, b
ref2.CFG["gdn_body"] = True
with scope():
    g(rnd(16), None, fresh())                             # builds R-83's fused in_proj once
gx = ref2._gdnx()
kw = lambda conv, st: dict(conv_state=conv, conv_w=g.conv1d.weight, A_log=g.A_log, dt_bias=g.dt_bias, state=st,
                           norm_w=g.norm.weight, eps=g.norm.eps, scale_bf16=gx._scale())
for nsg in (8, 16):
    for g3 in (False, True):
        tag = f"NSG {nsg:2d} G3 {int(g3)}"
        x1, x2 = rnd(16), rnd(16)
        with scope():
            qkv1, z1, a1, b1 = body_inputs(x1, snap0[0], 16)
            ref = GB.gdn_body(qkv1, z1, a1, b1, **kw(snap0[0], snap0[1]), nsg=nsg, g3=g3)       # (y, q, k, v, cs, END state)
            lz = GB.gdn_body_lazy(qkv1, z1, a1, b1, **kw(snap0[0], snap0[1]), nsg=nsg, g3=g3)   # (y, q, k, v, cs, COMMITTED, pack)
            mx.eval(ref, lz)
        pack_ref = mx.concatenate([lz[1].reshape(1, 16, -1), lz[2].reshape(1, 16, -1), lz[3].reshape(1, 16, -1), a1, b1], axis=-1)
        e2 = (all(same(lz[i], ref[i]) for i in range(5)) and same(lz[5], snap0[1]) and same(lz[6], pack_ref))
        ok &= e2
        say(f"(2) {tag}: nothing pending: y / q / k / v / conv == gdn_body, committed == the input state, pack == q|k|v|a|b: {e2}")
        for keep in (1, 2, 9, 15, 16):
            with scope():
                # MTPLX's commit replay on the kept rows from the pre-check state (commit_verified_window's own call)
                _, st_ref = gated_delta_update(ref[1][:, :keep], ref[2][:, :keep], ref[3][:, :keep], a1[:, :keep], b1[:, :keep],
                                               g.A_log, g.dt_bias, snap0[1], None, use_kernel=True)
                conv_ref = mx.contiguous(mx.concatenate([snap0[0], qkv1[:, :keep]], axis=1)[:, -3:, :])
                qkv2, z2, a2, b2 = body_inputs(x2, conv_ref, 16)
                r2 = GB.gdn_body(qkv2, z2, a2, b2, **kw(conv_ref, st_ref), nsg=nsg, g3=g3)
                l2 = GB.gdn_body_lazy(qkv2, z2, a2, b2, **kw(conv_ref, lz[5]), pending=lz[6][:, :keep], nsg=nsg, g3=g3)
                mx.eval(st_ref, r2, l2)
            e3 = same(l2[5], st_ref) and all(same(l2[i], r2[i]) for i in range(5))
            ok &= e3
            say(f"(3) {tag}, keep {keep:2d}: committed state == MTPLX's commit replay, y / q / k / v / conv == today's next check: {e3}")

# ---- (4) the protocol through ref2 on the real module, MTPLX's own snapshot / rollback / commit
fake = types.SimpleNamespace(layers=[types.SimpleNamespace(linear_attn=g, ple=None)],
                             _refuse_commit=lambda i, why: (_ for _ in ()).throw(AssertionError(f"commit refused: {why}")))
commit = Q.Qwen4ExpTextModel.commit_verified_window     # ref2's patched hook (apply() installed it)
def stream(lazy, seed=11, n=40):
    ref2.CFG["gdn_lazy_commit"] = lazy
    ref2.CFG["gdn_body"] = True
    GB.NSG_DEFAULT[0] = 16
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    c = ArraysCache(size=2); c[0], c[1] = snap0
    cache, log = [c], []
    tok = Q._VERIFY_CAPTURE.set(True)                     # the engine's verify_capture_scope: MTPLX's commit needs the stash
    try:
        for ev in range(n):
            if ev == n // 2:                              # a prefill chunk in the middle (MTPLX's own forward: lazy flushes)
                y = g(rnd(37), None, c); mx.eval(y, c.state)
                log.append(("prefill", y, c.state[0], ref2.gdn_lazy_state(fake, cache)[0])); continue
            S = int(rng.choice([16, 16, 16, 8, 1, int(rng.integers(2, 16))]))
            x = rnd(S)
            sn = CS.snapshot_untrimmable_cache_lazy(cache)
            with scope():
                y = g(x, None, c)
            mx.eval(y, c.state)
            r = rng.random()
            if r < 0.1:                                   # abandoned
                CS.rollback_after_verify(cache, sn, S)
            else:
                if r < 0.25:                              # rolled back, the same rows again
                    CS.rollback_after_verify(cache, sn, S)
                    sn = CS.snapshot_untrimmable_cache_lazy(cache)
                    with scope():
                        y2 = g(x, None, c)
                    mx.eval(y2, c.state)
                    assert same(y, y2), "the re-forward after a rollback changed y"
                keep = S if rng.random() < 0.25 else int(rng.integers(1, S + 1))
                assert commit(fake, cache, sn.states, keep_tokens=keep, verified_tokens=S)
            st = ref2.gdn_lazy_state(fake, cache)[0]
            mx.eval(c.state, st)
            log.append((S, y, c.state[0], st))
    finally:
        Q._VERIFY_CAPTURE.reset(tok)
    return log, ref2._lz_form(c)
c0 = dict(ref2.COUNTS)
la, fa = stream(False)
lb, fb = stream(True)
e4 = len(la) == len(lb) and all(a[0] == b[0] and all(same(a[i], b[i]) for i in (1, 2, 3)) for a, b in zip(la, lb))
took = ref2.COUNTS["gdn_lazy"] - c0["gdn_lazy"]
ok &= e4 and took > 0
say(f"(4) protocol stream, {len(la)} events: lazy on == off after every event (y, conv state, TRUE recurrent state) bitwise: {e4}; "
    f"lazy body calls {took}, pending rows replayed {ref2.COUNTS['gdn_lazy_rows'] - c0['gdn_lazy_rows']}, flushes "
    f"{ref2.COUNTS['gdn_lazy_flush'] - c0['gdn_lazy_flush']}; final form off {fa} / on {fb}")

# ---- (5) ms per cycle over 36 chained layers: forward S 16 + commit keep K; the commit's work is pulled by the next forward
def cycles(lazy, K, L=36, n=3):
    """n cycles on L caches; returns the outputs to eval. Cycle = snapshot, forward 16 rows, commit keep K."""
    ref2.CFG["gdn_lazy_commit"] = lazy
    cs = []
    for _ in range(L):
        c = ArraysCache(size=2); c[0], c[1] = snap0; cs.append(c)
    tok = Q._VERIFY_CAPTURE.set(True)
    try:
        for _ in range(n):
            x = rnd(16)
            sn = CS.snapshot_untrimmable_cache_lazy(cs)
            with scope():
                for c in cs:
                    x = x + g(x, None, c)
            fk = types.SimpleNamespace(layers=[fake.layers[0]] * L, _refuse_commit=fake._refuse_commit)
            commit(fk, cs, sn.states, keep_tokens=K, verified_tokens=16)
            mx.eval(x)                                    # the forward only: the commit's replay stays lazy, as in the engine
    finally:
        Q._VERIFY_CAPTURE.reset(tok)
    return [x] + [c.state for c in cs]
T = {(lz_, K): [] for lz_ in (False, True) for K in (9, 16)}
for r in range(9):
    for key in (list(T) if r % 2 == 0 else list(T)[::-1]):
        mx.eval(cycles(*key, n=2))                        # warm
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(cycles(*key, n=4)); mx.synchronize()
        T[key].append((time.perf_counter() - t0) / 4)
m = {k: 1e3 * float(np.median(v)) for k, v in T.items()}
save9, cost16 = m[(False, 9)] - m[(True, 9)], m[(True, 16)] - m[(False, 16)]
say(f"(5) ms per cycle, 36 layers, S 16 (median of 9, 4 cycles each): keep 9 today {m[(False, 9)]:.2f} / lazy {m[(True, 9)]:.2f} "
    f"(saving {save9:+.2f}); keep 16 today {m[(False, 16)]:.2f} / lazy {m[(True, 16)]:.2f} (cost {cost16:+.2f}); "
    f"at 24 % full accepts {0.76 * save9 - 0.24 * cost16:+.2f} ms per check")
ref2.CFG["gdn_lazy_commit"] = False; ref2.CFG["gdn_body"] = False; GB.NSG_DEFAULT[0] = 8
st_ = ref2.self_test()
say(f"(6) ref2.self_test: {'PASS' if all(bool(v) for v in st_.values()) else 'FAIL ' + str(st_)}")
ref2.remove()
verdict = ("PASS" if ok and save9 >= 0.8 and cost16 <= 0.8 else "KILL (saving < 0.4 ms)" if save9 < 0.4
           else "BITS (fast enough, not bitwise: exact-track KILL, try the fallback)" if not ok and save9 >= 0.8 and cost16 <= 0.8
           else "BETWEEN")
say(f"GDN lazy commit check: {verdict}")
