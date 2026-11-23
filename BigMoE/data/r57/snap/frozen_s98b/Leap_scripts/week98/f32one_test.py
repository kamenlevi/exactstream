"""f32one_test.py (Leap, Mon 16 Nov 2026 12:4x; CHECK25): the FAST (lossy-45) stack's 32-row verify on a 4-layer slice of
the 125B, for f32one (the one-pass forms) and int8d (int8 dense activations). Bits / logits only, no timing.
Slice: layers 0-3 (G G G A; layer 1 carries the PLE), resident, loaded as c3_v32.py loads it; real code tokens after a
1,024-token prefill. ref2 with the lossy-45 env's REF2_* flags (minus REF2_GDN_LAZY, which v32 refuses; it changes no bits
outside GDNB) and the 45 GB preset's forms (K-73 att_rope / idx_prep / hc_write, K-83 hc_gate / gdn_inproj_x, HCB hc_b),
then the FAST pieces as BigMoE's screen applies them: DENSE-N-RAM (dense_native_convert), HC8 (hc8r.hc8_install +
hc8_fast), FAST-TAX att_single, SVH (screened_head_install, base 98,304). The MoE block is MTPLX's own with resident
experts (ref2's experts_nax at <= 16 rows), not the streamed one-call (ooo1c_m32_check.py covers that at 32 rows).
G1's form first: the 16-row check with int8d off / on (logits, argmax) and int8 through the commit replay (keep k, probe).
Arms at 17-32 rows (each from its own prefill; the cache is rebuilt per arm):
  R    <= 16-row verifies of the same rows with full accepts between (16 + 16): the FAST 16-row reference
  S    v32 (every 17-32-row op as 2 x 16, cores chunked): == R bit for bit by construction (C3's claim, now on FAST)
  O    v32 + f32one form qmm32n: matmuls / SVH one pass (bitwise forms), GDN and attention in ONE call (SDPA fused):
       expected == S except where the one-call attention / GDN rounds differently (reported per row)
  D    v32 + f32one form dn32 (FAST kernel, not bitwise): max |logit diff| vs S, argmax agreement
  C    D + f32one.CHUNK (the chunked gated delta rule, gdn32c.py; FAST): max |logit diff| vs D / S, argmax agreement
  I    D + int8d.ON (int8 dense activations, G1's lever): max |logit diff| vs D and vs S, argmax agreement
T2 (O only): a 32-row verify, commit keep k (1 / 9 / 16 / 17 / 24 / 31), then a 4-row probe; vs the same k rows verified
as <= 16-row checks: probe logits (the one-call GDN's commit stash and replay).
usage: [BigMoE venv] f32one_test.py   (MEM ~8 GB; share lock)"""
import contextlib, glob, os, sys, time
for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
               ("REF2_HC8_FAST", "1"), ("REF2_SMALL_NATIVE", "1"), ("REF2_DN_KS", "1"), ("REF2_OOO_1C", "1"),
               ("REF2_NAX_1CQ", "1"), ("REF2_NAX_COH", "1")):
    os.environ.setdefault(k_, v_)                       # lossy-45's REF2_* (release configs/lossy-45.env) minus GDN_LAZY
os.environ.pop("REF2_GDN_LAZY", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
os.environ.setdefault("V32_MOE1", "1")
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts/week98"))
t00 = time.time()
OUT = os.path.expanduser("~/Leap/data/week98/check25/f32one_test.txt")
L_ = []


def say(s):
    print(s, flush=True)
    L_.append(s)


MD = os.path.expanduser("~/BigMoE/models/bare")
NL = int(os.environ.get("C3_NL", "4"))
orig = mu.load_model


def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a:
        a[0] = True
    else:
        kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[:NL]
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
import mtplx.cache_state as CS
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
import ref2
ref2.apply()
for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "hc_b", "att_single"):
    ref2.CFG[k_] = True
_dn = ref2.dense_native_convert(model)
import hc8r
_h8 = hc8r.hc8_install(model)
ref2.CFG["hc8_fast"] = True
_svh = ref2.screened_head_install(model.language_model.lm_head)
ref2.CFG["screened_head"] = True
import v32
v32.install(ref2)
import f32one
import int8d
say(f"F32ONE TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; slice {KINDS}; DENSE-N-RAM {_dn[0]} weights "
    f"({_dn[1]:.0f} MB); HC8 {_h8[0]} mixers; SVH base {_svh}; ref2 forms on: "
    f"{sorted(k for k, v in ref2.CFG.items() if v is True)}; {mx.get_active_memory() / 1e9:.1f} GB active")

files = sorted(glob.glob(os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts/*.py")))
ids = list(rt.tokenizer.encode("".join(open(f).read() for f in files)))
P = 1024


@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield


def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]


def prefill():
    cache = model.make_cache()
    for a in range(0, P, 1024):
        lg = model(mx.array([ids[a:min(P, a + 1024)]]), cache=cache)
        mx.eval(lg, states(cache))
    ref2.screened_head_reset()
    ref2.screened_head_add(ids[:P + 64])                # the session ids (as SessionVocab feeds SVH)
    return cache


def verify(cache, pos, n, keep=None):
    snap = CS.snapshot_untrimmable_cache_lazy(cache)
    with verify_scope():
        lg = model(mx.array([ids[pos:pos + n]]), cache=cache)
    lg = lg[0].astype(mx.float32)
    mx.eval(lg)
    ok = model.commit_verified_window(cache, snap.states, keep_tokens=n if keep is None else keep, verified_tokens=n)
    if not ok:
        raise RuntimeError(f"commit refused (n {n}, keep {keep})")
    mx.eval(states(cache))
    return lg


def chunks(n):
    out, a = [], 0
    while a < n:
        out.append(min(16, n - a)); a += out[-1]
    return out


def cmp(a, b):
    """rows bitwise equal, max |diff| over finite logits, argmax agreement"""
    fin = mx.isfinite(a) & mx.isfinite(b)
    eq = mx.all(mx.where(fin, a == b, a.astype(mx.float32).view(mx.uint32) == b.astype(mx.float32).view(mx.uint32)), axis=-1)
    d = mx.max(mx.where(fin, mx.abs(a - b), 0))
    am = mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1)
    mx.eval(eq, d, am)
    return int(np.array(eq).sum()), int(eq.size), float(d.item()), int(np.array(am).sum())


def arm(name, form=None, i8=False, chunk=False):
    if form is None:
        f32one.remove()
    else:
        f32one.install(form)
        f32one.CHUNK[0] = chunk
    if i8:
        int8d.install()
    int8d.ON[0] = i8
    f32one.COUNTS.update({k: 0 for k in f32one.COUNTS})
    int8d.COUNTS.update({k: 0 for k in int8d.COUNTS})


VIEWS = {"Reshape", "Slice", "Broadcast", "Squeeze", "ExpandDims", "Transpose", "AsStrided", "Flatten", "Unflatten"}
SCR = os.environ.get("TMPDIR", "/tmp")


def opcount(tag, n, **armkw):
    """launch proxy: non-view primitives in one n-row verify forward's graph (MTPLX's mid-loop evals off, so the graph is
    whole), for the scoreboard (Fable B / the orchestrator: launches next to bytes and ms)"""
    import re
    c = prefill()
    if armkw:
        arm(tag, **armkw)
    else:
        arm(tag)
    old = getattr(Q, "_prefill_midloop_eval_layers", None)
    if old is not None:
        Q._prefill_midloop_eval_layers = lambda rows: 0
    try:
        with verify_scope():
            lg = model(mx.array([ids[P:P + n]]), cache=c)
        outs = [lg] + [a for st in states(c) for a in (st if isinstance(st, (list, tuple)) else [st]) if isinstance(a, mx.array)]
        path = os.path.join(SCR, f"f32one_ops_{tag}.dot")
        mx.export_to_dot(path, *outs)
        ops = [o for o in re.findall(r'label\s*=\s*"([A-Za-z]+)', open(path).read()) if o != "array"]
        nv = [o for o in ops if o not in VIEWS]
        kinds = {}
        for o in nv:
            kinds[o] = kinds.get(o, 0) + 1
        top = sorted(kinds.items(), key=lambda kv: -kv[1])[:6]
        say(f"  ops {tag:4s} ({n} rows, {KINDS}): {len(nv)} non-view primitives; top {top}")
    finally:
        if old is not None:
            Q._prefill_midloop_eval_layers = old
        f32one.remove()
        int8d.ON[0] = False
    del c
    return len(nv)


allok = True
res = {}
say("LAUNCH PROXY (graph primitives of one verify forward on the slice):")
opcount("A16", 16)
opcount("S", 32)
opcount("O", 32, form="qmm32n")
opcount("D", 32, form="dn32")
opcount("C", 32, form="dn32", chunk=True)
say("G1's form: the real loop's 16-row check, int8d off vs on (no v32 / f32one action at <= 16 rows)")
c = prefill()
arm("R")
r16 = verify(c, P, 16)
del c
c = prefill()
int8d.install()
int8d.ON[0] = True
int8d.COUNTS.update({k: 0 for k in int8d.COUNTS})
i16 = verify(c, P, 16)
say(f"  int8d counts {dict((k, v) for k, v in int8d.COUNTS.items() if v and k != 'passthru')}")
del c
eq, tot, d, am = cmp(i16, r16)
say(f"    I16 vs R16: rows bitwise {eq} / {tot}, max |logit diff| {d:.4g}, argmax agrees {am} / {tot}")
for k in (1, 9, 15):                                  # int8 on through verify + commit replay: keep k then a 4-row probe,
    c1 = prefill()                                   # vs the same k rows verified as a k-row check (int8 on), same probe
    verify(c1, P, 16, keep=k)
    pv = verify(c1, P + k, 4)
    c2 = prefill()
    verify(c2, P, k)
    pr = verify(c2, P + k, 4)
    eq, tot, d, am = cmp(pv, pr)
    say(f"    int8 keep {k}: probe rows bitwise {eq} / {tot} (max |diff| {d:.3g})")
    del c1, c2
int8d.remove()
for M in (32, 24, 17):
    say(f"M {M}:")
    c = prefill()                                        # R: <= 16-row verifies, full accepts between
    arm("R")
    parts, pos = [], P
    for n in chunks(M):
        parts.append(verify(c, pos, n)); pos += n
    res["R"] = mx.concatenate(parts, axis=0)
    del c
    for nm, form, i8, ch in (("S", None, False, False), ("O", "qmm32n", False, False), ("D", "dn32", False, False),
                             ("C", "dn32", False, True), ("I", "dn32", True, False)):
        c = prefill()
        arm(nm, form, i8, ch)
        res[nm] = verify(c, P, M)
        say(f"  {nm}: f32one {dict((k, v) for k, v in f32one.COUNTS.items() if v)}; int8d "
            f"{dict((k, v) for k, v in int8d.COUNTS.items() if v and k != 'passthru')}")
        del c
    int8d.remove()
    f32one.remove()
    for a, b in (("S", "R"), ("O", "S"), ("D", "S"), ("C", "D"), ("C", "S"), ("I", "D"), ("I", "S")):
        eq, tot, d, am = cmp(res[a], res[b])
        say(f"    {a} vs {b}: rows bitwise {eq} / {tot}, max |logit diff| {d:.4g}, argmax agrees {am} / {tot}")
        if (a, b) == ("S", "R"):
            allok &= eq == tot

say("T2 commit (O = f32one qmm32n; then C = dn32 + chunk, FAST): a 32-row verify, commit keep k, then a 4-row probe; "
    "vs k rows as <= 16-row checks")
for k in (1, 9, 16, 17, 24, 31):
    c1 = prefill()
    arm("O", "qmm32n")
    verify(c1, P, 32, keep=k)
    pv = verify(c1, P + k, 4)
    f32one.remove()
    c2 = prefill()
    pos = P
    for n in chunks(k):
        verify(c2, pos, n); pos += n
    pr = verify(c2, P + k, 4)
    eq, tot, d, am = cmp(pv, pr)
    say(f"  keep {k}: probe rows bitwise {eq} / {tot} (max |diff| {d:.3g}, argmax {am} / {tot})")
    del c1, c2
for k in (9, 24):                                    # the chunked form through verify + commit replay (FAST: argmax / diff)
    c1 = prefill()
    arm("C", "dn32", False, True)
    verify(c1, P, 32, keep=k)
    pv = verify(c1, P + k, 4)
    f32one.remove()
    c2 = prefill()
    pos = P
    for n in chunks(k):
        verify(c2, pos, n); pos += n
    pr = verify(c2, P + k, 4)
    eq, tot, d, am = cmp(pv, pr)
    say(f"  C keep {k}: probe rows bitwise {eq} / {tot} (FAST: max |diff| {d:.3g}, argmax {am} / {tot})")
    del c1, c2
say(f"S == R (v32 on the FAST stack, every row): {'ALL BITWISE' if allok else 'NOT BITWISE'}")
say(f"done in {time.time() - t00:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "a").write("\n".join(L_) + "\n\n")
