"""rec_layer.py (Breakthrough, Mon 16 Nov 2026 21:xx; MODULE 1a milestone 1, docs/MODULE1A.md): record one real 32-row verify
forward of a 1-layer slice of the bare 125B (layer 0 = a GDN layer + its MoE, real weights), then replay it natively.
  (a) the replay on the recorded inputs reproduces the recorded logits and cache states bit for bit;
  (b) new token ids written in place into the recorded token array, the cache restored to S0 (by reference: MLX arrays are
      immutable and S0 stays referenced), then a replay == MLX's own forward on them (logits + states), bit for bit;
  plus the CPU microseconds per replay (encode; commit + wait) and the dispatch count.
Route: MTPLX's stock path by default (the slice's experts are gather_qmm, all resident; no CPU loader in the layer);
--fast adds Leap's FAST stack as cpusplit_slice.py applies it (ref2 forms, HC8, v32, f32one off).
Memory: self-guard at REC_MAX_GB (default 6; Kamen at the Mac), MLX cache limit 0 (no buffer reuse while a record lives).
Usage (BigMoE venv, cwd ~/Breakthrough): python -I -B scripts/m1/rec1a/rec_layer.py [--fast]
"""
import ctypes
import glob
import os
import sys
import threading
import time

FAST = "--fast" in sys.argv
if FAST:
    for k_, v_ in (("REF2_SB_FIRST", "1"), ("REF2_PLE_LATE", "1"), ("REF2_COMMIT_SUB", "9"), ("REF2_KEEP_CACHES", "1"),
                   ("REF2_HC8_FAST", "1"), ("REF2_SMALL_NATIVE", "1"), ("REF2_DN_KS", "1"), ("REF2_OOO_1C", "1"),
                   ("REF2_NAX_1CQ", "1"), ("REF2_NAX_COH", "1")):
        os.environ.setdefault(k_, v_)
    os.environ.pop("REF2_GDN_LAZY", None)
    os.environ.setdefault("V32_MOE1", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtlrec as R
R.load()                                                     # before mlx: the swizzles must precede MLX's encoders
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu

T0 = time.time()
LIM = float(os.environ.get("REC_MAX_GB", "6"))


class _RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"u{i}", ctypes.c_uint64) for i in range(40)]


_LIBC = ctypes.CDLL(None)


def lifetime_gb():
    ru = _RU()
    return ru.u28 / 1e9 if _LIBC.proc_pid_rusage(os.getpid(), 4, ctypes.byref(ru)) == 0 else float("nan")


def _guard():
    while True:
        time.sleep(0.5)
        if lifetime_gb() > LIM:
            print(f"[rec] SELF-GUARD: lifetime footprint {lifetime_gb():.2f} GB > {LIM} GB: exit 7", flush=True)
            os._exit(7)


threading.Thread(target=_guard, daemon=True).start()
MD = os.path.expanduser("~/BigMoE/models/bare")
NL = int(os.environ.get("REC_NL", "1"))
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
        # the PLE lives on layer 2 (config ple_layer_ids [2]); the stage index was computed with all 48 layers, so recompute
        # it for the slice (None below 3 layers: layers 0-1 do not consume the PLE)
        inner._ple_stage_idx = next((i for i, l in enumerate(inner.layers) if getattr(l, "ple", None) is not None), None)
        mx.eval(model.parameters())
    return model, cfg


mu.load_model = lazy_load_model
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
model = rt.model
inner = model.language_model.model
KINDS = "".join("G" if l.is_linear else "A" for l in inner.layers)
if FAST:
    sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
    sys.path.insert(0, os.path.expanduser("~/Leap/scripts/week98"))
    import ref2
    ref2.apply()
    for k_ in ("att_rope", "idx_prep", "hc_write", "hc_gate", "gdn_inproj_x", "hc_b", "att_single"):
        ref2.CFG[k_] = True
    ref2.dense_native_convert(model)
    import hc8r
    hc8r.hc8_install(model)
    ref2.CFG["hc8_fast"] = True
    import v32
    v32.install(ref2)
    import f32one
    f32one.install("qmm32n")
    f32one.ON[0] = False
mx.set_cache_limit(0)
print(f"[rec] MLX {mx.__version__}; slice {KINDS}; route {'FAST' if FAST else 'stock'}; "
      f"{mx.get_active_memory() / 1e9:.2f} GB active; lifetime {lifetime_gb():.2f} GB; load {time.time() - T0:.0f} s", flush=True)

text = "".join(open(f).read() for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/docs/*.md"))))
ids = list(rt.tokenizer.encode(text))
P0, N = 1024, 32
assert len(ids) >= P0 + 4 * N + 8


def states(cache):
    out = []
    for c in cache:
        if hasattr(c, "state"):
            s = c.state
            out.extend([a for a in (s if isinstance(s, (list, tuple)) else [s]) if isinstance(a, mx.array)])
    return out


def scope():
    import contextlib
    st = contextlib.ExitStack()
    st.enter_context(attention_phase("decode_verify"))
    st.enter_context(model_forward_kind("target_verify"))
    st.enter_context(exact_verify(True))
    if hasattr(model, "verify_capture_scope"):
        st.enter_context(model.verify_capture_scope())
    return st


cache = model.make_cache()
lg0 = model(mx.array([ids[:P0]]), cache=cache)
mx.eval(lg0, states(cache))
S0 = [(c, (list(c.state) if isinstance(c.state, (list, tuple)) else c.state)) for c in cache if hasattr(c, "state")]
S0_off = [(c, getattr(c, "offset", None)) for c in cache]


def reset():
    for c, s in S0:
        c.state = list(s) if isinstance(s, list) else s
    for c, o in S0_off:
        if o is not None and hasattr(c, "offset"):
            try:
                c.offset = o
            except Exception:
                pass


def bits(a):
    return np.frombuffer(memoryview(a).cast("B"), dtype=np.uint8)


# ---- record (REC_SCOPE=model: the whole 1-layer model, embed + layer + head; REC_SCOPE=layer: layer 0's call alone, its
# input hidden states (embed tiled to the HC streams) computed outside the record and written in place for new tokens)
SCOPE = os.environ.get("REC_SCOPE", "model")
MOD = sys.modules[type(inner).__module__]


def prep(tk):
    h = mx.tile(inner.embed_tokens(tk), (1, 1, inner.args.hc_count))
    m = MOD.create_ssm_mask(h, cache[inner.ssm_idx])
    mx.eval(h)
    return h, m


def fwd(tk, h, m):
    if SCOPE == "layer":
        return inner.layers[0](h, input_ids=tk, ssm_mask=m, cache=cache[0])
    return model(tk, cache=cache)


reset()
toks = mx.array([ids[P0:P0 + N]], dtype=mx.uint32)
mx.eval(toks)
h_in, m_in = prep(toks)
mx.synchronize()
R.start()
with scope():
    out = fwd(toks, h_in, m_in)
mx.eval(out, states(cache))
mx.synchronize()
R.stop()
OUTS = [out] + states(cache)                                 # the recorded outputs (their buffers are the replay's targets)
REC = [bits(a).copy() for a in OUTS]
c = R.counts()
print(f"[rec] scope {SCOPE}; record: {c}; outputs: {tuple(out.shape)} {out.dtype} + {len(OUTS) - 1} state arrays "
      f"({sum(a.nbytes for a in OUTS[1:]) / 1e6:.1f} MB); ssm_mask {'None' if m_in is None else 'set'}; lifetime {lifetime_gb():.2f} GB",
      flush=True)

# ---- (a) replay, same inputs, in the last mode of REC_MODES (the merged mode 2 when given; 9 = the ICB form)
MODES = [int(x) for x in os.environ.get("REC_MODES", "0").split(",")]
if 9 in MODES:
    n_icb, e_icb = R.build_icb()
    print(f"[rec] ICB build: {n_icb} commands{(' ERROR: ' + e_icb) if n_icb < 0 else ''}", flush=True)
    if n_icb < 0:
        MODES = [m for m in MODES if m != 9] or [2]


def do_replay(md):
    return R.replay_icb() if md == 9 else R.replay(md)

for a in OUTS:
    bits(a)[:] = 0                                           # so (a) proves the replay WROTE them
rc, st = do_replay(MODES[-1])
same = [np.array_equal(bits(a), r) for a, r in zip(OUTS, REC)]
print(f"[rec] (a) replay (mode {MODES[-1]}) on the recorded inputs: rc {rc}; outputs bitwise == recorded: {all(same)} {same[:4]}; "
      f"cpu encode {st['cpu_encode_us']:.0f} us, commit+wait {st['cpu_commit_wait_us']:.0f} us, gpu {st['gpu_us']:.0f} us, "
      f"{st['dispatches']} dispatches", flush=True)

# ---- (b) new token ids (and, for the layer scope, new input hidden states) in place, replay vs MLX's own forward.
# REC_MODES: replay modes to try per window (0 = as recorded, 1 = serial encoders, 2 = merged: one encoder, a full barrier at
# each original encoder boundary); REC_WINDOWS windows; REC_DET=1 also runs MLX's forward twice per window (determinism).
NW = int(os.environ.get("REC_WINDOWS", "3"))
DET = os.environ.get("REC_DET", "0") == "1"
print(f"[rec] op types: {R.opcounts()}", flush=True)
res = {m: [] for m in MODES}
det = []
for k in range(1, NW + 1):
    nt = ids[P0 + k * N:P0 + (k + 1) * N]
    tk = mx.array([nt], dtype=mx.uint32)
    mx.eval(tk)
    reset()
    h_new, m_new = prep(tk)
    with scope():
        ref = fwd(tk, h_new, m_new)
    mx.eval(ref, states(cache))
    REF = [bits(ref).copy()] + [bits(a).copy() for a in states(cache)]
    if DET:
        reset()
        with scope():
            ref2 = fwd(tk, h_new, m_new)
        mx.eval(ref2, states(cache))
        det.append(all(np.array_equal(x, y) for x, y in zip(REF, [bits(ref2)] + [bits(a) for a in states(cache)])))
    line = []
    for md in MODES:
        reset()
        bits(toks)[:] = bits(tk)
        if SCOPE == "layer":
            bits(h_in)[:] = bits(h_new)
        for a in OUTS:
            bits(a)[:] = 0
        rc, st = do_replay(md)
        eq = [np.array_equal(bits(a), r) for a, r in zip(OUTS, REF)]
        res[md].append((all(eq), st))
        line.append(f"mode {md}: {'OK' if all(eq) else 'DIFF out ' + str(eq[0]) + ' states ' + str(sum(eq[1:])) + '/' + str(len(eq) - 1)} "
                    f"({st['cpu_encode_us']:.0f} us)")
    print(f"[rec] (b{k}) " + "; ".join(line) + (f"; MLX twice bitwise: {det[-1]}" if DET else ""), flush=True)
for md in MODES:
    print(f"[rec] mode {md}: {sum(r for r, _ in res[md])}/{NW} windows bitwise == MLX", flush=True)
if DET:
    print(f"[rec] MLX determinism: {sum(det)}/{NW} windows bitwise between two MLX forwards", flush=True)
# ---- CPU per replay over more repeats (the same inputs; outputs checked once more at the end)
enc = []
for _ in range(int(os.environ.get("REC_REPS", "20"))):
    rc, st = do_replay(MODES[-1])
    enc.append(st["cpu_encode_us"])
ok_end = all(np.array_equal(bits(a), r) for a, r in zip(OUTS, REF))
enc.sort()
print(f"[rec] RESULT: route {'FAST' if FAST else 'stock'}, slice {KINDS}, scope {SCOPE}: (a) {all(same)}, (b) " + ", ".join(f"mode {m} {sum(r for r, _ in res[m])}/{NW}" for m in MODES) + f" "
      f"bitwise; {c['dispatches']} dispatches in {c['encoders']} encoders / {c['commits']} commits ({c['ops']} ops); CPU encode per "
      f"replay over {len(enc)} repeats: min {enc[0]:.0f}, median {enc[len(enc) // 2]:.0f}, max {enc[-1]:.0f} us "
      f"({enc[len(enc) // 2] / max(c['dispatches'], 1):.2f} us per dispatch); repeats still bitwise: {ok_end}; lifetime peak "
      f"{lifetime_gb():.2f} GB; wall {time.time() - T0:.0f} s", flush=True)

# ---- REC_RSET=1 (with mode 9): the ICB replay's per-replay CPU split, with useResources vs a residency set built once
if 9 in MODES and os.environ.get("REC_RSET", "0") == "1":
    def split(use_rs, n):
        rows = []
        for _ in range(n):
            rc_, st_ = R.replay_icb2(use_rs)
            rows.append([st_[k] for k in ("cpu_encode_us", "cb_us", "enc_us", "use_us", "exec_us")])
        a_ = np.array(rows)
        return np.median(a_, axis=0), a_[:, 0].min()
    REPS = int(os.environ.get("REC_REPS", "20"))
    med, mn = split(0, REPS)
    print(f"[rec] ICB + useResources ({R.icb_nres()} buffers): encode median {med[0]:.0f} us (min {mn:.0f}) = command buffer "
          f"{med[1]:.1f} + encoder {med[2]:.1f} + useResources {med[3]:.1f} + execute/end {med[4]:.1f} us", flush=True)
    nrs = R.build_rset()
    print(f"[rec] residency set: {nrs} allocations", flush=True)
    if nrs > 0:
        okr = []
        for k in range(1, 5):
            nt = ids[P0 + k * N:P0 + (k + 1) * N]
            tk = mx.array([nt], dtype=mx.uint32)
            mx.eval(tk)
            reset()
            h_new, m_new = prep(tk)
            with scope():
                ref = fwd(tk, h_new, m_new)
            mx.eval(ref, states(cache))
            REF = [bits(ref).copy()] + [bits(a).copy() for a in states(cache)]
            reset()
            bits(toks)[:] = bits(tk)
            if SCOPE == "layer":
                bits(h_in)[:] = bits(h_new)
            for a in OUTS:
                bits(a)[:] = 0
            rc, st = R.replay_icb2(1)
            okr.append(rc == 0 and all(np.array_equal(bits(a), r) for a, r in zip(OUTS, REF)))
        med, mn = split(1, REPS)
        print(f"[rec] ICB + residency set: {sum(okr)}/4 windows bitwise == MLX; encode median {med[0]:.0f} us (min {mn:.0f}) = "
              f"command buffer {med[1]:.1f} + encoder {med[2]:.1f} + useResources {med[3]:.1f} + execute/end {med[4]:.1f} us", flush=True)
