"""B5 (Leap, Sun 2026-10-25; NewIdeas' order): how long a 16-row check leaves the GPU idle waiting on the CPU (Python
graph build + MLX encode). Measured on a resident 12-layer slice of the real 125B through MTPLX's own forward (read
only). Layers 0-11 = 9 GDN (incl. the PLE layer 2) + 3 QSA; the full model is 4 such groups. The loader uses BigMoE's
lazy trick: mlx_lm's load_model forced lazy, and the layer list cut to 12 before any weight is evaluated (~18.5 GB
resident). Each check is 16 real tokens of a code prompt after a ~600-token prefill, under the verify scopes that
MTPLX's generation loop opens (decode_verify phase, target_verify, exact verify, capture). Variants (fresh prefill
each, same tokens):
  A  eager, as the engine runs 16 rows: Python build (GPU idle), then mx.eval of logits + cache states
  B  the same forward with mx.async_eval every 4 layers (the build overlaps the GPU); logits must equal A's bitwise
  C  MTPLX's compiled GDN runs (S <= 4 in the engine) forced at S 16
Usage: [BigMoE venv] b5_slice.py [time|trace] -> ~/Leap/data/b5_slice.txt (trace: A only, data/b5_trace_wall.txt)"""
import contextlib, os, sys, time
import numpy as np
import mlx.core as mx
import mlx_lm.utils as mu

MD = os.path.expanduser("~/BigMoE/models/bare")
NL = 12
MODE = sys.argv[1] if len(sys.argv) > 1 else "time"
N_CHK, N_WARM = (20, 3) if MODE == "time" else (10, 3)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

orig = mu.load_model
def lazy_load_model(path, *a, **kw):
    a = list(a)
    if a: a[0] = True
    else: kw["lazy"] = True
    model, cfg = orig(path, *a, **kw)
    if hasattr(model, "language_model"):
        inner = model.language_model.model
        inner.layers = inner.layers[:NL]               # layers 12-47 are never read
        mx.eval(model.parameters())
    return model, cfg
mu.load_model = lazy_load_model
t_load = time.time()
try:
    from mtplx.runtime import load
    rt = load(MD, mtp=False)
finally:
    mu.load_model = orig
import mtplx.models.qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

model = rt.model
tm = model.language_model
inner = tm.model
say(f"B5 slice ({MODE}): {len(inner.layers)} layers resident, types "
    f"{''.join('G' if l.is_linear else 'A' for l in inner.layers)}, {mx.get_active_memory() / 1e9:.1f} GB active, "
    f"loaded in {time.time() - t_load:.0f} s")

text = "".join(open(os.path.expanduser(f"/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts/{f}")).read() for f in ("b1_energy.py", "par32.py", "gdn31.py"))
ids = list(rt.tokenizer.encode(text))
P = 600
assert len(ids) >= P + 16 * (N_CHK + N_WARM), len(ids)

@contextlib.contextmanager
def verify_scope():
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True), \
            model.verify_capture_scope():
        yield

def states(cache):
    return [c.state for c in cache if hasattr(c, "state")]

def _body(x, cache, layers_fn):
    """Qwen4ExpTextModel._forward + TextModel's head, with the layer loop swapped"""
    h = inner.embed_tokens(x)
    ssm_mask = Q.create_ssm_mask(h, cache[inner.ssm_idx])
    if inner._ple_stage_idx is not None and Q._COMPILED_VERIFY_PLE.get() is None:
        ple = inner.layers[inner._ple_stage_idx].ple
        ple.ple_embedding.stage(x, cache[inner._ple_stage_idx], ple.NGRAM_IDX)
    h = mx.tile(h, (1, 1, inner.args.hc_count))
    h = layers_fn(h, x, cache, ssm_mask)
    inner._last_widened = h
    out = inner.hyper_connection_mixer(h)
    return inner.embed_tokens.as_linear(out) if tm.args.tie_word_embeddings else tm.lm_head(out)

def body(x, cache, layers_fn):
    if Q.qwen4_opdiet_enabled("rope"):
        with Q._rope_table_scope():
            return _body(x, cache, layers_fn)
    return _body(x, cache, layers_fn)

def layers_mid(h, x, cache, ssm_mask, every=4):
    capture = Q._VERIFY_CAPTURE.get()
    last = len(inner.layers) - 1
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if capture and c is not None and getattr(layer, "ple", None) is not None:
            c._mtplx_verify_ple = (h, x)
        h = layer(h, input_ids=x, ssm_mask=ssm_mask, cache=c)
        if i < last and (i + 1) % every == 0:
            mx.async_eval(h, *Q._midloop_state_arrays(cache[i + 1 - every:i + 1]))
    return h

def layers_comp(h, x, cache, ssm_mask):
    return inner._decode_layers_compiled(h, x, cache)

def run(variant):
    cache = model.make_cache()
    lg = model(mx.array([ids[:P]]), cache=cache)
    mx.eval(lg, states(cache))
    b, e, w, outs, stamps = [], [], [], [], []
    for k in range(N_CHK + N_WARM):
        x = mx.array([ids[P + 16 * k:P + 16 * (k + 1)]])
        mx.eval(x)
        mx.synchronize()
        if MODE == "trace" and k == N_WARM:
            time.sleep(1.0)                              # a 1 s GPU gap marks the measured block in the trace
        t0 = time.perf_counter()
        with verify_scope():
            if variant == "A":
                lg = model(x, cache=cache)
            else:
                lg = body(x, cache, layers_mid if variant == "B" else layers_comp)
        t1 = time.perf_counter()
        mx.eval(lg, states(cache))
        t2 = time.perf_counter()
        model.clear_verify_capture(cache)
        outs.append(np.array(lg[0].astype(mx.float32)))
        if k >= N_WARM:
            b.append(t1 - t0); e.append(t2 - t1); w.append(t2 - t0); stamps.append((t0, t2))
    if MODE == "trace":
        time.sleep(1.0)
    return [1e3 * np.array(v) for v in (b, e, w)], outs, stamps

med = lambda v: float(np.median(v))
res = {}
for v in (("A", "B", "C") if MODE == "time" else ("A",)):
    try:
        (b, e, w), outs, stamps = run(v)
    except Exception as ex:
        say(f"  {v}: not runnable ({type(ex).__name__}: {str(ex)[:160]})")
        continue
    res[v] = outs
    tail = ""
    if v != "A" and "A" in res:
        d = max(float(np.abs(o - r).max()) for o, r in zip(outs, res["A"]))
        tail = f"; logits vs A: max |diff| {d:.3g}" + (" (bitwise)" if d == 0 else "")
    say(f"  {v}: build {med(b):.2f} ms, eval {med(e):.2f} ms, wall {med(w):.2f} ms [min {w.min():.2f}] per 12-layer "
        f"check -> x4 {4 * med(w):.1f} ms per 48-layer check{tail}")
    if MODE == "trace":
        open(os.path.expanduser("~/Leap/data/b5_trace_wall.txt"), "w").write(
            "\n".join(f"{a:.6f} {z:.6f} {1e3 * (z - a):.3f}" for a, z in stamps) + "\n")
if MODE == "time":
    open(os.path.expanduser("~/Leap/data/b5_slice.txt"), "w").write("\n".join(lines) + "\n")
