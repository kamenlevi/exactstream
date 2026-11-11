"""Exactness check that needs no fully resident model: for a few layers, compare the streamed
SwitchGLU (slot pools + remapped ids, via the synchronous path) with the same layer computed from its
full expert tensors (loaded for that layer only), bit for bit, for random inputs and expert ids.
usage: check_layer.py MODEL_DIR FRAC [LAYERS]   (works for Qwen3.6-35B and Qwen3.8-Flash-Next)
-> prints one line per layer and data/check_layer.txt"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")   # MTPLX would pre-read up to ~18 GiB of the n-gram table into page cache
for k in ("MTPLX_COMPILED_VERIFY", "MTPLX_QWEN4EXP_COMPILE", "MTPLX_COMPILE_AR_FORWARD", "MTPLX_COMPILED_TARGET_PREFIX"):
    os.environ.setdefault(k, "0")
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import numpy as np, mlx.core as mx
import mlx_lm.utils as mu
from bigmoe.stream import ExpertStreamer, PIECES
from bigmoe.gated import ext

md = os.path.expanduser(sys.argv[1]); frac = float(sys.argv[2])
cfg = __import__("json").load(open(os.path.join(md, "config.json")))
tc = cfg.get("text_config", cfg)
E_all, K, D = tc["num_experts"], tc["num_experts_per_tok"], tc["hidden_size"]
C = max(16, round(frac * E_all))
if cfg.get("model_type") == "qwen4_exp":             # Qwen3.8-Flash-Next through MTPLX
    from bigmoe.loader import load_streamed
    rt, st = load_streamed(md, C)
    model = rt.model
else:                                                # Qwen3.5/3.6 MoE through mlx-lm
    st = ExpertStreamer(md, C)
    orig = mu.load_model
    def lazy(path, *a, **kw):
        a = list(a)
        if a: a[0] = True
        else: kw["lazy"] = True
        m, c = orig(path, *a, **kw); st.install(m); mx.eval(m.parameters()); return m, c
    mu.load_model = lazy
    model, _ = __import__("mlx_lm").load(md)
    mu.load_model = orig
layers = model.language_model.model.layers
NL = len(layers)
which = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, NL // 2, NL - 1]
rng = np.random.default_rng(0)
lines = []
for L in which:
    sw = layers[L].mlp.switch_mlp
    base = type(sw).__mro__[1]                       # the original mlx-lm SwitchGLU class
    pre = st.prefix_fmt.format(L=L)
    E = st.hdr[f"{pre}.gate_proj.weight"][3][0]
    full = {}
    for proj, fld in PIECES:                           # this layer's full expert tensors, from disk
        path, off, dt, shape = st.hdr[f"{pre}.{proj}.{fld}"]
        full[(proj, fld)] = mx.load(path)[f"{pre}.{proj}.{fld}"]
    bad = 0
    for trial in range(6):
        S = [1, 3][trial % 2]
        x = mx.array(rng.standard_normal((S, D)).astype(np.float32)).astype(mx.bfloat16)
        ids = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(S)]).astype(np.uint32))
        slots = mx.array(np.array(ext.ensure(L, mx.array(np.array(ids).reshape(-1).astype(np.int32)))).reshape(np.array(ids).shape)).astype(mx.uint32)
        y_s = base.__call__(sw, x, slots)              # streamed: pools + slot ids
        saved = {k: getattr(getattr(sw, k[0]), k[1]) for k in full}
        for (proj, fld), arr in full.items(): setattr(getattr(sw, proj), fld, arr)
        y_f = base.__call__(sw, x, ids)                # reference: full tensors + expert ids
        for (proj, fld), arr in saved.items(): setattr(getattr(sw, proj), fld, arr)
        mx.eval(y_s, y_f)
        bad += int(not mx.array_equal(y_s, y_f).item())
    lines.append(f"layer {L}: {6 - bad}/6 trials bit-identical (S=1 and S=3, random ids incl. misses)")
    print(lines[-1], flush=True)
    del full
open(os.path.expanduser("~/BigMoE/data/check_layer.txt"), "a").write(f"{md} frac {frac}\n" + "\n".join(lines) + "\n")
