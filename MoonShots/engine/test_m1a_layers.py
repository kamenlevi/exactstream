"""M1a debug: where does the engine's error vs MTPLX come from? First 4-token chunk of real text, fresh caches. For every
layer L: (a) single-layer error: the engine's layer L fed MTPLX's own input H_{L-1}; (b) accumulated error: the engine's own
trajectory. Relative error of the residual H after the layer, per layer type (gdn / qsa, +ple at layer 1). Reference MoE =
exact synchronous residency (rle/ref.py).
usage: test_m1a_layers.py   (GPU, ~10 GB, ~1 min; quiet or share lock)"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from rle.model import Engine
from rle import ref
from bigmoe.loader import load_streamed
from bigmoe.gated import ext

MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, 48, mtp=False)
model = rt.model
ref.install(ext)
rec = ref.capture_layers()
eng = Engine(model, ext)
tok = rt.tokenizer if hasattr(rt, "tokenizer") else None
text = open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:2000]
ids = mx.array([tok.encode(text)[:4]])
rel = lambda a, b: (mx.linalg.norm((a.astype(mx.float32) - b.astype(mx.float32)).reshape(-1)) /
                    mx.maximum(mx.linalg.norm(b.astype(mx.float32).reshape(-1)), 1e-30)).item()
lr = model(ids, cache=model.make_cache())
mx.eval(lr, rec)
assert len(rec) == len(eng.P), len(rec)
st1 = eng.new_state()                                        # single-layer: fresh engine state, MTPLX's inputs
trace = []
st2 = eng.new_state()
le = eng.forward(ids, st2, trace)                            # accumulated: the engine's own trajectory
mx.eval(le, trace)
print(" L  type      single-layer  accumulated", flush=True)
worst = []
for i, (hin, hout) in enumerate(rec):
    _, h = eng.layer(i, hin, ids, st1)
    mx.eval(h)
    e1, e2 = rel(h, hout), rel(trace[i][1], hout[0])
    kind = ("gdn" if eng.P[i]["gdn"] else "qsa") + ("+ple" if "ple" in eng.P[i]["layer"] else "")
    worst.append((e1, i, kind))
    print(f"{i:2d}  {kind:8s}  {e1:.2e}      {e2:.2e}", flush=True)
lr = (lr[0] if isinstance(lr, tuple) else lr)[0].astype(mx.float32)
lr, le = lr - mx.logsumexp(lr, -1, keepdims=True), le.astype(mx.float32) - mx.logsumexp(le.astype(mx.float32), -1, keepdims=True)
print(f"logits: KL {(mx.exp(lr) * (lr - le)).sum(-1).mean().item():.2e}; worst single layers: "
      + ", ".join(f"L{i} {k} {e:.1e}" for e, i, k in sorted(worst, reverse=True)[:6]), flush=True)
by = {}
for e, i, k in worst:
    by.setdefault(k, []).append(e)
print("mean single-layer error by type: " + ", ".join(f"{k} {np.mean(v):.2e} (n={len(v)})" for k, v in by.items()), flush=True)
