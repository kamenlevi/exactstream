"""MTP-STEP-0 (Leap, Wed 2026-11-04 15:2x; the orchestrator's route: what's inside a ~2.2 ms MTP step): the 125B's 4-bit
lm_head (248,320 x 2,560, real weights) at 1 row, the size an MTP chain step runs it at, and screened forms:
  full     MLX quantized_matmul [1, 2560] x lm_head^T + argmax (today's head)
  full16   the same through ref2's qmm16c (16-row tile, row padded)
  scr32k   only the first 32,768 vocab rows (a stand-in for an FR-Spec top-32K set; the cost, not the pick) + argmax
  scr32k+lse  scr32k + a logsumexp over the subset (the pick's p)
  gath2k   a gathered set of 2,048 rows (session tokens / candidates) + argmax
ms per call in 64-call dependent chains (each call's x depends on the previous output), median of 15 after 3.
Usage: [BigMoE venv] mtp_head_time.py -> ~/Leap/data/mtp_step0.txt (GPU, MEM ~1.2 GB; share lock)"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
import mlx.core as mx
from qmm16 import qmm16c

OUT = os.path.expanduser("~/Leap/data/mtp_step0.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
pre = "language_model.lm_head"
sh = mx.load(f"{D}/{wmap[pre + '.weight']}")
w, s, b = sh[pre + ".weight"], sh[pre + ".scales"], sh[pre + ".biases"]
mx.eval(w, s, b)
V = w.shape[0]
say(f"\nMTP-STEP-0 {time.strftime('%a %H:%M:%S')} ({os.environ.get('MTP0_COND', 'share lock')}): lm_head {V} x {w.shape[1] * 8}, "
    f"{(w.nbytes + s.nbytes + b.nbytes) / 1e6:.0f} MB")
sel = mx.array(np.random.default_rng(0).choice(V, 2048, replace=False).astype(np.int32))
w2, s2, b2 = mx.take(w, sel, axis=0), mx.take(s, sel, axis=0), mx.take(b, sel, axis=0)
mx.eval(w2, s2, b2)
q = lambda x, a, c, d: mx.quantized_matmul(x, a, scales=c, biases=d, transpose=True, group_size=64, bits=4)
ARMS = {
    "full": lambda x: mx.argmax(q(x, w, s, b), axis=-1),
    "full16": lambda x: mx.argmax(qmm16c(x, w, s, b, ks=4), axis=-1),
    "scr32k": lambda x: mx.argmax(q(x, w[:32768], s[:32768], b[:32768]), axis=-1),
    "scr32k+lse": lambda x: (lambda lg: (mx.argmax(lg, axis=-1), mx.logsumexp(lg.astype(mx.float32), axis=-1)))(
        q(x, w[:32768], s[:32768], b[:32768])),
    "gath2k": lambda x: mx.argmax(q(x, w2, s2, b2), axis=-1),
}
x0 = (mx.random.normal((1, 2560)) * 0.5).astype(mx.bfloat16)
mx.eval(x0)
def chain(f):
    x = x0
    for _ in range(64):
        o = f(x)
        o = o[0] if isinstance(o, tuple) else o
        x = mx.depends(x0, o)
    return x
for f in ARMS.values():
    mx.eval(chain(f))
T = {k: [] for k in ARMS}
for r in range(18):
    for k in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(chain(ARMS[k])); mx.synchronize()
        T[k].append(1e3 * (time.perf_counter() - t0) / 64)
med = {k: float(np.median(v[3:])) for k, v in T.items()}
say("  ms per head call (1 row, 64-call dependent chain): " + ", ".join(f"{k} {v:.3f}" for k, v in med.items()))
say(f"  full head: {(w.nbytes + s.nbytes + b.nbytes) / 1e6 / med['full']:.0f} GB/s effective")
