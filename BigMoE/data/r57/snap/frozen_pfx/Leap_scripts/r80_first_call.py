"""R-80 FIRST-CALL probe (Leap, Sun 2026-11-01; shared GPU, day-sized; Breakthrough's FIXED-PART result 12:16: the first
check after a 9-12K-token prefill costs 1.6-3.4 s vs 82-191 ms steady, the second ~290-380 ms; candidates: the prefill's
lazy tail, cold pools, one-time buffers / compiles at the new context). This probes the compile candidate on one real
QSA attention layer (R-72's harness, layer 3's real weights, under REF2) in a FRESH process:
- prefill to T0 (short warm-up first at 700 keys, as a harness warm-up would), then 16-row check calls at T0 = 10,000
  keys (the indexer selects blocks: kernels a short warm-up never ran), each call timed with its own eval;
- call 1 vs calls 2-6 at that length, then a NEW length (+1 bucket) and a third.
If call 1 at a new length is >> later calls (tens / hundreds of ms per layer, x 12 QSA layers + the GDN / MoE side),
one-time kernel builds are a real part of the first-check excess; if not, it is the prefill tail or cold pools.
Usage: [BigMoE venv] r80_first_call.py -> ~/Leap/data/r80_first_call.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfx/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
qc = cfg["quantization"]
args = Q.TextArgs.from_dict(cfg["text_config"])
pre = "language_model.model.layers.3.self_attn."
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}")
    w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
att = Q.Attention(args)
def qpred(p, m):
    if not (isinstance(m, nn.Linear) and f"{p}.scales" in w):
        return False
    c = qc.get(pre + p, {})
    return {"group_size": c.get("group_size", qc["group_size"]), "bits": c.get("bits", qc["bits"])}
nn.quantize(att, group_size=qc["group_size"], bits=qc["bits"], class_predicate=qpred)
att.load_weights(list(w.items()), strict=False)
att.eval()
mx.eval(att.parameters())
ref2.apply()
mx.random.seed(0)

def check_call(c, x16):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        y = att(x16, c)
    mx.eval(y)
    return y

def prefill(c, T):
    for a in range(0, T, 1024):
        n = min(1024, T - a)
        y = att(mx.random.normal((1, n, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])

# warm-up at a short context (as a harness warm-up prompt would): checks at 700 keys
c0 = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
prefill(c0, 700)
for i in range(4):
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16); mx.eval(x16)
    t0 = time.perf_counter(); check_call(c0, x16); dt = time.perf_counter() - t0
    if i in (0, 3):
        say(f"warm-up at 700 keys: call {i + 1} {1e3 * dt:.1f} ms")
del c0
for T0 in (10000, 10600, 14000):
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    t0 = time.perf_counter(); prefill(c, T0); tp = time.perf_counter() - t0
    ts = []
    for i in range(6):
        x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16); mx.eval(x16)
        t0 = time.perf_counter(); check_call(c, x16); ts.append(time.perf_counter() - t0)
    say(f"T0 {T0} keys (prefill {tp:.1f} s): 16-row check calls {', '.join(f'{1e3 * t:.1f}' for t in ts)} ms; call 1 / "
        f"median of 2-6 = {ts[0] / np.median(ts[1:]):.1f}x")
say(f"counts {ref2.counts()}")
open(os.path.expanduser("~/Leap/data/r80_first_call.txt"), "w").write("\n".join(lines) + "\n")
