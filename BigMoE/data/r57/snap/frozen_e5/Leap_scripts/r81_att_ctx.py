"""R-81 ATT-CTX (Leap, Sun 2026-11-01; shared GPU, rough, day-sized): what one 16-row check call of a real QSA attention
layer costs as the context grows (Breakthrough's fixed-part fit: +2.27 ms per K context per check; R-80 saw 3.7 ms per call
at 10K keys and ~11 ms at 14K). Layer 3's real weights under REF2 (R-72's harness), per context length T0:
  ref2t  = REF2-T as shipped (12 head-chunked SDPA calls over the key bucket, the indexer's selection mask past ~2K);
  single = + att_single (REF2c: one SDPA call; 5 kernels, head_dim 256 is unfused in MLX 0.32);
  nosel  = single with the indexer's selection off (the plain causal mask; same SDPA work): single - nosel = the
           selection's own cost.
ms per call: median of 5 synced calls after 2 warm-ups at that length (prefill fresh per length).
Usage: [BigMoE venv] r81_att_ctx.py -> ~/Leap/data/r81_att_ctx.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
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
say(f"indexer: block_topk {att.indexer.block_topk}, ratio {att.indexer.ratio}")
mx.random.seed(0)
_idx_call = Q.QSAIndexer.__call__
NOSEL = [False]
def _idx(self, *a, **kw):
    out = _idx_call(self, *a, **kw)
    return None if NOSEL[0] else out
Q.QSAIndexer.__call__ = _idx

def call(c, x16):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        y = att(x16, c)
    mx.eval(y)
    c.trim(16)

def prefill(c, T):
    for a in range(0, T, 1024):
        n = min(1024, T - a)
        y = att(mx.random.normal((1, n, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])

ARMS = {"ref2t": (False, False), "single": (True, False), "nosel": (True, True)}
for T0 in (1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000):
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    prefill(c, T0)
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16); mx.eval(x16)
    res = {}
    for name, (single, nosel) in ARMS.items():
        ref2.CFG["att_single"], NOSEL[0] = single, nosel
        for _ in range(2):
            call(c, x16)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); call(c, x16); ts.append(time.perf_counter() - t0)
        res[name] = 1e3 * float(np.median(ts))
    say(f"T0 {T0:6d}: ms per 16-row call ref2t {res['ref2t']:.2f}, single {res['single']:.2f}, single w/o selection "
        f"{res['nosel']:.2f} (selection {res['single'] - res['nosel']:+.2f}); x 12 QSA layers: ref2t {12 * res['ref2t']:.1f}, "
        f"single {12 * res['single']:.1f} ms per check")
    del c
ref2.CFG["att_single"], NOSEL[0] = False, False
open(os.path.expanduser("~/Leap/data/r81_att_ctx.txt"), "w").write("\n".join(lines) + "\n")
