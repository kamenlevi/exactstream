"""ATT-PARTS (Leap, Tue 2026-11-03 15:2x; the orchestrator's routing 15:1x: the attention layer's other long-context work):
where one 16-row QSA layer call's time goes at 2K / 10K / 20K keys, with fs4 (B6's FATT16-S) as the SDPA. Knockouts in
R-72 / B6's harness (layer 3's real weights, MTPLX's Attention + indexer under REF2, 8 chained calls with trim, median of
20), each with the live engine's K-73 prep fusions (ref2 CFG att_rope + idx_prep) on and off:
  fs4        the whole call
  stub       the SDPA replaced by zeros that still depend on q, k, v and the mask (mx.depends), so the projections, RoPE,
             cache writes, the indexer and the mask build all still run: the call's "other" work
  stub-nosel the stub with the indexer's selection off (MTPLX's plain causal mask instead; R-81's patch)
SDPA in the layer = fs4 - stub; the indexer's selection + mask = stub - stub-nosel.
Usage: [BigMoE venv] att_parts.py -> ~/Leap/data/att_parts.txt (GPU, ~2 GB; share lock)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
from fatt16s import fatt16s
import ref2

OUT = os.path.expanduser("~/Leap/data/att_parts.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"ATT-PARTS {time.strftime('%a %H:%M')} ({os.environ.get('AP_COND', 'shared GPU, rough')}); chain {os.environ.get('AP_CHAIN', '8')}")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
qc = cfg["quantization"]
args = Q.TextArgs.from_dict(cfg["text_config"])
pre = "language_model.model.layers.3.self_attn."
keys = [k_ for k_ in wmap if k_.startswith(pre)]
w = {}
for f in sorted({wmap[k_] for k_ in keys}):
    sh = mx.load(f"{D}/{f}")
    w.update({k_[len(pre):]: sh[k_] for k_ in keys if wmap[k_] == f})
att = Q.Attention(args)
def qpred(p, m_):
    if not (isinstance(m_, nn.Linear) and f"{p}.scales" in w):
        return False
    c_ = qc.get(pre + p, {})
    return {"group_size": c_.get("group_size", qc["group_size"]), "bits": c_.get("bits", qc["bits"])}
nn.quantize(att, group_size=qc["group_size"], bits=qc["bits"], class_predicate=qpred)
att.load_weights(list(w.items()), strict=False)
att.eval()
mx.eval(att.parameters())
ref2.apply()
_VS = Q._verify_sdpa
MODE = ["ref2t"]
HITS = {"fs4": 0, "stub": 0, "ref2": 0}
def _vsdpa(q, k, v, *, scale, mask):
    S = q.shape[2]
    if MODE[0] in ("fs4", "stub") and mask is not None and mask.dtype == mx.bool_ and S == 16:
        HITS[MODE[0]] += 1
        if MODE[0] == "fs4":
            return fatt16s(q, k, v, scale, mask)
        return mx.depends(mx.zeros(q.shape, q.dtype), [q, k, v, mask])
    HITS["ref2"] += 1
    return _VS(q, k, v, scale=scale, mask=mask)
Q._verify_sdpa = _vsdpa
_idx_call = Q.QSAIndexer.__call__
NOSEL = [False]
def _idx(self, *a, **kw):
    out = _idx_call(self, *a, **kw)
    return None if NOSEL[0] else out
Q.QSAIndexer.__call__ = _idx
def call(c, x):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        return att(x, c)
CHAIN = int(os.environ.get("AP_CHAIN", "8"))      # 1: each call evaluated on its own (no chained cache updates)
def lbench(c, x, chain=CHAIN, n=20 if CHAIN > 1 else 60):
    def go():
        ys = []
        for _ in range(chain):
            ys.append(call(c, x)); c.trim(16)
        return ys
    for _ in range(2):
        mx.eval(go())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
ARMS = [("fs4", "fs4", False), ("stub", "stub", False), ("stub-nosel", "stub", True)]
for T0 in (2000, 10000, 20000):
    mx.random.seed(T0)
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    MODE[0], NOSEL[0] = "ref2t", False
    ref2.CFG["att_rope"] = ref2.CFG["idx_prep"] = False
    for a0 in range(0, T0, 1024):
        n_ = min(1024, T0 - a0)
        y = att(mx.random.normal((1, n_, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16); mx.eval(x16)
    for k73 in (True, False):
        ref2.CFG["att_rope"] = ref2.CFG["idx_prep"] = k73
        ms, hit = {}, {}
        for name, mode, nosel in ARMS:
            MODE[0], NOSEL[0] = mode, nosel
            for k_ in HITS:
                HITS[k_] = 0
            y = call(c, x16); mx.eval(y); c.trim(16)
            hit[name] = {k_: v_ for k_, v_ in HITS.items() if v_}
            ms[name] = lbench(c, x16)
        sd, ix = ms["fs4"] - ms["stub"], ms["stub"] - ms["stub-nosel"]
        say(f"T0 {T0} K-73 prep {'on ' if k73 else 'off'}: fs4 {ms['fs4']:.3f}, stub {ms['stub']:.3f}, stub-nosel "
            f"{ms['stub-nosel']:.3f} ms per 16-row call -> SDPA in the layer {sd:.3f}, the indexer's selection + mask "
            f"{ix:.3f}, the rest {ms['stub-nosel']:.3f}; SDPA taken {hit}")
    MODE[0], NOSEL[0] = "ref2t", False
    ref2.CFG["att_rope"] = ref2.CFG["idx_prep"] = False
    del c
    mx.clear_cache()
