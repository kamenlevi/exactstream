"""R-72 ATT-ANAT (Leap, Sun 2026-11-01; shared GPU, real weights, day-sized ~1 GB): one real QSA attention layer of the bare
125B (MTPLX's Attention + indexer, layer 3's 4-/8-bit weights from the shards) under REF2, as a 16-row check sees it.
1. Kernel census of one 16-row call (kernel-launching primitives, views excluded; R-65's method) per arm:
   ref2t   = REF2-T as shipped (12 head-chunked SDPA calls, stock RoPE chain);
   single  = + CFG att_single (REF2c);
   rope    = + MTPLX's one-dispatch QSA RoPE (mtplx.kernels.qwen4_m4_rope.rope_qk, "bit-exact against the stock
             spelling"; MTPLX wires it for 1-8 rows only, here at REF2's 16) on top of single;
   rope0   = rope_qk on top of ref2t (the exact-safe form: no reference change);
   idx0    = ref2 CFG idx_prep on top of ref2t (MTPLX's shipped indexer prep kernel: norm + RoPE, queries + pooled keys);
   exact   = att_rope + idx_prep on ref2t; singleX = the same on single (ref2 switches since R-72's first run).
2. Bits: each exact-safe arm == its base (y and the KV / raw / pooled cache rows), prefill 383-4100 keys (the indexer
   selects blocks past ~2048).
3. ms per 16-row call (8 chained calls per eval, best of 3 medians; shared GPU, rough).
Usage: [BigMoE venv] r72_att_anat.py -> ~/Leap/data/r72_att_anat.txt"""
import os, sys, json, re, tempfile, collections, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
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
L = int(os.environ.get("R72_LAYER", "3"))
pre = f"language_model.model.layers.{L}.self_attn."
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
say(f"layer {L}: {len(w)} tensors; indexer {att.indexer is not None}; heads {att.n_heads}/{att.n_kv_heads} x {att.head_dim}")
ref2.apply()

ARMS = {"ref2t": (False, False, False), "rope0": (False, True, False), "idx0": (False, False, True),
        "exact": (False, True, True), "single": (True, False, False), "singleX": (True, True, True)}
SAME = (("rope0", "ref2t"), ("idx0", "ref2t"), ("exact", "ref2t"), ("singleX", "single"))
def set_arm(a):
    ref2.CFG["att_single"], ref2.CFG["att_rope"], ref2.CFG["idx_prep"] = ARMS[a]

VIEWS = {"Reshape", "Broadcast", "BroadcastAxes", "Transpose", "ExpandDims", "Squeeze", "Slice", "AsStrided",
         "Flatten", "Unflatten", "StopGradient", "Depends", "Split"}
def census(*arrs):
    cnt = collections.Counter()
    fd, p = tempfile.mkstemp(suffix=".dot"); os.close(fd)
    try:
        mx.export_to_dot(p, *arrs)
        for line in open(p):
            m = re.search(r'label\s*=\s*"([A-Z][A-Za-z0-9_]*)', line)
            if m:
                cnt[m.group(1)] += 1
    finally:
        os.remove(p)
    return cnt

def fresh(T0, seed):
    mx.random.seed(seed)
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    x0 = mx.random.normal((1, T0, 2560)).astype(mx.bfloat16)
    set_arm("ref2t")
    y0 = att(x0, c)                                   # prefill (> 16 rows: MLX's own calls in every arm)
    mx.eval(y0)
    mx.eval([a for a in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if a is not None])
    return c

def check(c, x16, arm, keep=False):
    set_arm(arm)
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        y = att(x16, c)
    return y

def run_trim(c, x16, arm):
    y = check(c, x16, arm)
    mx.eval(y)
    c.trim(16)
    return y

# 1. census + 2. bits
fails = 0
def cstate(c):
    T = c.offset
    arrs = [c.kv.keys[:, :, :T], c.kv.values[:, :, :T], c.raw_keys[:, :T]]
    if c.pooled is not None and c.pooled_len:
        arrs.append(c.pooled[:, :c.pooled_len])
    mx.eval(arrs)
    return [np.array(a.view(mx.uint16)) for a in arrs]
for T0, seed in ((700, 0), (383, 1), (1100, 2), (2600, 3), (4100, 4)):
    c = fresh(T0, seed)
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
    mx.eval(x16)
    out, cen, st = {}, {}, {}
    for arm in ARMS:
        y = check(c, x16, arm)
        cen[arm] = census(y)
        mx.eval(y)
        out[arm] = np.array(y.view(mx.uint16))
        st[arm] = cstate(c)
        c.trim(16)
    nk = {a: sum(v for k_, v in cen[a].items() if k_ not in VIEWS) for a in ARMS}
    eq = {f"{a}=={b}": bool((out[a] == out[b]).all()) and all((p == q).all() for p, q in zip(st[a], st[b]))
          for a, b in SAME}
    fails += sum(not v for v in eq.values())
    say(f"T0 {T0}: kernels per 16-row call: " + ", ".join(f"{a} {nk[a]}" for a in ARMS)
        + f"; bits (y + KV / raw / pooled cache) {eq}; single vs ref2t y differ "
        f"{float((out['single'] != out['ref2t']).mean()):.3f}")
    if T0 in (700, 4100):
        for a in ("ref2t", "exact", "singleX"):
            say(f"  {a}: " + ", ".join(f"{k_} {v}" for k_, v in cen[a].most_common() if k_ not in VIEWS))
say(f"counts {ref2.counts()}")
say(f"exact-safe attention fusions (att_rope, idx_prep) bits: {'PASS' if fails == 0 else f'FAIL ({fails})'}")

# 3. speed (8 chained 16-row calls per eval; each call trims afterwards)
def bench(c, arm, chain=8, n=15):
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
    def go():
        ys = []
        for _ in range(chain):
            ys.append(check(c, x16, arm))
            mx.eval(ys[-1])
            c.trim(16)
        return ys
    for _ in range(2):
        go()
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter()
        ys = []
        for _ in range(chain):
            ys.append(check(c, x16, arm)); c.trim(16)
        mx.eval(ys); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / chain
c = fresh(700, 9)
res = {a: [] for a in ARMS}
for _ in range(3):
    for a in ARMS:
        res[a].append(bench(c, a))
say("ms per 16-row attention call at T0 700 (shared GPU, rough; graph built per call): "
    + ", ".join(f"{a} {1e3 * min(v):.3f}" for a, v in res.items()))
open(os.path.expanduser("~/Leap/data/r72_att_anat.txt"), "w").write("\n".join(lines) + "\n")
