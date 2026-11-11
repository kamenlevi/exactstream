"""R-83 GDN-INPROJ-X (Leap, Sun 2026-11-01; shared GPU, real weights, day-sized): ref2 CFG gdn_inproj_x (qkv | z as one
qmm16c call, b | a as one MLX 16-row call) must keep REF2-T's bits exactly (unlike REF2c's gdn_inproj, which moves b / a to
qmm16c).
1. Tensor level, real packs of GDN layers 0 / 16 / 32 / 46: every part of the fused calls == the separate REF2 calls,
   bitwise, 16-row and 1-row (padded) inputs at 4 scales.
2. Layer level, MTPLX's GatedDeltaNet with layer 0's real weights under REF2 + GDNX: after a 37-token prefill, a 16-row
   call and 16 one-row calls, y and the conv / recurrent state, switch on == switch off bitwise; then the switch turned
   off on a built layer (row slices) == stock.
3. Kernels per 16-row GDN call, off vs on.
Usage: [BigMoE venv] r83_gdn_x.py -> ~/Leap/data/r83_gdn_x.txt"""
import os, sys, json, re, tempfile, collections
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
from mlx_lm.models.cache import ArraysCache

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
qc = cfg["quantization"]
args = Q.TextArgs.from_dict(cfg["text_config"])
b16 = lambda a: np.array(a.view(mx.uint16))
ref2.apply()
NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
shards = {}
def get(k):
    f = wmap[k]
    if f not in shards:
        shards[f] = mx.load(f"{D}/{f}")
    return shards[f][k]

# 1. tensor level
fails1 = 0
mx.random.seed(0)
for L in (0, 16, 32, 46):
    gdn = nn.Module()
    for n in NAMES:
        pre = f"language_model.model.layers.{L}.linear_attn.{n}"
        lin = nn.QuantizedLinear(2560, get(pre + ".weight").shape[0], bias=False, group_size=64, bits=4)
        lin.weight, lin.scales, lin.biases = get(pre + ".weight"), get(pre + ".scales"), get(pre + ".biases")
        setattr(gdn, n, lin)
    mx.eval(gdn.parameters())
    sep = [getattr(gdn, n) for n in NAMES]
    f = ref2._InProjExact(gdn)
    ref2.CFG["gdn_inproj_x"] = True
    for sc in (0.5, 1.0, 2.0, 4.0):
        for S in (16, 1):
            x = (mx.random.normal((1, S, 2560)) * sc).astype(mx.bfloat16)
            yf, ys = f(x), [m(x) for m in sep]
            mx.eval(yf, ys)
            fails1 += sum(int((b16(a) != b16(b)).sum()) for a, b in zip(yf, ys))
    ref2.CFG["gdn_inproj_x"] = False
say(f"tensor level (4 layers x 4 scales x 16 / 1 rows, all four parts): {'PASS' if fails1 == 0 else f'FAIL ({fails1} values)'}")

# 2. layer level
L = 0
pre = f"language_model.model.layers.{L}.linear_attn."
keys = [k for k in wmap if k.startswith(pre)]
w = {k[len(pre):]: get(k) for k in keys}
gdn = Q.GatedDeltaNet(args)
nn.quantize(gdn, group_size=qc["group_size"], bits=qc["bits"], class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in w)
gdn.load_weights(list(w.items()), strict=False)
gdn.eval()
mx.eval(gdn.parameters())
mx.random.seed(1)
xp = mx.random.normal((1, 37, 2560)).astype(mx.bfloat16)
x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
mx.eval(xp, x16)
def run(on, rows_at_once):
    ref2.CFG["gdn_inproj_x"] = on
    c = ArraysCache(size=2)
    mx.eval(gdn(xp, cache=c), c.state)
    y = gdn(x16, cache=c) if rows_at_once else mx.concatenate([gdn(x16[:, j:j + 1], cache=c) for j in range(16)], axis=1)
    mx.eval(y, c.state)
    return [b16(y)] + [np.array(s.view(mx.uint32) if s.dtype == mx.float32 else s.view(mx.uint16)) for s in c.state]
A16, A1 = run(False, True), run(False, False)              # stock (the layer is not built yet)
B16, B1 = run(True, True), run(True, False)                # builds _InProjExact on the first call
built = isinstance(gdn.__dict__.get("in_proj_fused"), ref2._InProjExact)
C16 = run(False, True)                                      # built layer, switch off: row slices
eq = lambda a, b: all((p == q).all() for p, q in zip(a, b))
ok2 = built and eq(A16, B16) and eq(A1, B1) and eq(A16, C16)
say(f"layer level (layer 0, REF2 + GDNX): built {built}; 16-row call on == off {eq(A16, B16)}; 16 one-row calls on == off "
    f"{eq(A1, B1)}; switch off after the build == stock {eq(A16, C16)} -> {'PASS' if ok2 else 'FAIL'}")

# 3. kernels per 16-row call
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
    return sum(v for k_, v in cnt.items() if k_ not in VIEWS)
kn = {}
for on in (False, True):
    ref2.CFG["gdn_inproj_x"] = on
    c = ArraysCache(size=2)
    mx.eval(gdn(xp, cache=c), c.state)
    y = gdn(x16, cache=c)
    kn[on] = census(y, *c.state)
    mx.eval(y)
ref2.CFG["gdn_inproj_x"] = False
say(f"kernels per 16-row GDN call (y + state): off {kn[False]} -> on {kn[True]}; counts {ref2.counts()['gdn_inproj_x']} fused calls")
say(f"R-83: {'PASS' if fails1 == 0 and ok2 else 'FAIL'}")
open(os.path.expanduser("~/Leap/data/r83_gdn_x.txt"), "w").write("\n".join(lines) + "\n")
