"""R-82 HC-GATE (Leap, Sun 2026-11-01; shared GPU, real weights, tiny): ref2 CFG hc_gate (leapk.hc_read fuse_gate: the S4
partition sum + gate1 in one kernel) vs off, on the real HC weights of layers 3 / 20 / 44 (attention / MLP hyper-connection
reads, with and without the pending write), REF2 on, 16-row calls: mixed / written / inject bitwise; kernels per read.
Usage: [BigMoE venv] r82_hc_gate.py -> ~/Leap/data/r82_hc_gate.txt"""
import os, sys, json, re, tempfile, collections
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
args = Q.TextArgs.from_dict(cfg["text_config"])
def load(prefix, mod):
    keys = [k for k in wmap if k.startswith(prefix)]
    w = {}
    for f in sorted({wmap[k] for k in keys}):
        sh = mx.load(f"{D}/{f}")
        w.update({k[len(prefix):]: sh[k] for k in keys if wmap[k] == f})
    mod.load_weights(list(w.items()), strict=False)
    mod.eval()
ref2.apply()
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
b16 = lambda a: np.array(a.view(mx.uint16))
mx.random.seed(0)
fails, kc = 0, {}
for L in (3, 20, 44):
    for name in ("attn_hyper_connection", "mlp_hyper_connection"):
        hc = Q.GatedResidual(args)
        load(f"language_model.model.layers.{L}.{name}.", hc)
        mx.eval(hc.parameters())
        for t in range(4):
            sc = (0.5, 1.0, 2.0, 4.0)[t]
            h = (mx.random.normal((1, 16, 4 * 2560)) * sc).astype(mx.bfloat16)
            pend = (((mx.random.normal((1, 16, 2560)) * sc).astype(mx.bfloat16),
                     (mx.random.normal((1, 16, 4)) * 0.5).astype(mx.bfloat16)) if t % 2 else None)
            mx.eval(h, pend)
            outs = {}
            for on in (False, True):
                ref2.CFG["hc_gate"] = on
                o = hc(h, pend) if pend is not None else hc(h)
                if L == 3 and t < 2:
                    kc[(name, t % 2, on)] = census(*o)
                mx.eval(o)
                outs[on] = [b16(a) for a in o]
            eq = all((a == b).all() for a, b in zip(outs[False], outs[True]))
            fails += not eq
            if not eq:
                say(f"L{L} {name} t{t}: DIFFERS")
ref2.CFG["hc_gate"] = False
say("kernels per read (layer 3): " + ", ".join(f"{n.split('_')[0]} pending {bool(p)}: {kc[(n, p, False)]} -> {kc[(n, p, True)]}"
                                               for (n, p, o) in kc if not o))
say(f"R-82 bits (3 layers x 2 reads x 4 inputs, with / without pending): {'PASS' if fails == 0 else f'FAIL ({fails})'}; "
    f"hc_read calls {ref2.counts()['hc_read']}")
open(os.path.expanduser("~/Leap/data/r82_hc_gate.txt"), "w").write("\n".join(lines) + "\n")
