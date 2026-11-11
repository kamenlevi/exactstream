"""R-71b REF2c BUNDLE INV, GDN layer (Leap, Sun 2026-11-01; shared GPU, real weights): one real GDN layer of the bare
125B (MTPLX's GatedDeltaNet, layer 0's weights from shard 16) with REF2 on (GDNX + gated norm exact) and CFG gdn_inproj:
- wiring: the layer's first call builds ref2._InProjPad and MTPLX's forward runs it (counts gdn_inproj > 0);
- INV: after a 37-token prefill, a 16-row call == 16 one-row calls in sequence (y and the conv / recurrent state,
  bitwise), as the AR and the check see it;
- vs the switch off (today's REF2-T): the y bits that move.
Usage: [BigMoE venv] r71b_gdn_layer_inv.py -> ~/Leap/data/r71b_gdn_layer_inv.txt"""
import os, sys, json, copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
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
args = Q.TextArgs.from_dict(cfg["text_config"])
L = 0
pre = f"language_model.model.layers.{L}.linear_attn."
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}")
    w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
gdn = Q.GatedDeltaNet(args)
qcfg = cfg["quantization"]
nn.quantize(gdn, group_size=qcfg["group_size"], bits=qcfg["bits"],
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in w)
gdn.load_weights(list(w.items()), strict=False)
gdn.eval()                                         # inference mode, as mlx_lm loads it (GDNX needs it)
mx.eval(gdn.parameters())
say(f"layer {L}: {len(w)} tensors loaded; modules {sorted(k for k in gdn.keys())}")
ref2.apply()
mx.random.seed(1)
T0 = 37
xp = (mx.random.normal((1, T0, 2560)) * 1.0).astype(mx.bfloat16)
x16 = (mx.random.normal((1, 16, 2560)) * 1.0).astype(mx.bfloat16)

def run(on, rows_at_once):
    ref2.CFG["gdn_inproj"] = on
    c = ArraysCache(size=2)
    yp = gdn(xp, cache=c)                          # prefill (rows > 16: MLX's own calls, same in both arms)
    mx.eval(yp, c.state)
    if rows_at_once:
        y = gdn(x16, cache=c)
    else:
        y = mx.concatenate([gdn(x16[:, j:j + 1], cache=c) for j in range(16)], axis=1)
    mx.eval(y, c.state)
    return y, [np.array(s.view(mx.uint32) if s.dtype == mx.float32 else s.view(mx.uint16)) for s in c.state]
b16 = lambda a: np.array(a.view(mx.uint16))
ref2.reset_counts()
ya, sa = run(True, True)
built = isinstance(gdn.__dict__.get("in_proj_fused"), ref2._InProjPad)
n_on = ref2.counts()["gdn_inproj"]
yb, sb = run(True, False)
inv_y = bool((b16(ya) == b16(yb)).all())
inv_s = all((p == q).all() for p, q in zip(sa, sb))
yo, so = run(False, True)
yo1, so1 = run(False, False)
say(f"wiring: built {built}, gdn_inproj calls {n_on} (after the first 16-row call), counts {ref2.counts()}")
say(f"INV with the switch on: 16-row y == 16 one-row calls {'PASS' if inv_y else 'FAIL'}; conv + recurrent state "
    f"{'PASS' if inv_s else 'FAIL'}")
say(f"switch off (REF2-T) INV control: y {bool((b16(yo) == b16(yo1)).all())}, state "
    f"{all((p == q).all() for p, q in zip(so, so1))}; on vs off: y outputs differing {float((b16(ya) != b16(yo)).mean()):.4f}")
ok = built and n_on > 0 and inv_y and inv_s
say(f"R-71b GDN-layer INV: {'PASS' if ok else 'FAIL'}")
open(os.path.expanduser("~/Leap/data/r71b_gdn_layer_inv.txt"), "w").write("\n".join(lines) + "\n")
