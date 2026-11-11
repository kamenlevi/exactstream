"""R-71 REF2c BUNDLE INV, in-proj part (Leap, Sun 2026-11-01; shared GPU, tensor level, real weights): ref2's CFG
gdn_inproj (R-68's padded fused GDN projection) on real 4-bit packs of GDN layers from the bare 125B (shard 16).
1. INV: with REF2 on and the switch on, each row of a 16-row call == the same row run alone (1 row, padded to 16 by
   REF2) and == the same row among 15 different companions, bitwise, for all four parts (qkv / z / b / a).
2. Which bits change vs today's REF2-T path (the four separate modules through REF2's routing).
3. Switch off on a built layer == the separate modules, bitwise (the reversible path).
(att_single's INV is R-62b: PASS.) Usage: [BigMoE venv] r71_bundle_inv.py -> ~/Leap/data/r71_bundle_inv.txt"""
import os, sys, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
q = cfg["quantization"]
shards = {}
def get(k):
    f = wmap[k]
    if f not in shards:
        shards[f] = mx.load(f"{D}/{f}")
    return shards[f][k]
ref2.apply()
NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
gdn_layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in wmap if ".linear_attn.in_proj_qkv.weight" in k})
pick = [gdn_layers[0], gdn_layers[len(gdn_layers) // 3], gdn_layers[2 * len(gdn_layers) // 3], gdn_layers[-1]]
say(f"GDN layers {len(gdn_layers)}; testing layers {pick}")
bits16 = lambda a: np.array(a.view(mx.uint16))
fails = 0
mx.random.seed(0)
for L in pick:
    gdn = nn.Module()
    for n in NAMES:
        pre = f"language_model.model.layers.{L}.linear_attn.{n}"
        w, s, b = get(pre + ".weight"), get(pre + ".scales"), get(pre + ".biases")
        lin = nn.QuantizedLinear(2560, w.shape[0], bias=False, group_size=q["group_size"], bits=q["bits"])
        lin.weight, lin.scales, lin.biases = w, s, b
        setattr(gdn, n, lin)
    mx.eval(gdn.parameters())
    sep = [getattr(gdn, n) for n in NAMES]
    f = ref2._InProjPad(gdn)
    for trial in range(4):
        x = (mx.random.normal((1, 16, 2560)) * (0.5 + trial)).astype(mx.bfloat16)
        xc = mx.concatenate([x[:, :1], (mx.random.normal((1, 15, 2560)) * 2).astype(mx.bfloat16)], axis=1)
        ref2.CFG["gdn_inproj"] = True
        y16 = f(x)
        y1 = [f(x[:, j:j + 1]) for j in range(16)]
        yc = f(xc)
        ref2.CFG["gdn_inproj"] = False
        ys = [m(x) for m in sep]                   # today's REF2-T path
        yo = f(x)                                  # switch off on the built layer
        mx.eval(y16, y1, yc, ys, yo)
        inv = all((bits16(y16[p][:, j]) == bits16(y1[j][p][:, 0])).all() for p in range(4) for j in range(16))
        comp = all((bits16(y16[p][:, 0]) == bits16(yc[p][:, 0])).all() for p in range(4))
        same = [bool((bits16(y16[p]) == bits16(ys[p])).all()) for p in range(4)]
        frac = [float((bits16(y16[p]) != bits16(ys[p])).mean()) for p in range(4)]
        off = all((bits16(yo[p]) == bits16(ys[p])).all() for p in range(4))
        fails += (not inv) + (not comp) + (not off)
        say(f"L{L} t{trial}: INV 16-row == 1-row {'PASS' if inv else 'FAIL'}; companions {'PASS' if comp else 'FAIL'}; "
            f"vs REF2-T separate calls equal per part (qkv, z, b, a) {same} (differing outputs {[f'{v:.3f}' for v in frac]}); "
            f"switch off == separate {'PASS' if off else 'FAIL'}")
say(f"counts {ref2.counts()}")
say(f"R-71 in-proj INV: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
open(os.path.expanduser("~/Leap/data/r71_bundle_inv.txt"), "w").write("\n".join(lines) + "\n")
