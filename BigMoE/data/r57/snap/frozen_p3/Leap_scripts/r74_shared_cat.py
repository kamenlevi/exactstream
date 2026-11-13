"""R-74 SHARED-CAT probe (Leap, Sun 2026-11-01; shared GPU, real weights, tiny): can the MoE input's small 4-bit projections
(shared expert gate_proj N 640, up_proj N 640, shared_expert_gate N 1; all on MLX's own 16-row call under REF2, N < 1024)
run as ONE call on the row-concatenated packs with every part's bits unchanged? REF2 would route a fused N >= 1024 to
qmm16c (other bits), so the fused call goes to MLX's quantized_matmul directly, padded to 16 rows like REF2 does.
Checks per layer (real packs of layers 3 / 20 / 44, 4 input scales, 16-row and 1-row (padded) calls): each part of the
fused output == the separate REF2 call, bitwise; zero rows appended to N % 8 / 32 / 64 variants.
Usage: [BigMoE venv] r74_shared_cat.py -> ~/Leap/data/r74_shared_cat.txt"""
import os, sys, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
shards = {}
def get(k):
    f = wmap[k]
    if f not in shards:
        shards[f] = mx.load(f"{D}/{f}")
    return shards[f][k]
ref2.apply()
ORIG = ref2._ORIG["qmm"]
b16 = lambda a: np.array(a.view(mx.uint16))
NAMES = ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert_gate")
mx.random.seed(0)
fails = {}
for L in (3, 20, 44):
    pre = f"language_model.model.layers.{L}.mlp."
    parts = [(get(pre + n + ".weight"), get(pre + n + ".scales"), get(pre + n + ".biases")) for n in NAMES]
    rows = [p[0].shape[0] for p in parts]
    o = np.cumsum([0] + rows)
    for padto in (1, 8, 32, 64):
        n = int(o[-1]); pad = (-n) % padto
        cat = [mx.concatenate([p[i] for p in parts] + ([mx.zeros((pad,) + parts[0][i].shape[1:], parts[0][i].dtype)]
                                                        if pad else [])) for i in range(3)]
        mx.eval(cat)
        bad = 0
        for sc in (0.5, 1.0, 2.0, 4.0):
            for S in (16, 1):
                x = (mx.random.normal((1, S, 2560)) * sc).astype(mx.bfloat16)
                sep = [mx.quantized_matmul(x, *p, transpose=True, group_size=64, bits=4) for p in parts]   # REF2's routing
                x16 = ref2._pad16(x.reshape(S, 2560))
                y = ORIG(x16, *cat, transpose=True, group_size=64, bits=4)[:S].reshape(1, S, -1)
                mx.eval(sep, y)
                for i in range(3):
                    bad += int((b16(y[..., int(o[i]):int(o[i + 1])]) != b16(sep[i])).sum())
        fails[(L, padto)] = bad
        say(f"layer {L}, N {n} + {pad} zero rows (pad to {padto}): parts differing from the separate calls: {bad} values")
say(f"R-74: {'PASS' if all(v == 0 for v in fails.values()) else 'FAIL'} ({sum(v == 0 for v in fails.values())} / {len(fails)} "
    f"configs bitwise); counts {ref2.counts()}")
open(os.path.expanduser("~/Leap/data/r74_shared_cat.txt"), "w").write("\n".join(lines) + "\n")
