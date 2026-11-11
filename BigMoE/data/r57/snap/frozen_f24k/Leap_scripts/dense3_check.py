"""DENSE-3 check (Leap, Wed 2026-11-04 17:0x): dense3.sim and dense3.to3 on real weights (layer 4's GDN in_proj_qkv /
out_proj, layer 7's attention q_proj, 4-bit g64):
 (1) to3's arrays are MLX's own 3-bit layout: mx.dequantize(bits=3) of them == the MSB-3 values s3 q3 + b3 computed
     from the 4-bit codes, bitwise; and mx.quantized_matmul(bits=3) runs on them;
 (2) sim (the 4-bit format) dequantizes to the same values as to3's 3-bit arrays, bitwise;
 (3) the 3-bit error vs the 4-bit weights (relative rms of the dequantized weights; output error on random x).
usage: [BigMoE venv] dense3_check.py -> ~/Leap/data/dense3_check.txt (GPU, small)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
import dense3

OUT = os.path.expanduser("~/Leap/data/dense3_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nDENSE-3 check {time.strftime('%a %H:%M:%S')}")
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
ok = True
mx.random.seed(2)
for pre in ("language_model.model.layers.4.linear_attn.in_proj_qkv", "language_model.model.layers.4.linear_attn.out_proj",
            "language_model.model.layers.7.self_attn.q_proj"):
    sh = mx.load(f"{D}/{wmap[pre + '.weight']}")
    w, s, b = sh[pre + ".weight"], sh[pre + ".scales"], sh[pre + ".biases"]
    if w.dtype != mx.uint32 or w.shape[1] * 8 != s.shape[1] * 64:
        say(f"{pre}: not 4-bit g64 ({w.dtype}, {w.shape}, {s.shape}); skipped"); continue
    N, K = w.shape[0], w.shape[1] * 8
    w3, s3, b3 = dense3.to3(w, s, b)
    d3 = mx.dequantize(w3, s3, b3, group_size=64, bits=3)
    q4 = mx.dequantize(w, mx.ones_like(s), mx.zeros_like(b), group_size=64, bits=4)    # the raw codes as floats
    q3 = mx.floor(q4.astype(mx.float32) / 2.0)
    ref = (mx.repeat(s3.astype(mx.float32), 64, axis=1) * q3 + mx.repeat(b3.astype(mx.float32), 64, axis=1)).astype(d3.dtype)
    lay = bool(mx.all(d3.view(mx.uint16) == ref.view(mx.uint16)).item()) if d3.dtype == mx.bfloat16 else bool(mx.all(d3 == ref).item())
    m4 = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=4)
    m4.update({"weight": w, "scales": s, "biases": b})
    class Holder(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.linear_attn = {"in_proj_qkv": m}
    hold = Holder(m4)
    nmod, mb = dense3.sim(hold)
    dsim = mx.dequantize(m4["weight"], m4["scales"], m4["biases"], group_size=64, bits=4)
    simeq = bool(mx.all(dsim.view(mx.uint16) == d3.view(mx.uint16)).item())
    d4 = mx.dequantize(w, s, b, group_size=64, bits=4).astype(mx.float32)
    rel = float(mx.sqrt(mx.mean((d3.astype(mx.float32) - d4) ** 2)).item() / mx.sqrt(mx.mean(d4 ** 2)).item())
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    y4 = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4).astype(mx.float32)
    y3 = mx.quantized_matmul(x, w3, s3, b3, transpose=True, group_size=64, bits=3).astype(mx.float32)
    yrel = float(mx.sqrt(mx.mean((y3 - y4) ** 2)).item() / mx.sqrt(mx.mean(y4 ** 2)).item())
    good = lay and simeq and nmod == 1
    ok &= good
    say(f"{pre.split('model.')[-1]} [{N}x{K}]: MLX bits=3 dequant of to3 == s3 q3 + b3 bitwise {lay}; sim (4-bit format) "
        f"== to3 bitwise {simeq} ({nmod} module); 3-bit vs 4-bit weights rel rms {rel:.3f}, outputs {yrel:.3f}; "
        f"bytes {w.nbytes + s.nbytes + b.nbytes} -> {w3.nbytes + s3.nbytes + b3.nbytes}")
say(f"DENSE-3 check: {'PASS' if ok else 'FAIL'}")
