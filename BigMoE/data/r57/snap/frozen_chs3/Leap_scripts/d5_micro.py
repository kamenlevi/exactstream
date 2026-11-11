"""D5 micro (Leap, Wed 2026-11-04 16:4x): the per-boundary saving alone: Q._hyper_residual_write + the next attention-side HC
read (layer 20's real weights, REF2 on, 16 rows) with ref2 hc_pend_mlp off vs on; 64 independent boundaries per eval,
median of 15 interleaved rounds; R-73's hc_write off and on. -> ~/Leap/data/d5_check.txt (appends)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import ref2
from mtplx.models import qwen4_exp as Q
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
args = Q.TextArgs.from_dict(json.load(open(f"{D}/config.json"))["text_config"])
pre = "language_model.model.layers.20.attn_hyper_connection."
hc = Q.GatedResidual(args)
keys = [k for k in wmap if k.startswith(pre)]
w = {}
for f in sorted({wmap[k] for k in keys}):
    sh = mx.load(f"{D}/{f}"); w.update({k[len(pre):]: sh[k] for k in keys if wmap[k] == f})
hc.load_weights(list(w.items()), strict=False); mx.eval(hc.parameters())
ref2.apply()
mx.random.seed(1)
H = [(mx.random.normal((1, 16, 4 * 2560)) * 0.8).astype(mx.bfloat16) for _ in range(64)]
B = [(mx.random.normal((1, 16, 2560)) * 0.8).astype(mx.bfloat16) for _ in range(64)]
I = [(mx.random.normal((1, 16, 4)) * 0.5).astype(mx.bfloat16) for _ in range(64)]
mx.eval(H, B, I)
def run():
    return [hc(Q._hyper_residual_write(h, b, i))[0] for h, b, i in zip(H, B, I)]
res = {}
for hw in (False, True):
    ref2.CFG["hc_write"] = hw
    T = {False: [], True: []}
    for r in range(15):
        for on in ((False, True) if r % 2 == 0 else (True, False)):
            ref2.CFG["hc_pend_mlp"] = on
            mx.eval(run())
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(run()); mx.synchronize()
            T[on].append((time.perf_counter() - t0) / 64)
    res[hw] = (np.median(T[False]), np.median(T[True]))
ref2.CFG["hc_pend_mlp"] = False; ref2.CFG["hc_write"] = False
line = "D5 micro (write + next HC read, layer 20, 16 rows, us per boundary): " + "; ".join(
    f"hc_write {'on' if hw else 'off'}: off {1e6 * a:.1f}, D5 {1e6 * b:.1f} ({1e6 * (b - a):+.1f})" for hw, (a, b) in res.items())
print(line); open(os.path.expanduser("~/Leap/data/d5_check.txt"), "a").write(line + "\n")
ref2.remove()
