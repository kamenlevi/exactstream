"""TrainLab (CPU only): what a trained 125B MTP head run changed, layer by layer, against the stock head's storage.
For each trained layer: rms(change) / rms(weight); for 4/8-bit layers the share of weights whose change exceeds half
of their group's step (scale): roughly the weights whose stored code would move if re-quantized with the stock grid;
the share of the change re-quantization keeps (<dq, d> / <d, d>); and how much of the change's energy ranks 8-128
capture (a 4-bit base + bf16 low-rank delta would deploy that).
usage: delta125.py [RUN]  (default mtp125n_d4)   -> stdout + data/delta125_<run>.txt"""
import json, os, sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)
TL = os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/TrainLab_root")
run = sys.argv[1] if len(sys.argv) > 1 else "mtp125n_d4"
stock = mx.load(os.path.expanduser("~/BigMoE/models/bare/mtp.safetensors"))
tuned = mx.load(os.path.join(TL, "ckpt", run + ".mtp125.safetensors"))
out = []


def say(s):
    print(s, flush=True); out.append(s)


say(f"{run}: {len(tuned)} trained layers")
tot_d, tot_keep, tot_keep8 = 0.0, 0.0, 0.0
for k in sorted(tuned):
    name = k[:-len(".weight")]
    sk = "mtp." + name
    t = tuned[k].astype(mx.float32)
    if sk + ".scales" in stock:
        wq, sc, bi = stock[sk + ".weight"], stock[sk + ".scales"], stock[sk + ".biases"]
        bits = wq.shape[1] * 32 // t.shape[1]
        g = t.shape[1] // sc.shape[1]
        w = mx.dequantize(wq, sc, bi, group_size=g, bits=bits).astype(mx.float32)
        step = mx.abs(mx.repeat(sc.astype(mx.float32), g, axis=1))          # MLX scales can be negative
        d = t - w
        frac_half = float((mx.abs(d) > 0.5 * step).astype(mx.float32).mean())
        q2 = mx.quantize(t.astype(mx.bfloat16), group_size=g, bits=bits)
        dq = mx.dequantize(*q2, group_size=g, bits=bits).astype(mx.float32) - w
        keep = float((dq * d).sum() / mx.maximum((d * d).sum(), 1e-30))
        moved = float((mx.abs(dq) > 0).astype(mx.float32).mean())
        q8 = mx.quantize(t.astype(mx.bfloat16), group_size=g, bits=8)
        dq8 = mx.dequantize(*q8, group_size=g, bits=8).astype(mx.float32) - w
        keep8 = float((dq8 * d).sum() / mx.maximum((d * d).sum(), 1e-30))
        fmt = f"{bits}-bit g{g}"
        extra = (f"|d| > step/2 {100 * frac_half:5.2f} %  requant keeps {keep:+.3f} of the change (codes moved "
                 f"{100 * moved:5.2f} %), at 8 bits {keep8:+.3f}")
        tot_keep += keep * float((d * d).sum()); tot_keep8 += keep8 * float((d * d).sum())
    else:
        w = stock[sk + ".weight"].astype(mx.float32)
        d = t - w
        fmt = str(stock[sk + ".weight"].dtype).split(".")[-1]
        extra = "stored as is (bf16 keeps it)"
        tot_keep += float((d * d).sum()); tot_keep8 += float((d * d).sum())
    e = float((d * d).sum()); tot_d += e
    rel = float(mx.sqrt((d * d).mean() / (w * w).mean()))
    dn = np.array(d)
    G = dn.T @ dn if dn.shape[0] >= dn.shape[1] else dn @ dn.T
    ev = np.sort(np.linalg.eigvalsh(G.astype(np.float64)))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    ranks = " ".join(f"r{r} {cum[r - 1]:.2f}" for r in (8, 32, 128) if r <= len(cum))
    say(f"{name:42s} {str(tuple(t.shape)):14s} {fmt:9s} rms d/w {rel:.4f}  {extra}  energy {ranks}")
say(f"all layers: re-quantization keeps {tot_keep / tot_d:.3f} of the change energy, at 8 bits {tot_keep8 / tot_d:.3f} "
    "(bf16 layers count fully)")
open(os.path.join(TL, f"data/delta125_{run}.txt"), "w").write("\n".join(out) + "\n")
