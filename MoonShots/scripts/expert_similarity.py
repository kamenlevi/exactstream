"""Are the 125B's experts upcycled / correlated? If experts share structure, a missing expert could be streamed as a lossless
delta against a resident base. For layers 0/24/47, gate/up/down, a sample of experts: Pearson correlation of dequantized
weights between expert pairs and with the sample mean, and the order-0 entropy of 4-bit code differences (code_e - code_r)
mod 16 vs the codes themselves (~3.70 bits). CPU only.   usage: expert_similarity.py [experts]"""
import json, os, struct, sys
import numpy as np
NE = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
rng = np.random.default_rng(1)


def raw(name, sel, dt):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    mm = np.memmap(f, dt, "r", offset=8 + n + a, shape=((b - a) // np.dtype(dt).itemsize,)).reshape(m["shape"])
    return np.array(mm[sel])


def H(x):
    c = np.bincount(x.ravel(), minlength=16).astype(np.float64); p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


for L in (0, 24, 47):
    sel = np.sort(rng.choice(512, NE, replace=False))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        pre = f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}."
        w = raw(pre + "weight", sel, np.uint32)                                   # [NE, rows, K/8]
        s = raw(pre + "scales", sel, np.uint16); b = raw(pre + "biases", sel, np.uint16)
        q = np.stack([(w >> (4 * j)) & 15 for j in range(8)], -1).reshape(NE, w.shape[1], -1).astype(np.int16)   # [NE, rows, K]
        f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
        sc, bi = f32(s), f32(b)                                                    # [NE, rows, K/64]
        deq = q.reshape(NE, q.shape[1], -1, 64) * sc[..., None] + bi[..., None]   # [NE, rows, groups, 64]
        x = deq.reshape(NE, -1)
        x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        c = (x @ x.T) / x.shape[1]
        off = c[~np.eye(NE, dtype=bool)]
        mean = x.mean(0); cm = np.array([np.corrcoef(x[i], mean)[0, 1] for i in range(NE)])
        dq = ((q[1:] - q[:1]) % 16).astype(np.uint8)
        print(f"L{L:2d} {proj:9s}: pair corr mean {off.mean():+.3f} max {np.abs(off).max():.3f} | corr with mean {cm.mean():.3f} "
              f"| H(code) {H(q.astype(np.uint8)):.3f}  H(code diff vs expert 0) {H(dq):.3f} bits", flush=True)
