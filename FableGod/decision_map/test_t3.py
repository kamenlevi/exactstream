"""Unit test of dm_t3 (part 1 on the CPU, part 2 on the GPU: MLX 0.32.2's CPU 3-bit qmm disagrees with dequantize by 5-10 %) (the M-18 arm): (1) records3 read with my offsets rebuilds the bare model's 4-bit codes and scales
of layer 0 / 47 experts bit-exactly (q = 2m + l, s = s3 / 2); (2) routed_top2 at bits=3 equals a plain reference (mx.dequantize
bits=3 + matmuls + the block's normalised top-2 weights) on random rows. Usage (BigMoE venv): python test_t3.py -> ALL OK."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
import dm_t3 as T
bad = 0

def bits_unpack(words, bits, n):
    """MLX bitstream (value i at bit bits*i, little-endian uint32 words) -> n codes per row."""
    w = words.astype(np.uint64); rows = w.shape[0]
    flat = np.zeros((rows, words.shape[1] * 32), np.uint8)
    for b in range(32): flat[:, b::32] = ((w >> b) & 1).astype(np.uint8)
    flat = flat[:, :n * bits].reshape(rows, n, bits)
    return (flat * (1 << np.arange(bits, dtype=np.uint64))).sum(-1).astype(np.uint8)

wm = json.load(open(os.path.join(T.MD, "model.safetensors.index.json")))["weight_map"]
for L, ids in ((0, [0, 311]), (47, [7, 500])):
    W = T.read(L, np.array(ids), lsb=True)
    for proj in ("gate_proj", "down_proj"):
        key = f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}"
        src = mx.load(os.path.join(T.MD, wm[key + ".weight"]))
        q4w, s4 = np.array(src[key + ".weight"][mx.array(ids)]), src[key + ".scales"][mx.array(ids)]
        o, k = T._SHP[proj]
        for u in range(len(ids)):
            m = bits_unpack(np.array(W[proj]["w3"][u]), 3, k); l = bits_unpack(np.array(W[proj]["l1"][u]), 1, k)
            q = bits_unpack(q4w[u], 4, k)
            ok = np.array_equal(2 * m + l, q) and mx.array_equal(W[proj]["s3"][u] * mx.array(0.5, mx.bfloat16), s4[u]).item()
            bad += not ok; print(f"layer {L} expert {ids[u]} {proj}: rebuilt 4-bit codes + scales {'ok' if ok else 'MISMATCH'}")
        del src

class Blk:                                                    # the block surface routed_top2 uses
    def __init__(s):
        s.gate = nn.Linear(2560, 512, bias=False); s.top_k = 10; s.norm_topk_prob = True
        s.switch_mlp = type("S", (), {"activation": staticmethod(lambda up, gt: nn.silu(gt) * up)})()
mx.set_default_device(mx.gpu)                                 # MLX 0.32.2's CPU bits=3 qmm is off by 5-10 % (GPU is the run's path)
blk = Blk(); mx.eval(blk.gate.parameters())
rng = np.random.default_rng(0); xg = mx.array(rng.standard_normal((5, 2560)).astype(np.float32)).astype(mx.bfloat16)
y = T.routed_top2(blk, 3, xg, bits=3)
g = mx.softmax(blk.gate(xg), axis=-1, precise=True); top = np.argsort(-np.array(g.astype(mx.float32)), axis=1)[:, :10]
ref = []
for r in range(5):
    gr = np.array(g[r].astype(mx.float32)); sc = gr[top[r]] / gr[top[r]].sum(); acc = 0
    Wr = T.read(3, top[r][:2])
    for j in range(2):
        f = lambda p: mx.dequantize(Wr[p]["w3"][j], Wr[p]["s3"][j], Wr[p]["b3"][j], group_size=64, bits=3)
        h = xg[r:r + 1]
        a = nn.silu(h @ f("gate_proj").T) * (h @ f("up_proj").T)
        acc = acc + float(sc[j]) * (a @ f("down_proj").T).astype(mx.float32)
    ref.append(acc)
ref = mx.concatenate(ref).astype(mx.float32); err = float(mx.max(mx.abs(y.astype(mx.float32) - ref)) / mx.max(mx.abs(ref)))
ok = err < 3e-2; bad += not ok
print(f"routed_top2 bits=3 vs dequantize reference: max rel err {err:.2e} {'ok' if ok else 'MISMATCH'}; stats {T.STATS}")
y10 = T.routed_topk(blk, 3, xg, bits=3, K=10)                 # t3_all (Mon 23:3x): all ten ranks
ref = []
for r in range(5):
    gr = np.array(g[r].astype(mx.float32)); sc = gr[top[r]] / gr[top[r]].sum(); acc = 0
    Wr = T.read(3, top[r][:10])
    for j in range(10):
        f = lambda p: mx.dequantize(Wr[p]["w3"][j], Wr[p]["s3"][j], Wr[p]["b3"][j], group_size=64, bits=3)
        h = xg[r:r + 1]
        a = nn.silu(h @ f("gate_proj").T) * (h @ f("up_proj").T)
        acc = acc + float(sc[j]) * (a @ f("down_proj").T).astype(mx.float32)
    ref.append(acc)
ref = mx.concatenate(ref).astype(mx.float32); err = float(mx.max(mx.abs(y10.astype(mx.float32) - ref)) / mx.max(mx.abs(ref)))
ok = err < 3e-2; bad += not ok
print(f"routed_topk K=10 bits=3 vs dequantize reference: max rel err {err:.2e} {'ok' if ok else 'MISMATCH'}")
kp = np.ones((5, 10), bool); kp[:, 4:] = False; kp[1, 1] = False   # a rank mask (P1-K1 at 3 bits): ranks 0-3 minus one
yk = T.routed_topk(blk, 3, xg, bits=3, K=10, keep=kp)
ref = []
for r in range(5):
    gr = np.array(g[r].astype(mx.float32)); sc = gr[top[r]] / gr[top[r]].sum(); acc = 0
    Wr = T.read(3, top[r][:10])
    for j in range(10):
        if not kp[r, j]: continue
        f = lambda p: mx.dequantize(Wr[p]["w3"][j], Wr[p]["s3"][j], Wr[p]["b3"][j], group_size=64, bits=3)
        h = xg[r:r + 1]
        a = nn.silu(h @ f("gate_proj").T) * (h @ f("up_proj").T)
        acc = acc + float(sc[j]) * (a @ f("down_proj").T).astype(mx.float32)
    ref.append(acc)
ref = mx.concatenate(ref).astype(mx.float32); err = float(mx.max(mx.abs(yk.astype(mx.float32) - ref)) / mx.max(mx.abs(ref)))
ok = err < 3e-2; bad += not ok
print(f"routed_topk K=10 with a rank mask, bits=3 vs dequantize reference: max rel err {err:.2e} {'ok' if ok else 'MISMATCH'}")
from bigmoe import tier                                     # the control's 4-bit rebuild (Metal), vs the bare weights
Wc = T.read(3, np.array([10, 20]), lsb=True)
for proj in ("gate_proj", "down_proj"):
    key = f"language_model.model.layers.3.mlp.switch_mlp.{proj}"
    src = mx.load(os.path.join(T.MD, wm[key + ".weight"]))
    w4 = tier.rebuild4(Wc[proj]["w3"], Wc[proj]["l1"]); ref4 = src[key + ".weight"][mx.array([10, 20])]
    ok = mx.array_equal(w4, ref4).item() and mx.array_equal(Wc[proj]["b"], src[key + ".biases"][mx.array([10, 20])]).item()
    bad += not ok; print(f"control rebuild4 {proj} (layer 3, experts 10 / 20) == bare 4-bit words + biases: {'ok' if ok else 'MISMATCH'}")
    del src
print("ALL OK" if not bad else f"{bad} MISMATCHES")
