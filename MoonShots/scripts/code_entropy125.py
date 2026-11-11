"""How compressible (losslessly) are the 125B's routed-expert 4-bit codes? For layers 0,12,24,36,47 x gate/up/down, a sample of
experts: order-0 entropy H0 of the nibbles, a static Huffman code's mean length (one table per tensor), and the same with one
table per (tensor, nibble position in its 64-value group) and per expert, plus the bytes saved per expert (scales/biases
kept raw). Lossless => a decoder reproduces the exact codes. CPU only.   usage: code_entropy125.py [experts_per_tensor]"""
import heapq, json, os, struct, sys
import numpy as np
NE = int(sys.argv[1]) if len(sys.argv) > 1 else 48
MD = os.path.expanduser("~/BigMoE/models/bare")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
rng = np.random.default_rng(0)


def raw(name, sel):
    f = f"{MD}/{idx[name]}"
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; m = json.loads(fh.read(n))[name]
    a, b = m["data_offsets"]
    mm = np.memmap(f, np.uint32, "r", offset=8 + n + a, shape=((b - a) // 4,)).reshape(m["shape"])
    return np.array(mm[sel])                                     # [ne, rows, words]


def nibbles(w):
    w = w.astype(np.uint32)
    return np.stack([(w >> (4 * j)) & 15 for j in range(8)], axis=-1).reshape(w.shape[0], -1).astype(np.uint8)   # [ne, rows*K]


def huff_len(counts):
    """mean code length of a Huffman code for these symbol counts."""
    c = [int(x) for x in counts if x > 0]
    if len(c) == 1: return 1.0
    h = [(x, i, (i,)) for i, x in enumerate(c)]; heapq.heapify(h); depth = [0] * len(c); k = len(c)
    while len(h) > 1:
        a, _, sa = heapq.heappop(h); b, _, sb = heapq.heappop(h)
        for s in sa + sb: depth[s] += 1
        heapq.heappush(h, (a + b, k, sa + sb)); k += 1
    return sum(d * x for d, x in zip(depth, c)) / sum(c)


def H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


tot_bits, tot_raw = 0.0, 0.0
for L in (0, 12, 24, 36, 47):
    sel = np.sort(rng.choice(512, NE, replace=False))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        w = raw(f"language_model.model.layers.{L}.mlp.switch_mlp.{proj}.weight", sel)
        q = nibbles(w)                                            # [NE, rows*K]
        cnt = np.bincount(q.ravel(), minlength=16).astype(np.float64)
        h0, hl = H(cnt), huff_len(cnt)
        pe = np.mean([huff_len(np.bincount(q[i], minlength=16)) for i in range(NE)])        # one table per expert
        pos = q.reshape(NE, -1, 64)
        pp = np.mean([huff_len(np.bincount(pos[:, :, j].ravel(), minlength=16)) for j in range(64)])   # per group position
        print(f"L{L:2d} {proj:9s}: H0 {h0:.3f} bits | Huffman per tensor {hl:.3f}, per expert {pe:.3f}, per group position {pp:.3f} "
              f"| top code share {cnt.max()/cnt.sum():.3f}", flush=True)
        tot_bits += min(hl, pe) * q.size; tot_raw += 4 * q.size
print(f"routed-expert codes: Huffman {tot_bits / tot_raw * 4:.3f} bits per 4-bit code -> codes {100 * (1 - tot_bits / tot_raw):.1f} % "
      f"smaller; per expert (codes are {2.6/2.77:.0%} of 2.77 MB, scales/biases raw): {100 * (1 - tot_bits / tot_raw) * 0.94:.1f} % fewer bytes")
