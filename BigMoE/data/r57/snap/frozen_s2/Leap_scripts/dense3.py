"""DENSE-3 (Leap, Wed 2026-11-04 17:0x; TABLE_350 'DENSE-3'; the orchestrator's dense diet (b), FAST): the trunk's dense
4-bit g64 projections at their MSB-3 (the expert tiers' 3-bit form): q3 = q4 >> 1, s3 = 2 s4 (exact), b3 = bf16(b4 + s4 / 2)
(the dropped bit's mean), so w3 = s3 q3 + b3 = s4 (2 q3 + 1/2) + b4.
- sim(model, which): the QUALITY screen with no new kernel: in place in the 4-bit format, every code's low bit cleared
  (w & 0xEEEEEEEE) and biases set to bf16(b4 + s4 / 2): w' = s4 (2 q3) + b4 + s4 / 2 == w3 exactly (the same bf16 bias,
  the same scale arithmetic); today's kernels run it unchanged.
- to3(w, s, b): the real 3-bit arrays in MLX's 3-bit layout (8 codes per 24 bits, as records3's MSB parts and
  mx.quantize(bits=3)), for the 3-bit kernel / MLX's bits=3 prefill.
which: module-path substrings to convert (default: GDN in / out projections and attention q / k / v / o):
  "linear_attn.in_proj", "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"
(not the router, the shared expert, the indexer, the experts, embeddings or the lm_head)."""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

WHICH = ("linear_attn.in_proj", "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj")


def _targets(model, which):
    for path, m in model.named_modules():
        if (isinstance(m, nn.QuantizedLinear) and m.bits == 4 and m.group_size == 64 and "biases" in m
                and m["weight"].ndim == 2 and any(t in path for t in which) and ".mtp." not in f".{path}."):
            yield path, m


def sim(model, which=WHICH):
    """the 3-bit values in the 4-bit format, in place; returns (modules, MB of 4-bit weights touched)"""
    n, nb = 0, 0
    for path, m in _targets(model, which):
        w, s, b = m["weight"], m["scales"], m["biases"]
        m.weight = mx.bitwise_and(w, mx.array(0xEEEEEEEE, mx.uint32))
        m.biases = (b.astype(mx.float32) + 0.5 * s.astype(mx.float32)).astype(b.dtype)
        mx.eval(m.weight, m.biases)
        n += 1
        nb += w.nbytes + s.nbytes + b.nbytes
    return n, nb / 1e6


def to3(w, s, b):
    """4-bit g64 (MLX: [N, K/8] uint32 nibbles, code k at bits 4 (k % 8) of word k / 8) -> 3-bit g64 in MLX's 3-bit
    layout ([N, 3 K / 32] uint32: 8 codes per 24 bits, code k at bit 3 (k % 8) of its 3-byte chunk), s3 = 2 s, b3 =
    bf16(b + s / 2)"""
    W = np.array(w).view(np.uint32)
    N, KW = W.shape
    nib = ((W[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).reshape(N, KW * 8)     # [N, K] codes
    q3 = (nib >> 1).astype(np.uint32).reshape(N, -1, 8)                                           # [N, K/8, 8]
    chunk = (q3 << (3 * np.arange(8, dtype=np.uint32))).sum(-1, dtype=np.uint64).astype(np.uint32)   # 24-bit chunks
    by = np.stack([(chunk >> (8 * i)) & 0xFF for i in range(3)], -1).astype(np.uint8).reshape(N, -1)   # 3 bytes each
    w3 = mx.array(np.ascontiguousarray(by).view(np.uint32).reshape(N, -1))
    s3 = (s.astype(mx.float32) * 2.0).astype(s.dtype)
    b3 = (b.astype(mx.float32) + 0.5 * s.astype(mx.float32)).astype(b.dtype)
    return w3, s3, b3
