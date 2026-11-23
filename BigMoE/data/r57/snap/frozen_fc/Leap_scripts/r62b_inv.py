"""R-62b INV (Leap, Sun 2026-11-01; synthetic, shared GPU): is the single-call 16-row SDPA (ref2 CFG att_single, R-62) usable
as a reference, i.e. are a row's bits independent of everything but its own query, its visible keys and the padded key
length? MLX's full (NAX) attention kernel at q_len 16, 24 q heads, 2 kv heads, head_dim 256, bf16, bool mask.
  a. row position: query row x at slot j (0..15) of the 16-row call, other rows random, same keys / mask row -> bitwise
  b. companions: the same slot, 5 different sets of other rows -> bitwise
  c. padding: the same row and visible keys, actual key count N varied inside one padded length L (zeros, masked) -> bitwise
  d. control: the chunked MTPLX path vs the single call past 512 keys must differ somewhere (R-62), else the test is blind
Key lengths: padded L in (384, 512, 1024, 2048) (REF2b runs >= 512-key calls at 512-multiples; below 512 at N itself).
Usage: [BigMoE venv] r62b_inv.py -> ~/Leap/data/r62b_inv.txt"""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
SC = 256 ** -0.5
mx.random.seed(3)
bits = lambda a: np.array(a.view(mx.uint16)).astype(np.int64)
one = lambda q, k, v, m: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m)
fails = 0
for L in (384, 512, 1024, 2048):
    k = (mx.random.normal((1, 2, L, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, L, 256)).astype(mx.bfloat16)
    x = (mx.random.normal((1, 24, 1, 256)) * 0.8).astype(mx.bfloat16)
    nvis = L - 40                                              # the probe row sees keys 0 .. nvis - 1
    mrow = mx.arange(L) < nvis
    def call(slot, others_seed, N=L):
        mx.random.seed(others_seed)
        q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
        q = mx.concatenate([q[:, :, :slot], x, q[:, :, slot + 1:]], axis=2)
        vis = mx.arange(L)[None, :] < (mx.arange(16)[:, None] + N - 16 + 1)     # causal rows over the first N keys
        vis = mx.where((mx.arange(16) == slot)[:, None], mrow[None, :], vis)
        kk = mx.where((mx.arange(L) < N)[None, None, :, None], k, 0)
        vv = mx.where((mx.arange(L) < N)[None, None, :, None], v, 0)
        y = one(q, kk, vv, vis[None, None])
        return bits(y[:, :, slot])
    base = call(0, 11)
    a = all((call(j, 11) == base).all() for j in range(16))
    b = all((call(5, s) == base).all() for s in (21, 22, 23, 24, 25))
    c = all((call(9, 31, N) == base).all() for N in (nvis + 1, nvis + 10, L - 1, L))
    fails += (not a) + (not b) + (not c)
    if L >= 512:
        q = (mx.random.normal((1, 24, 16, 256)) * 0.8).astype(mx.bfloat16)
        m = (mx.arange(L)[None, :] <= mx.arange(L - 16, L)[:, None])[None, None]
        d = int((bits(Q._sdpa_head_chunked(q, k, v, scale=SC, mask=m, heads_per_chunk=2)) != bits(one(q, k, v, m))).sum())
    else:
        d = "-"
    say(f"L {L}: a row position {'PASS' if a else 'FAIL'}; b companions {'PASS' if b else 'FAIL'}; c padding inside L "
        f"{'PASS' if c else 'FAIL'}; d control (chunked vs one call, differing outputs) {d}")
say(f"R-62b INV: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
open(os.path.expanduser("~/Leap/data/r62b_inv.txt"), "w").write("\n".join(lines) + "\n")
