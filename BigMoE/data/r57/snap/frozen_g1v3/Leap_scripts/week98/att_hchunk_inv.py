"""ATT-HCHUNK (CHECK55 16-row lever (c); Leap, Sun 15 Nov 2026 11:4x): the exact verify SDPA runs MTPLX's head plan,
12 calls of 2 query heads (one kv head each) per layer: 36+ dispatches per attention layer. ATT32-INV (90dae21) found
2-head chunks at 32 rows (M 64) bitwise with the 16-row calls (M 32). Do larger chunks keep every row's bits?
At 16 rows: ref2._verify_sdpa (MTPLX's 2-head plan, the certified form) vs the same class logic on 4 / 6 / 12-head
chunks (12 = all query heads of a kv head: 2 calls per layer). At 32 rows: v32's two 16-row chunks vs one 32-row call
per class on 2 / 4 / 6 / 12-head chunks.
REF = v32's path: rows 0-15 through ref2._verify_sdpa at N1 = L + 16 keys, then rows 16-31 at N2 = L + 32 (REF2-T's
exact verify SDPA: MTPLX's head-chunked call, each row in its AR step's key class, bucket pads past 512 keys).
CAND = one 32-row call at N2 with the same class logic per row (row j's AR class from its AR key count L + 16 + j; a
row whose class differs from the call's is recomputed in its class and selected in, as ref2 does at 16 rows), in three
head plans:
  mtplx  MTPLX's own plan at 32 rows (24 chunks of 1 query head x 32 rows);
  h2     12 chunks of 2 query heads x 32 rows (the 16-row plan's heads);
  one    one call, all 24 query heads (MLX takes GQA natively).
Synthetic data at the model's shapes (24 / 2 heads, head dim 256, bf16, bool causal mask), peaky scores, several seeds,
contexts across the classes (below 128, the 128-511 band, the 512-key buckets).
Line: a plan with every row bitwise at every context and seed -> the attention core can be one call per class at 32
rows (exact); otherwise it stays 2 x 16 until step 3. Bits only (timing comes later, under a quiet ticket).
usage: [BigMoE venv] att_hchunk_inv.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1v3/Leap_scripts"))
import mtplx.models.qwen4_exp as Q
import ref2

ref2._ORIG["vsdpa"] = Q._verify_sdpa                     # as ref2.apply() sets it, without patching the model classes
ref2._ON[0] = True
CFG = ref2.CFG
assert CFG["sdpa_classes"] and CFG["key_buckets"] and not CFG["fatt_verify"] and not CFG["att_single"], CFG
HQ, HK, D = 24, 2, 256
SCALE = D ** -0.5


def say(s):
    print(s, flush=True)


def causal(rows, N, first):
    """bool [1, 1, rows, N]: row i (global row first + i, at position L + first + i) sees keys <= L + first + i"""
    L = N - (first + rows)
    pos = L + first + np.arange(rows)[:, None]
    return mx.array((np.arange(N)[None, :] <= pos)[None, None])


def plan_fn(name):
    h = int(name[1:])
    return lambda q, k, v, *, scale, mask: Q._sdpa_head_chunked(q, k, v, scale=scale, mask=mask, heads_per_chunk=h)


def sdpa_at(fn, q, k, v, mask, n):
    """ref2._sdpa_at with the call function fn: k / v zero-padded to n keys, the mask extended with False; n < N
    (a row class below the call's, 32 rows only) truncates to n keys, every such row's visible keys being < n"""
    N = k.shape[2]
    if n == N:
        return fn(q, k, v, scale=SCALE, mask=mask)
    if n < N:
        return fn(q, k[:, :, :n], v[:, :, :n], scale=SCALE, mask=mask[..., :n])
    pad = n - N
    kp = mx.concatenate([k, mx.zeros(k.shape[:2] + (pad, D), k.dtype)], axis=2)
    vp = mx.concatenate([v, mx.zeros(v.shape[:2] + (pad, D), v.dtype)], axis=2)
    mp = mx.concatenate([mask, mx.zeros(mask.shape[:-1] + (pad,), mx.bool_)], axis=-1)
    return fn(q, kp, vp, scale=SCALE, mask=mp)


def cand32(fn, q, k, v, mask):
    """ref2._verify_sdpa's class logic for an S-row call (S = 32): row j's AR step had N - S + 16 + j keys"""
    S, N = q.shape[2], k.shape[2]
    N1 = N - S + 16                                      # the first 16-row chunk's key count (v32's chunk 1)
    runN = lambda c, n: c if c >= CFG["kb_from"] else n if c == 0 else max(n, CFG["sdpa_bounds"][c - 1])
    c0 = ref2._klass(N)
    cls = [ref2._klass(N - S + 16 + j) for j in range(S)]
    out = sdpa_at(fn, q, k, v, mask, runN(c0, N))
    for c in sorted(set(cls) - {c0}):
        sel = mx.array([cls[j] == c for j in range(S)])[None, None, :, None]
        n = runN(c, N) if c > c0 else runN(c, N1)        # a lower class only holds rows 0-15: chunk 1's length
        out = mx.where(sel, sdpa_at(fn, q, k, v, mask, n), out)
    return out


def ref32(q, k, v):
    """v32's two chunks through ref2's exact verify SDPA: rows 0-15 at N1 = N2 - 16 keys, rows 16-31 at N2"""
    N2 = k.shape[2]
    N1 = N2 - 16
    y1 = ref2._verify_sdpa(q[:, :, :16], k[:, :, :N1], v[:, :, :N1], scale=SCALE, mask=causal(16, N1, 0))
    y2 = ref2._verify_sdpa(q[:, :, 16:], k, v, scale=SCALE, mask=causal(16, N2, 16))
    return mx.concatenate([y1, y2], axis=2)


t0 = time.time()
say(f"ATT-HCHUNK {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; CFG sdpa_bounds {CFG['sdpa_bounds']}, "
    f"kb_from {CFG['kb_from']}, kb_step {CFG['kb_step']}")
PLANS = ("h2", "h4", "h6", "h12")
CTX = (60, 100, 200, 383, 470, 497, 520, 700, 1000, 1500, 2050, 3000, 4500)
SEEDS = (1, 2, 3)
tot = {(p, S): [0, 0] for p in PLANS for S in (16, 32)}
for L in CTX:
    N2 = L + 32
    N16 = L + 16
    line = []
    for s_ in SEEDS:
        rng = np.random.default_rng(1000 * L + s_)
        q = mx.array((rng.standard_normal((1, HQ, 32, D)) * 1.6).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array((rng.standard_normal((1, HK, N2, D)) * 1.6).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array(rng.standard_normal((1, HK, N2, D)).astype(np.float32)).astype(mx.bfloat16)
        r32 = ref32(q, k, v)
        q16, k16, v16 = q[:, :, :16], k[:, :, :N16], v[:, :, :N16]
        m16 = causal(16, N16, 0)
        r16 = ref2._verify_sdpa(q16, k16, v16, scale=SCALE, mask=m16)
        for p in PLANS:
            for S, (c, r) in ((16, (cand32(plan_fn(p), q16, k16, v16, m16), r16)),
                              (32, (cand32(plan_fn(p), q, k, v, causal(32, N2, 0)), r32))):
                eq = mx.all(c == r, axis=(1, 3))[0]
                mx.eval(eq)
                ne = int(np.array(eq).sum())
                tot[(p, S)][0] += ne; tot[(p, S)][1] += S
                if s_ == SEEDS[0]:
                    line.append(f"{p}@{S} {ne}/{S}")
    say(f"  L {L:5d}: " + "; ".join(line) + " (seed 1)")
for p in PLANS:
    say(f"  {p:4s}: 16 rows bitwise {tot[(p, 16)][0]} / {tot[(p, 16)][1]}; 32 rows {tot[(p, 32)][0]} / {tot[(p, 32)][1]}")
ok = [p for p in PLANS if all(tot[(p, S)][0] == tot[(p, S)][1] for S in (16, 32))]
say(f"ATT-HCHUNK verdict: bitwise at 16 and 32 rows with: {', '.join(ok) if ok else 'none'} ({time.time() - t0:.0f} s)")
