"""ATT32-INV (CHECK55 step 1; Leap, Sun 15 Nov 2026 11:0x): can the exact 32-row check run its attention core as ONE
call instead of v32's two 16-row chunks, with every row bitwise?
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
usage: [BigMoE venv] att32_inv.py   (MEM < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
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
    if name == "mtplx":
        return Q._verify_sdpa
    if name == "one":
        return lambda q, k, v, *, scale, mask: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    if name == "h2":
        return lambda q, k, v, *, scale, mask: Q._sdpa_head_chunked(q, k, v, scale=scale, mask=mask, heads_per_chunk=2)
    raise ValueError(name)


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
say(f"ATT32-INV {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}; CFG sdpa_bounds {CFG['sdpa_bounds']}, "
    f"kb_from {CFG['kb_from']}, kb_step {CFG['kb_step']}")
PLANS = ("mtplx", "h2", "one")
CTX = (60, 100, 200, 383, 470, 497, 520, 700, 1000, 1500, 2050, 3000, 4500)
SEEDS = (1, 2, 3)
tot = {p: [0, 0] for p in PLANS}
worst = {p: 0.0 for p in PLANS}
for L in CTX:
    N2 = L + 32
    line = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 * L + s)
        q = mx.array((rng.standard_normal((1, HQ, 32, D)) * 1.6).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array((rng.standard_normal((1, HK, N2, D)) * 1.6).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array(rng.standard_normal((1, HK, N2, D)).astype(np.float32)).astype(mx.bfloat16)
        r = ref32(q, k, v)
        m32 = causal(32, N2, 0)
        for p in PLANS:
            c = cand32(plan_fn(p), q, k, v, m32)
            eq = mx.all(c == r, axis=(1, 3))[0]                  # per row: all heads, all dims
            d = mx.max(mx.abs(c.astype(mx.float32) - r.astype(mx.float32)))
            mx.eval(eq, d)
            ne = int(np.array(eq).sum())
            tot[p][0] += ne; tot[p][1] += 32
            worst[p] = max(worst[p], float(d.item()))
            line.append(f"{p} {ne}/32")
    say(f"  L {L:5d} (N2 {N2:5d}, call class {ref2._klass(N2)}, row classes {sorted(set(ref2._klass(L + 16 + j) for j in range(32)))}): "
        + "; ".join(line))
for p in PLANS:
    say(f"  {p:6s}: rows bitwise {tot[p][0]} / {tot[p][1]}, max |diff| {worst[p]:.3g}")
ok = [p for p in PLANS if tot[p][0] == tot[p][1]]
say(f"ATT32-INV verdict: {'ONE CALL IS EXACT with plan(s) ' + ', '.join(ok) if ok else 'NO plan is bitwise: attention stays 2 x 16 until step 3'}"
    f" ({time.time() - t0:.0f} s)")
