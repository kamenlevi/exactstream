"""B6-G screen (Leap, Tue 2026-11-03 15:1x; pre-registered in ~/Leap/TABLE_350.md 'B6-G'; FAST-K): fatt16g (GQA-packed
FATT16: each K / V block loaded once per threadgroup for all query heads of its kv head) vs fatt16s (B6's fs4).
1. SDPA level, B6's synthetic setup (24 / 2 heads x 256, bf16, the indexer's mask shape):
   a. accuracy vs fp64 at N 2,000;
   b. INV (a)-(e) as B6 at N 2K / 10K / 20K, R 16 and R 8, primary config (12, 8); (a) and (c) at 10K for the others;
   c. ms per call (8 chained, median of 20): fs4, fatt16g (HG, KS) = (12, 8) (12, 4) (12, 16) (6, 8) (4, 6), the 12
      chunks and MLX's one call; R 8 at 10K.
2. Layer level, R-72's harness (layer 3): ref2t, single, fs4, fg (12, 8) [+ the best SDPA config, post-hoc] at T0 2K / 10K
   / 20K, 16-row calls chained 8 with trim, median of 20.
Lines: PASS = INV PASS and fg (12, 8) >= 2.0x vs fs4 at 10K (SDPA level); KILL < 1.3x or INV FAIL.
Usage: [BigMoE venv] b6g_screen.py -> ~/Leap/data/b6g_screen.txt (GPU, ~2 GB; share lock)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
from fatt16s import fatt16s
from fatt16g import fatt16g

OUT = os.path.expanduser("~/Leap/data/b6g_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"B6-G FATT16-G screen {time.strftime('%a %H:%M')} ({os.environ.get('B6G_COND', 'shared GPU, rough')})")
SC = 256 ** -0.5
bits = lambda a: np.array(a.view(mx.uint16)).astype(np.int64)
rng = np.random.default_rng(3)
def selmask(N, R=16):
    m = np.zeros((R, N), bool)
    for r in range(R):
        last = N - R + r
        if last + 1 <= 2048:
            m[r, :last + 1] = True
            continue
        for b in rng.choice((last + 1) // 4, 512, replace=False):
            m[r, 4 * b:4 * b + 4] = True
        m[r, max(0, last - 63):last + 1] = True
        m[r, last + 1:] = False
    return mx.array(m)
def qkv(N, R=16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, 24, R, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v
CFGS = [(12, 8), (12, 4), (12, 16), (6, 8), (4, 6)]
def fg(cfg, R=16):
    hg, ks = cfg
    return lambda q, k, v, s_, m: fatt16g(q, k, v, s_, m, hg=hg if R == 16 else hg // 2, ks=ks)
# ---------------------------------------------------------------- 1a accuracy
N = 2000
q, k, v = qkv(N)
m = selmask(N)
outs = {"fatt16g (12, 8)": fg((12, 8))(q, k, v, SC, m), "fatt16s KS 4": fatt16s(q, k, v, SC, m)}
mx.eval(list(outs.values()))
qn, kn, vn = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v))
mn = np.array(m)
ref = np.zeros((24, 16, 256))
for h in range(24):
    s_ = np.where(mn, qn[0, h] @ kn[0, h // 12].T * SC, -np.inf)
    p_ = np.exp(s_ - s_.max(-1, keepdims=True)); p_ /= p_.sum(-1, keepdims=True)
    ref[h] = p_ @ vn[0, h // 12]
rms = float(np.sqrt((ref ** 2).mean()))
for nm, y in outs.items():
    e = np.abs(np.array(y.astype(mx.float32))[0].astype(np.float64) - ref)
    say(f"1a. {nm}: |err| vs fp64 max {e.max() / rms:.2e} mean {e.mean() / rms:.2e}")
# ---------------------------------------------------------------- 1b INV
def inv(f, N, R, full=True):
    q, k, v = qkv(N, R, seed=N + R)
    m = selmask(N, R)
    P, P2 = (5, 11) if R == 16 else (2, 6)
    base = bits(f(q, k, v, SC, m)[0, :, P])
    q2 = (mx.random.normal(q.shape) * 0.8).astype(mx.bfloat16)
    qa = mx.concatenate([q2[:, :, :P], q[:, :, P:P + 1], q2[:, :, P + 1:]], axis=2)
    res = {"a": (bits(f(qa, k, v, SC, m)[0, :, P]) == base).all()}
    c = True
    for extra in (37, 300):
        k3 = mx.concatenate([k, (mx.random.normal((1, 2, extra, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
        v3 = mx.concatenate([v, mx.random.normal((1, 2, extra, 256)).astype(mx.bfloat16)], axis=2)
        m3 = mx.concatenate([m, mx.zeros((R, extra), mx.bool_)], axis=1)
        c &= bool((bits(f(q, k3, v3, SC, m3)[0, :, P]) == base).all())
    res["c"] = c
    if full:
        qb = mx.concatenate([q2[:, :, :P2], q[:, :, P:P + 1], q2[:, :, P2 + 1:]], axis=2)
        mb = mx.concatenate([m[:P2], m[P:P + 1], m[P2 + 1:]], axis=0)
        res["b"] = (bits(f(qb, k, v, SC, mb)[0, :, P2]) == base).all()
        qd = mx.concatenate([mx.zeros((1, 24, P, 256), mx.bfloat16), q[:, :, P:P + 1],
                             mx.zeros((1, 24, R - 1 - P, 256), mx.bfloat16)], axis=2)
        k4 = mx.concatenate([k, (mx.random.normal((1, 2, 7, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
        v4 = mx.concatenate([v, mx.random.normal((1, 2, 7, 256)).astype(mx.bfloat16)], axis=2)
        m4 = mx.concatenate([m, mx.zeros((R, 7), mx.bool_)], axis=1)
        res["d"] = bool((bits(f(qd, k4, v4, SC, m4)[0, :, P]) == base).all())
        if R == 8:
            q16 = mx.concatenate([q, mx.zeros((1, 24, 8, 256), mx.bfloat16)], axis=2)
            m16 = mx.concatenate([m, mx.zeros((8, N), mx.bool_)], axis=0)
            f16 = lambda *a: fatt16g(*a, hg=12, ks=8)
            res["e"] = (bits(f16(q16, k, v, SC, m16)[0, :, :8]) == bits(f(q, k, v, SC, m)[0])).all()
    return {t: bool(x) for t, x in sorted(res.items())}
inv_ok = True
for N in (2000, 10000, 20000):
    for R in (16, 8):
        r = inv(fg((12, 8), R), N, R)
        inv_ok &= all(r.values())
        say(f"1b. INV fg (12, 8) N {N} R {R}: " + ", ".join(f"({t}) {'PASS' if x else 'FAIL'}" for t, x in r.items()))
for cfg in CFGS[1:]:
    r = inv(fg(cfg), 10000, 16, full=False)
    inv_ok &= all(r.values())
    say(f"1b. INV fg {cfg} N 10000 R 16: " + ", ".join(f"({t}) {'PASS' if x else 'FAIL'}" for t, x in r.items()))
# ---------------------------------------------------------------- 1c SDPA speed
def bench(f, chain=8, n=20):
    for _ in range(3):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
sdpa10 = {}
for N in (2000, 10000, 20000):
    q, k, v = qkv(N)
    m = selmask(N); m4d = m[None, None]
    mx.eval(m, m4d)
    t = {"12 chunks": bench(lambda: Q._sdpa_head_chunked(q, k, v, scale=SC, mask=m4d, heads_per_chunk=2)),
         "one call": bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m4d)),
         "fs4": bench(lambda: fatt16s(q, k, v, SC, m))}
    for cfg in CFGS:
        t[f"fg {cfg}"] = bench(lambda: fg(cfg)(q, k, v, SC, m))
    say(f"1c. N {N}: " + ", ".join(f"{a} {b:.3f}" for a, b in t.items()) + f" ms per call; fg (12, 8) "
        f"{t['fs4'] / t['fg (12, 8)']:.2f}x vs fs4, {t['12 chunks'] / t['fg (12, 8)']:.2f}x vs the 12 chunks")
    if N == 10000:
        sdpa10 = t
        q8, k8, v8 = qkv(N, 8)
        m8 = selmask(N, 8)
        t8 = bench(lambda: fg((12, 8), 8)(q8, k8, v8, SC, m8))
        say(f"1c. N 10000 R 8: fg (12, 8) {t8:.3f} ms per call = {t['fg (12, 8)'] / t8:.2f}x faster than its R 16")
x10 = sdpa10["fs4"] / sdpa10["fg (12, 8)"]
best = min(CFGS, key=lambda c: sdpa10[f"fg {c}"])
say(f"1c. best config at 10K: {best} ({sdpa10[f'fg {best}']:.3f} ms)")
# ---------------------------------------------------------------- 2. layer level
import ref2
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
qc = cfg["quantization"]
args = Q.TextArgs.from_dict(cfg["text_config"])
pre = "language_model.model.layers.3.self_attn."
keys = [k_ for k_ in wmap if k_.startswith(pre)]
w = {}
for f in sorted({wmap[k_] for k_ in keys}):
    sh = mx.load(f"{D}/{f}")
    w.update({k_[len(pre):]: sh[k_] for k_ in keys if wmap[k_] == f})
att = Q.Attention(args)
def qpred(p, m_):
    if not (isinstance(m_, nn.Linear) and f"{p}.scales" in w):
        return False
    c_ = qc.get(pre + p, {})
    return {"group_size": c_.get("group_size", qc["group_size"]), "bits": c_.get("bits", qc["bits"])}
nn.quantize(att, group_size=qc["group_size"], bits=qc["bits"], class_predicate=qpred)
att.load_weights(list(w.items()), strict=False)
att.eval()
mx.eval(att.parameters())
ref2.apply()
_VS = Q._verify_sdpa
MODE = ["ref2t", None]
HITS = {"fatt": 0, "ref2": 0}
def _vsdpa(q, k, v, *, scale, mask):
    S = q.shape[2]
    if MODE[0] != "ref2t" and mask is not None and mask.dtype == mx.bool_ and S in (16, 8):
        HITS["fatt"] += 1
        if MODE[0] == "fs4":
            return fatt16s(q, k, v, scale, mask)
        hg, ks = MODE[1]
        return fatt16g(q, k, v, scale, mask, hg=hg if S == 16 else hg // 2, ks=ks)
    HITS["ref2"] += 1
    return _VS(q, k, v, scale=scale, mask=mask)
Q._verify_sdpa = _vsdpa
ARMS = {"ref2t": (False, "ref2t", None), "single": (True, "ref2t", None), "fs4": (False, "fs4", None),
        "fg (12, 8)": (False, "fg", (12, 8))}
if best != (12, 8):
    ARMS[f"fg {best}"] = (False, "fg", best)
def set_arm(a):
    ref2.CFG["att_single"], MODE[0], MODE[1] = ARMS[a]
def call(c, x):
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        return att(x, c)
def lbench(c, x, chain=8, n=20):
    S = x.shape[1]
    def go():
        ys = []
        for _ in range(chain):
            ys.append(call(c, x)); c.trim(S)
        return ys
    for _ in range(2):
        mx.eval(go())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(go()); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
for T0 in (2000, 10000, 20000):
    mx.random.seed(T0)
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    set_arm("ref2t")
    for a0 in range(0, T0, 1024):
        n_ = min(1024, T0 - a0)
        y = att(mx.random.normal((1, n_, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16); mx.eval(x16)
    ys, ms, hit = {}, {}, {}
    for a in ARMS:
        set_arm(a)
        HITS["fatt"] = HITS["ref2"] = 0
        ys[a] = call(c, x16); mx.eval(ys[a]); c.trim(16)
        hit[a] = "fatt" if HITS["fatt"] and not HITS["ref2"] else "ref2" if HITS["ref2"] and not HITS["fatt"] else str(HITS)
        ms[a] = lbench(c, x16)
    r = np.array(ys["ref2t"].astype(mx.float32)); rr = float(np.sqrt((r ** 2).mean()))
    dif = {a: float(np.abs(np.array(ys[a].astype(mx.float32)) - r).mean() / rr) for a in ARMS if a != "ref2t"}
    say(f"2. T0 {T0}: ms per 16-row layer call " + ", ".join(f"{a} {ms[a]:.3f}" for a in ARMS)
        + f"; fg (12, 8) {ms['ref2t'] / ms['fg (12, 8)']:.2f}x vs ref2t, {ms['fs4'] / ms['fg (12, 8)']:.2f}x vs fs4; "
        + "y vs ref2t mean |diff| / RMS " + ", ".join(f"{a} {d_:.1e}" for a, d_ in dif.items()) + f"; SDPA taken {hit}")
    del c
    mx.clear_cache()
set_arm("ref2t")
v_ = "KILL (INV)" if not inv_ok else "PASS" if x10 >= 2.0 else "KILL" if x10 < 1.3 else "BETWEEN"
say(f"B6-G verdict (INV {'PASS' if inv_ok else 'FAIL'}; fg (12, 8) vs fs4 at 10K, SDPA level {x10:.2f}x; PASS >= 2.0x, "
    f"KILL < 1.3x): {v_}")
