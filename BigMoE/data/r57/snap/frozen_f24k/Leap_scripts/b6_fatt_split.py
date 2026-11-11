"""B6 FATT16-S screen (Leap, Tue 2026-11-03 00:2x; Fable B's mission B6, ~/FableGod/fableb/LONG_CONTEXT.md section 5;
TABLE_350 'B6'; FAST-K): fatt16s.py (R-66's FATT16 with a KS-way split of the key walk over absolute 32-key blocks, a
fixed-order merge, all-masked blocks skipped, 16 or 8 real rows).
1. SDPA level, R-66's synthetic setup (24 / 2 heads x 256, bf16), masks = the indexer's shape (each row: 512 random
   4-key blocks of the past + its causal tail; plain causal at 2K):
   a. accuracy vs fp64 at N 2,000 (fatt16s, fatt16, MLX one call);
   b. INV (bitwise) at N 2,000 / 10,000 / 20,000, R 16 and R 8: probe row with (a) other rows' queries changed, (b) the
      same query at another row, (c) N + 37 / + 300 keys masked for every row, (d) an AR-style call (the query alone,
      N + 7), (e) R 8 only: the 8 rows as the first half of an R 16 call (head stacking does not matter);
   c. ms per call (8 chained, median of 20): MTPLX's 12 head chunks, MLX one call, fatt16, fatt16s KS 4 (+ KS 2 / 8);
      R 8 vs R 16 at 10,000.
2. Layer level, R-72 / R-81's harness (layer 3's real weights, MTPLX's Attention + indexer under REF2, prefill T0 in
   1,024-row chunks): 16-row check calls, 8 chained with trim, median of 20, arms ref2t (REF2-T's chunk set), single
   (att_single), fatt16, fs4 (pre-registered) and fs2 / fs8 (post-hoc), at T0 2,000 / 10,000 / 20,000; 8-row calls at
   10,000 (fs4 at R 8); each arm's y vs ref2t's (mean |diff| / RMS).
Lines (Fable B's): PASS = INV PASS and fs4 >= 3x vs REF2-T at 10K (R-81: 2.03 ms -> <= 0.68 ms per call); BETWEEN
1.5-3x; KILL < 1.5x or INV FAIL. The factor is taken against ref2t measured here (same method); R-81's is shown too.
Usage: [BigMoE venv] b6_fatt_split.py -> ~/Leap/data/b6_fatt_split.txt (GPU, ~2 GB; share lock)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from mtplx.models import qwen4_exp as Q
from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
from fatt16 import fatt16
from fatt16s import fatt16s

OUT = os.path.expanduser("~/Leap/data/b6_fatt_split.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"B6 FATT16-S screen {time.strftime('%a %H:%M')} ({os.environ.get('B6_COND', 'shared GPU, rough')})")
SC = 256 ** -0.5
bits = lambda a: np.array(a.view(mx.uint16)).astype(np.int64)
rng = np.random.default_rng(3)
def selmask(N, R=16):
    """the indexer's shape: per row 512 random 4-key blocks of the past (<= 2,048 keys: all of them) + the causal tail"""
    m = np.zeros((R, N), bool)
    for r in range(R):
        last = N - R + r                                             # row r sees keys <= last
        if last + 1 <= 2048:
            m[r, :last + 1] = True
            continue
        nb = (last + 1) // 4
        sel = rng.choice(nb, 512, replace=False)
        for b in sel:
            m[r, 4 * b:4 * b + 4] = True
        m[r, max(0, last - 63):last + 1] = True                      # the recent tail
        m[r, last + 1:] = False
    return mx.array(m)
def qkv(N, R=16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, 24, R, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v
# ---------------------------------------------------------------- 1a accuracy
N = 2000
q, k, v = qkv(N)
m = selmask(N)
outs = {"fatt16s": fatt16s(q, k, v, SC, m), "fatt16": fatt16(q, k, v, SC, m),
        "MLX one call": mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m[None, None])}
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
inv_ok = True
for N in (2000, 10000, 20000):
    for R in (16, 8):
        q, k, v = qkv(N, R, seed=N + R)
        m = selmask(N, R)
        P, P2 = (5, 11) if R == 16 else (2, 6)
        base = bits(fatt16s(q, k, v, SC, m)[0, :, P])
        q2 = (mx.random.normal(q.shape) * 0.8).astype(mx.bfloat16)
        qa = mx.concatenate([q2[:, :, :P], q[:, :, P:P + 1], q2[:, :, P + 1:]], axis=2)
        a = (bits(fatt16s(qa, k, v, SC, m)[0, :, P]) == base).all()
        qb = mx.concatenate([q2[:, :, :P2], q[:, :, P:P + 1], q2[:, :, P2 + 1:]], axis=2)
        mb = mx.concatenate([m[:P2], m[P:P + 1], m[P2 + 1:]], axis=0)
        b = (bits(fatt16s(qb, k, v, SC, mb)[0, :, P2]) == base).all()
        c = True
        for extra in (37, 300):
            k3 = mx.concatenate([k, (mx.random.normal((1, 2, extra, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
            v3 = mx.concatenate([v, mx.random.normal((1, 2, extra, 256)).astype(mx.bfloat16)], axis=2)
            m3 = mx.concatenate([m, mx.zeros((R, extra), mx.bool_)], axis=1)
            c &= bool((bits(fatt16s(q, k3, v3, SC, m3)[0, :, P]) == base).all())
        qd = mx.concatenate([mx.zeros((1, 24, P, 256), mx.bfloat16), q[:, :, P:P + 1],
                             mx.zeros((1, 24, R - 1 - P, 256), mx.bfloat16)], axis=2)
        k4 = mx.concatenate([k, (mx.random.normal((1, 2, 7, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
        v4 = mx.concatenate([v, mx.random.normal((1, 2, 7, 256)).astype(mx.bfloat16)], axis=2)
        m4 = mx.concatenate([m, mx.zeros((R, 7), mx.bool_)], axis=1)
        d = bool((bits(fatt16s(qd, k4, v4, SC, m4)[0, :, P]) == base).all())
        res = [a, b, c, d]
        if R == 8:
            q16 = mx.concatenate([q, mx.zeros((1, 24, 8, 256), mx.bfloat16)], axis=2)
            m16 = mx.concatenate([m, mx.zeros((8, N), mx.bool_)], axis=0)
            e_ = (bits(fatt16s(q16, k, v, SC, m16)[0, :, :8]) == bits(fatt16s(q, k, v, SC, m)[0])).all()
            res.append(e_)
        ok = all(bool(x) for x in res)
        inv_ok &= ok
        say(f"1b. INV N {N} R {R}: " + ", ".join(f"({t}) {'PASS' if x else 'FAIL'}" for t, x in zip("abcde", res))
            + f" -> {'PASS' if ok else 'FAIL'}")
# ---------------------------------------------------------------- 1c SDPA speed
def bench(f, chain=8, n=20):
    for _ in range(3):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
for N in (2000, 10000, 20000):
    q, k, v = qkv(N)
    m = selmask(N); m4d = m[None, None]
    mx.eval(m, m4d)
    t = {"12 chunks": bench(lambda: Q._sdpa_head_chunked(q, k, v, scale=SC, mask=m4d, heads_per_chunk=2)),
         "one call": bench(lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m4d)),
         "fatt16": bench(lambda: fatt16(q, k, v, SC, m))}
    for ks in (4, 2, 8):
        t[f"fatt16s KS {ks}"] = bench(lambda: fatt16s(q, k, v, SC, m, ks=ks))
    say(f"1c. N {N} (selection mask): " + ", ".join(f"{a} {b:.3f}" for a, b in t.items()) + " ms per call; fatt16s KS 4 "
        f"{t['12 chunks'] / t['fatt16s KS 4']:.2f}x vs the 12 chunks, {t['fatt16'] / t['fatt16s KS 4']:.2f}x vs fatt16")
    if N == 10000:
        q8, k8, v8 = qkv(N, 8)
        m8 = selmask(N, 8)
        t8 = bench(lambda: fatt16s(q8, k8, v8, SC, m8))
        say(f"1c. N 10000 R 8: fatt16s KS 4 {t8:.3f} ms per call = {t['fatt16s KS 4'] / t8:.2f}x faster than R 16")
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
_VS = Q._verify_sdpa                                                     # REF2's verify SDPA (R-40 classes, buckets)
MODE = ["ref2t", 4]
HITS = {"fatt": 0, "ref2": 0}                                            # which SDPA each verify call took
def _vsdpa(q, k, v, *, scale, mask):
    S = q.shape[2]
    if MODE[0] != "ref2t" and mask is not None and mask.dtype == mx.bool_:
        if MODE[0] == "fatt16" and S == 16:
            HITS["fatt"] += 1
            return fatt16(q, k, v, scale, mask)
        if MODE[0] == "fatt16s" and S in (16, 8):
            HITS["fatt"] += 1
            return fatt16s(q, k, v, scale, mask, ks=MODE[1])
    HITS["ref2"] += 1
    return _VS(q, k, v, scale=scale, mask=mask)
Q._verify_sdpa = _vsdpa
ARMS = {"ref2t": (False, "ref2t", 4), "single": (True, "ref2t", 4), "fatt16": (False, "fatt16", 4),
        "fs4": (False, "fatt16s", 4), "fs2": (False, "fatt16s", 2), "fs8": (False, "fatt16s", 8)}
R81 = {2000: (1.05, 0.83), 10000: (2.03, 1.71), 20000: (3.10, 2.91)}
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
verdict_x = None
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
        + f"; fs4 {ms['ref2t'] / ms['fs4']:.2f}x vs ref2t here ({R81[T0][0] / ms['fs4']:.2f}x vs R-81's {R81[T0][0]}),"
        f" {ms['single'] / ms['fs4']:.2f}x vs single; y vs ref2t mean |diff| / RMS "
        + ", ".join(f"{a} {d_:.1e}" for a, d_ in dif.items()) + f"; SDPA taken {hit}")
    if T0 == 10000:
        verdict_x = ms["ref2t"] / ms["fs4"]
        x8 = mx.random.normal((1, 8, 2560)).astype(mx.bfloat16); mx.eval(x8)
        set_arm("fs4"); t8 = lbench(c, x8)
        set_arm("ref2t"); t8r = lbench(c, x8)
        say(f"2. T0 10000, 8-row calls: fs4 (R 8) {t8:.3f} ms vs its 16-row call {ms['fs4']:.3f} ({ms['fs4'] / t8:.2f}x);"
            f" ref2t at 8 rows {t8r:.3f}")
    del c
    mx.clear_cache()
set_arm("ref2t")
v_ = "KILL (INV)" if not inv_ok else "PASS" if verdict_x >= 3 else "KILL" if verdict_x < 1.5 else "BETWEEN"
say(f"B6 verdict (INV {'PASS' if inv_ok else 'FAIL'}; fs4 vs ref2t at 10K {verdict_x:.2f}x; PASS >= 3x, KILL < 1.5x): {v_}")
