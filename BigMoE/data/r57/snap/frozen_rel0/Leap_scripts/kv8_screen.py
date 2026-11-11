"""KV8 screen (Leap, Tue 2026-11-03 21:4x; TABLE_350 'KV8', pre-registered 21:32; FAST-Q): 8-bit K / V as a
long-context speed lever. Arms at the SDPA level, R 16, B6's synthetic setup with the indexer's mask shape:
  mlx1  MLX's one bf16 call            mlxq8  MLX-LM's quantized SDPA (8 bits, group 64; the published MLX form)
  fs4   fatt16s bf16 (KS 4)             q8s    fatt16q MODE 0 (int8 storage, converted to bf16 for fs4's MMAs)
  q8m   fatt16q MODE 1 (int8 Q x int8 K -> int32 on matmul2d (16, 32, 32); PV as q8s)
KV8_PART=acc: errors vs fp64 attention on the bf16 inputs (mean and max |diff| / RMS) and vs fs4, at N 2K / 10K / 20K,
  plus an outlier-K variant at 10K and REAL-weight K / V at 10K (layer 3's projections, k_norm and RoPE over random
  hidden states, B6's layer harness, captured in the 16-row verify call).
KV8_PART=speed: ms per call (8 chained, median of 20) at N 2K / 10K / 20K; the verdict by the pre-registered lines.
Usage: [BigMoE venv] KV8_PART=acc|speed kv8_screen.py -> ~/Leap/data/kv8_screen.txt"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel0/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
from fatt16s import fatt16s
from fatt16q import fatt16q, quant_kv
from mlx_lm.models.base import quantized_scaled_dot_product_attention as _mlxq_published   # noqa: F401 (the source)


def mlxq_sdpa(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
    """MLX-LM 0.31.3's quantized_scaled_dot_product_attention, verbatim except `queries = queries * scale`: its
    `queries *= scale` rebinds the CALLER's array in MLX, which scaled the shared q of every later arm (Tue 21:35 run)"""
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    queries = queries * scale
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tuple(mx.expand_dims(x, axis=-3) for x in q_keys)
        q_values = tuple(mx.expand_dims(x, axis=-3) for x in q_values)
    scores = mx.quantized_matmul(queries, *q_keys, transpose=True, group_size=group_size, bits=bits)
    if mask is not None:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *q_values, transpose=False, group_size=group_size, bits=bits)
    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out

OUT = os.path.expanduser("~/Leap/data/kv8_screen.txt")
PART = os.environ.get("KV8_PART", "acc")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"KV8 screen {PART} {time.strftime('%a %H:%M')} ({os.environ.get('KV8_COND', 'shared GPU, rough')})")
SC = 256 ** -0.5
rng = np.random.default_rng(3)


def selmask(N, R=16):
    """B6's, verbatim: the indexer's shape: per row 512 random 4-key blocks of the past (<= 2,048 keys: all of them) +
    the causal tail"""
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


def qkv(N, R=16, seed=0, outlier=False):
    mx.random.seed(seed)
    q = (mx.random.normal((1, 24, R, 256)) * 0.8).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, N, 256)) * 0.8
    if outlier:   # K's outlier channels: 8 channels x 6 the scale + a per-channel offset (KIVI / SageAttention's case)
        sc_ = np.ones(256, np.float32); sc_[rng.choice(256, 8, replace=False)] = 6.0
        k = k * mx.array(sc_) + mx.array(rng.normal(0, 1.5, 256).astype(np.float32))
    k = k.astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def ref64(q, k, v, m):
    qn, kn, vn = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v))
    mn = np.array(m)
    Hq, R = qn.shape[1], qn.shape[2]
    rep = Hq // kn.shape[1]
    out = np.zeros((Hq, R, qn.shape[3]))
    for h in range(Hq):
        s_ = np.where(mn, qn[0, h] @ kn[0, h // rep].T * SC, -np.inf)
        p_ = np.exp(s_ - s_.max(-1, keepdims=True)); p_ /= p_.sum(-1, keepdims=True)
        out[h] = p_ @ vn[0, h // rep]
    return out


def make_arms(q, k, v, m):
    c = quant_kv(k, v)
    kq, vq = mx.quantize(k, group_size=64, bits=8), mx.quantize(v, group_size=64, bits=8)
    mx.eval(kq, vq)
    return {"mlx1": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=SC, mask=m[None, None]),
            "mlxq8": lambda: mlxq_sdpa(q, kq, vq, SC, m, group_size=64, bits=8),
            "fs4": lambda: fatt16s(q, k, v, SC, m, ks=4),
            "q8s": lambda: fatt16q(q, c, SC, m, 0),
            "q8m": lambda: fatt16q(q, c, SC, m, 1)}


def errors(arms, ref, tag):
    rms = float(np.sqrt((ref ** 2).mean()))
    ys = {a: np.array(f().astype(mx.float32))[0].astype(np.float64) for a, f in arms.items()}
    res = {}
    for a, y in ys.items():
        d = np.abs(y - ref) / rms
        e4 = np.abs(y - ys["fs4"]) / rms
        res[a] = float(d.mean())
        say(f"  {tag} {a:6s}: vs fp64 mean {d.mean():.2e} max {d.max():.2e}; vs fs4 mean {e4.mean():.2e} max {e4.max():.2e}")
    return res


def bench(f, chain=8, n=20):
    for _ in range(2):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain


if PART == "acc":
    for N in (2000, 10000, 20000):
        q, k, v = qkv(N, seed=N)
        m = selmask(N)
        errors(make_arms(q, k, v, m), ref64(q, k, v, m), f"N {N:5d} synthetic")
    q, k, v = qkv(10000, seed=7, outlier=True)
    m = selmask(10000)
    errors(make_arms(q, k, v, m), ref64(q, k, v, m), "N 10000 outlier-K")
    # REAL-weight K / V: B6's layer harness (layer 3's weights, a 10K prefill of random hidden states), captured
    from mtplx.models import qwen4_exp as Q
    from mtplx.attention_context import attention_phase, exact_verify, model_forward_kind
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
    CAP = {}
    def _cap(q, k, v, *, scale, mask):
        if q.shape[2] == 16 and mask is not None and mask.dtype == mx.bool_ and "q" not in CAP:
            CAP.update(q=q, k=k, v=v, mask=mask, scale=scale)
        return _VS(q, k, v, scale=scale, mask=mask)
    Q._verify_sdpa = _cap
    T0 = 10000
    mx.random.seed(T0)
    c = Q.QSACache(compress_ratio=args.indexer_compress_ratio)
    for a0 in range(0, T0, 1024):
        n_ = min(1024, T0 - a0)
        y = att(mx.random.normal((1, n_, 2560)).astype(mx.bfloat16), c)
        mx.eval(y, [z for z in (c.kv.keys, c.kv.values, c.raw_keys, c.pooled) if z is not None])
    x16 = mx.random.normal((1, 16, 2560)).astype(mx.bfloat16)
    with attention_phase("decode_verify"), model_forward_kind("target_verify"), exact_verify(True):
        mx.eval(att(x16, c))
    assert "q" in CAP, "no 16-row verify SDPA call was captured"
    q, k, v = CAP["q"], mx.contiguous(CAP["k"]), mx.contiguous(CAP["v"])
    m = CAP["mask"]
    m = (mx.broadcast_to(m, m.shape[:-2] + (16, k.shape[2])).reshape(-1, 16, k.shape[2])[0] if m.ndim > 2 else m).astype(mx.bool_)
    mx.eval(q, k, v, m)
    assert abs(CAP["scale"] - SC) < 1e-6, CAP["scale"]
    kf = np.array(k.astype(mx.float32))
    say(f"  real-weight K / V: q {tuple(q.shape)}, k {tuple(k.shape)}, visible keys per row {int(np.array(m).sum(-1).mean())}; "
        f"K channel |mean| max {np.abs(kf.mean(2)).max():.2f}, channel RMS max / median "
        f"{np.sqrt((kf ** 2).mean(2)).max() / np.median(np.sqrt((kf ** 2).mean(2))):.1f}")
    errors(make_arms(q, k, v, m), ref64(q, k, v, m), "N 10000 real-weight")
else:
    ms = {}
    for N in (2000, 10000, 20000):
        q, k, v = qkv(N, seed=N)
        m = selmask(N)
        arms = make_arms(q, k, v, m)
        ms[N] = {a: bench(f) for a, f in arms.items()}
        say(f"  N {N:5d}: ms per call " + ", ".join(f"{a} {t:.3f}" for a, t in ms[N].items())
            + f"; vs fs4: q8s {ms[N]['fs4'] / ms[N]['q8s']:.2f}x, q8m {ms[N]['fs4'] / ms[N]['q8m']:.2f}x"
            + f"; mlxq8 vs mlx1 {ms[N]['mlx1'] / ms[N]['mlxq8']:.2f}x"
            + f"; per check (12 layers) q8 best vs fs4 {12 * (ms[N]['fs4'] - min(ms[N]['q8s'], ms[N]['q8m'])):+.2f} ms")
    best = lambda N: max(ms[N]["fs4"] / ms[N]["q8s"], ms[N]["fs4"] / ms[N]["q8m"])
    f10, f20 = best(10000), best(20000)
    v = "PASS" if f10 >= 1.25 and f20 >= 1.2 else "KILL" if f10 < 1.10 else "BETWEEN"
    say(f"KV8 speed verdict (the error clause from the acc part): {v}: best 8-bit form {f10:.2f}x at 10K, {f20:.2f}x at 20K vs fs4")
