"""M1a error budget: each engine component vs MTPLX on IDENTICAL inputs (real 125B weights, 4 real tokens, fresh caches),
for a GDN layer (0) and a QSA layer (3): (a) HC read; (b) GDN core given MTPLX's mixed input; (c) attention write + MLP HC
read given MTPLX's block output; (d) MoE given MTPLX's MLP input (same code: must be ~0); (e) MoE sensitivity: relative
output change and routing-set changes when the input gets bf16-level noise (x * (1 + 2^-9 N(0,1))).
usage: test_m1a_parts.py   (GPU, ~10 GB, ~1 min; lock)"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from rle.model import Engine, qmm
from rle import ref, kernels as KN
from bigmoe.loader import load_streamed
from bigmoe.gated import ext

MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, 48, mtp=False)
model = rt.model
ref.install(ext)
rec = ref.capture_layers()
eng = Engine(model, ext)
tok = rt.tokenizer
ids = mx.array([tok.encode(open(os.path.expanduser("~/MoonShots/FINDINGS.md")).read()[:2000])[:4]])
model(ids, cache=model.make_cache())                          # MTPLX's own per-layer inputs
mx.eval(rec)
rel = lambda a, b: (mx.linalg.norm((a.astype(mx.float32) - b.astype(mx.float32)).reshape(-1)) /
                    mx.maximum(mx.linalg.norm(b.astype(mx.float32).reshape(-1)), 1e-30)).item()
M = ids.shape[1]
for i in (0, 3, 30, 31):
    layer, d = model.language_model.model.layers[i], eng.P[i]
    H = rec[i][0]
    cache = model.make_cache()
    if "ple" in layer:
        H = H + layer.ple(H, ids, cache[i])
    mixed, hyper, inject = layer.attn_hyper_connection(H)                       # (a) HC read
    y1, mx_e = KN.hc_read(H[0], d["attn_hc"]["Wd"], d["attn_hc"]["Wu"], d["attn_hc"]["nw"], M)
    ea, eai = rel(mx_e, mixed[0]), rel(y1[:, 320:324], inject[0])
    if layer.is_linear:                                                         # (b) attention core on MTPLX's input
        out = layer.linear_attn(mixed, None, cache[i])
        from rle import gdn as G
        gin, _ = KN.gdn_prework(qmm(mixed[0], d["w_in"]), d["cw"], mx.zeros((3, 10240), mx.bfloat16), M)
        yr, _ = KN.gdn_recurrence(gin, mx.zeros((48 * 16384,), mx.float32), d["A_log"], d["dt_bias"], d["nwg"], M)
        out_e = qmm(yr, d["w_out"])
    else:
        out = layer.self_attn(mixed, cache[i])
        out_e = out[0]                                                          # (QSA core is MTPLX's own)
    eb = rel(out_e, out[0])
    mixed_m, hyper_m, inject_m = layer.mlp_hyper_connection(hyper, pending=(out, inject))   # (c) write + MLP read
    y1r = mx.concatenate([mx.zeros((M, 320), mx.float32), inject[0].astype(mx.float32)], axis=1)
    H1_e = KN.hc_write(H[0], y1r, out[0], M)
    ym_e, mm_e = KN.hc_read(H1_e, d["mlp_hc"]["Wd"], d["mlp_hc"]["Wu"], d["mlp_hc"]["nw"], M)
    ec, ech = rel(mm_e, mixed_m[0]), rel(H1_e, hyper_m[0])
    y_ref = layer.mlp(mixed_m)[0]                                               # (d) MoE, same input
    y_eng = eng.moe(i, mixed_m[0])
    ed = rel(y_eng, y_ref)
    rng = np.random.default_rng(0)                                              # (e) sensitivity to bf16-level noise
    x = mixed_m[0].astype(mx.float32)
    xn = (x * (1 + 2.0 ** -9 * mx.array(rng.standard_normal(x.shape).astype(np.float32)))).astype(mx.bfloat16)
    y_n = eng.moe(i, xn)
    lg = lambda v: np.argsort(-np.array(qmm(v, d["router"], d["router_bits"]).astype(mx.float32)), axis=-1)[:, :10]
    flips = sum(len(set(a) ^ set(b)) // 2 for a, b in zip(lg(mixed_m[0]), lg(xn)))
    print(f"L{i} {'gdn' if layer.is_linear else 'qsa'}: (a) HC read mixed {ea:.1e} inject {eai:.1e} | (b) core {eb:.1e} | "
          f"(c) write+MLP read mixed {ec:.1e} resid {ech:.1e} | (d) MoE same input {ed:.1e} | (e) MoE under 2^-9 noise "
          f"{rel(y_n, y_eng):.1e}, {flips} routing swaps in {M} rows", flush=True)
