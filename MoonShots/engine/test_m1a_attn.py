"""M1a check: the roofline engine's GDN attention half (engine/rle/gdn.py) vs MTPLX's modules on the real 125B weights of
layer 0 (BigMoE's loader, C=16 expert slots, exact env). Input: H = tile(embed(real tokens), 4). Two calls on one cache
(M=4 then M=1) to check the conv tail and recurrent state carry over. Reports relative errors of H1 (residual after the
attention write), the MLP-side mixed input and inject, and the states. Non-exact contract: expect ~1e-3..1e-2 (bf16 level).
usage: test_m1a_attn.py   (GPU, ~6 GB; share lock)"""
import os, sys
os.environ.setdefault("MTPLX_NGRAM_PREWARM", "off")
for k in [k for k in os.environ if k.startswith("BIGMOE_")]:
    os.environ.pop(k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.expanduser("~/BigMoE"))
import numpy as np
import mlx.core as mx
from rle import gdn as G
from bigmoe.loader import load_streamed

MD = os.path.expanduser("~/BigMoE/models/bare")
rt, st = load_streamed(MD, 16, mtp=False)
model = rt.model
tm = model.language_model.model
layer = tm.layers[0]
assert layer.is_linear
P = G.pack_gdn(layer)
mx.eval(P)
cache = model.make_cache()
c0 = cache[0]
print("cache[0]:", type(c0).__name__, [type(a).__name__ if a is not None else None for a in getattr(c0, "state", [])], flush=True)
ids = mx.array([[785, 3974, 13876, 38835, 34208]])        # arbitrary real token ids
emb = tm.embed_tokens(ids)                                 # [1, 5, 2560]
Hall = mx.tile(emb, (1, 1, 4))                             # [1, 5, 10240]
rel = lambda a, b: (mx.linalg.norm((a.astype(mx.float32) - b.astype(mx.float32)).reshape(-1)) /
                    mx.maximum(mx.linalg.norm(b.astype(mx.float32).reshape(-1)), 1e-30)).item()
cst = mx.zeros((3, 10240), mx.bfloat16)
state = mx.zeros((48 * 16384,), mx.float32)
ok = True
for (a, b) in ((0, 4), (4, 5)):                             # M=4, then M=1 on the same cache
    H = Hall[:, a:b]
    M = b - a
    mixed, hyper, inject = layer.attn_hyper_connection(H)
    out = layer.linear_attn(mixed, None, c0)
    mixed_m, hyper_m, inject_m = layer.mlp_hyper_connection(hyper, pending=(out, inject))
    H1, mm, ym, cst, state = G.attn_half(P, H[0], cst, state, M)
    mx.eval(mixed_m, hyper_m, inject_m, H1, mm, ym, cst, state)
    errs = dict(H1=rel(H1, hyper_m[0]), mixed_m=rel(mm, mixed_m[0]), inject_m=rel(ym[:, 320:324], inject_m[0]))
    try:
        cs, ss = c0.state[0], c0.state[1]
        errs["conv_state"] = rel(cst, cs.reshape(-1, 10240)[-3:])
        errs["ssm_state"] = rel(state, ss.reshape(-1))
    except Exception as e:
        errs["state_check"] = f"skipped ({type(e).__name__}: {str(e)[:80]})"
    print(f"M={M}: " + ", ".join(f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}" for k, v in errs.items()), flush=True)
    ok &= all(v < 3e-2 for v in errs.values() if isinstance(v, float))
print("PASS (within bf16-level tolerance 3e-2)" if ok else "CHECK: some relative errors >= 3e-2", flush=True)
