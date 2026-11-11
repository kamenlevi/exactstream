"""L1-15 check (Leap, Sat 2026-10-24): why the GDN kernel program gave ~0 end to end (small35_gdnk.txt) when 30 mixers
timed 2.7x faster on captured inputs. Here everything runs as the loop runs it: SERIAL (each layer's input is the
previous layer's output), in the real verify pass.
  1. the whole 16-row verify pass (model forward, taps, the loop's GDN capture), stock vs DFL35_GDNK=1;
  2. only the 30 GDN mixers chained (x = x + mixer(x)), stock vs the kernel program, and the same 30 mixers on
     independent captured inputs (the earlier method), to see how much the earlier method hid by overlap.
The caches are restored after each timed pass (KV trimmed, GDN states put back).
Usage: [SpecDecode venv] test_pass16.py  -> ~/Leap/data/test_pass16.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import mlx.core as mx
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_third_party/sd35"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfd/Leap_scripts"))
import dflash35_model_mlx as dfm
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, ArraysCache
from mlx_lm.models.base import create_attention_mask

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
model, tok = dfm.load(MODEL)
mx.eval(model.parameters())
TM = model.language_model.model
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"].startswith("tf_code"):
        user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
        ids = list(ids if isinstance(ids[0], int) else ids["input_ids"])
        break
out = [int(r.token) for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=64, sampler=make_sampler(0.0))]
ctx, blk = ids + out[:48], out[48:64]
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
dfm._patch_model(model, [1, 6, 11, 16, 22, 27, 32, 37])     # the DFlash taps, as the loop
snap = [list(c.cache) if isinstance(c, ArraysCache) else None for c in cache]
def restore(S):
    for c, s in zip(cache, snap):
        if s is None:
            c.trim(S)
        else:
            c.cache = list(s)
inp = mx.array([[ctx[-1]] + blk[:15]])

def time_pass(gdnk, reps=15):
    os.environ["DFL35_GDNK"] = gdnk
    cap = dfm._GDNStateCapture()
    try:
        ts = []
        for k in range(reps + 2):
            cap.clear()
            mx.synchronize(); t = time.perf_counter()
            lg = model(inp, cache=cache)
            mx.eval(mx.argmax(lg, -1), [x for x in model._hidden_states if x is not None])
            dt = time.perf_counter() - t
            restore(16)
            if k >= 2:
                ts.append(dt)
        return sorted(ts)[len(ts) // 2] * 1e3, np.array(mx.argmax(lg, -1))
    finally:
        cap.close()
        os.environ["DFL35_GDNK"] = "0"

t0, a0 = time_pass("0"); t1, a1 = time_pass("1"); t0b, _ = time_pass("0"); t1b, _ = time_pass("1")
say(f"whole 16-row verify pass (serial, taps + capture): stock {t0:.2f} / {t0b:.2f} ms, GDN kernel program {t1:.2f} / {t1b:.2f} ms; "
    f"greedy tokens equal on {int((a0 == a1).sum())} / 16 rows")

# ---- 30 GDN mixers chained vs independent
import mlx.nn as nn
from qmm16 import qmm16
from gdn35_exact import prework35, gated_norm35, scales35, recurrence_gates
from mlx_lm.models import gated_delta as gd
L = TM.layers
lin = [i for i, l in enumerate(L) if l.is_linear]
pre = {i: (cache[i][0], cache[i][1]) for i in lin}
U = {}
h = TM.embed_tokens(inp)
fa_mask = create_attention_mask(h, cache[TM.fa_idx])
for i, (layer, c) in enumerate(zip(L, cache)):
    u = layer.input_layernorm(h)
    if layer.is_linear:
        U[i] = u
        h = h + layer.linear_attn(u, None, c)
    else:
        h = h + layer.self_attn(u, fa_mask, c)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
mx.eval(list(U.values()))
restore(16)
FUSED = {}
for i in lin:
    m = L[i].linear_attn
    FUSED[i] = tuple(mx.concatenate([getattr(w, n) for w in (m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a)], axis=0)
                     for n in ("weight", "scales", "biases"))
mx.eval(FUSED)
SC = scales35()
def kern(i, u):
    m = L[i].linear_attn; S = u.shape[1]; cs, st = pre[i]
    p = qmm16(u.reshape(S, -1), *FUSED[i], ks=2, nt=2)
    q, k, v, ns = prework35(p, cs, m.conv1d.weight, SC)
    y, st2 = recurrence_gates(q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 32, 128), p, m.A_log,
                              m.dt_bias, st, r=8)
    o = gated_norm35(y, p, m.norm.weight, m.norm.eps, zoff=8192)
    return qmm16(o.reshape(S, -1), m.out_proj.weight, m.out_proj.scales, m.out_proj.biases, ks=4, nt=2)[None]
def stock(i, u):
    m = L[i].linear_attn; B, S, _ = u.shape; cs, st = pre[i]
    qkv = m.in_proj_qkv(u); z = m.in_proj_z(u).reshape(B, S, 32, 128); b, a = m.in_proj_b(u), m.in_proj_a(u)
    co = nn.silu(m.conv1d(mx.concatenate([cs, qkv], axis=1)))
    q, k, v = [t.reshape(B, S, hh, 128) for t, hh in zip(mx.split(co, [2048, 4096], -1), (16, 16, 32))]
    s = 128 ** -0.5
    q = (s ** 2) * mx.fast.rms_norm(q, None, 1e-6); k = s * mx.fast.rms_norm(k, None, 1e-6)
    y, _ = gd.gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, st, None, use_kernel=True)
    return m.out_proj(m.norm(y, z).reshape(B, S, -1))
def timeit(fn, reps=21):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3
def chain(f):
    x = U[lin[0]]
    for i in lin:
        x = x + f(i, x)
    return x
for nm, f in (("stock", stock), ("kernel program", kern)):
    say(f"30 GDN mixers, {nm}: independent captured inputs {timeit(lambda: [f(i, U[i]) for i in lin]):.2f} ms, "
        f"chained (serial) {timeit(lambda: chain(f)):.2f} ms")
# per-layer serial costs: the other parts of the pass, chained the same way
def chain_attn():
    x = U[lin[0]]
    for i, layer in enumerate(L):
        if not layer.is_linear:
            x = x + layer.self_attn(x, fa_mask, cache[i])
    return x
def chain_moe():
    x = U[lin[0]]
    for layer in L:
        x = x + layer.mlp(x)
    return x
ta = timeit(chain_attn); restore(16)
say(f"chained: 10 attention mixers {ta:.2f} ms (KV grows during the reps), 40 MoE blocks {timeit(chain_moe):.2f} ms")
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/test_pass16.txt"), "w").write("\n".join(lines) + "\n")
