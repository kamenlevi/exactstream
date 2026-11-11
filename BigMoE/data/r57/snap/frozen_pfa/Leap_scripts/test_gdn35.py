"""L1-15 step 2 (Leap, Sat 2026-10-24): the fused 35B GDN pieces (gdn35_exact.py) vs MLX's eager mixer, bitwise, and their
time. Real data: all 30 GDN layers of Qwen3.6-35B-A3B 4-bit on one code context's 16 verify rows (as gdn16_profile.py),
S = 1, 2, 4, 8, 12, 16; plus random data at three scales for the prework and the gated norm.
Mixer variants, all timed over the 30 layers (median of 15):
  eager  = mlx_lm's ops (the reference),
  fused  = one in_proj matmul (qkv | z | b | a) + prework35 + MLX's beta / g + mlx_lm's recurrence + gated_norm35 + out_proj.
Usage: [SpecDecode venv] test_gdn35.py  -> ~/Leap/data/test_gdn35.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import numpy as np
import mlx.core as mx
import mlx.nn as nn
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_third_party/sd35"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
import dflash35_model_mlx as dfm
from gdn35_exact import prework35, gated_norm35, scales35, recurrence_tg
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models import gated_delta as gd

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def mism(a, b):
    a, b = np.array(a.astype(mx.float32)).reshape(-1), np.array(b.astype(mx.float32)).reshape(-1)
    return int((a.view(np.uint32) != b.view(np.uint32)).sum())

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

# ---------------------------------------------------------------- random data: prework and gated norm vs MLX's eager ops
conv = nn.Conv1d(8192, 8192, kernel_size=4, groups=8192, bias=False)
def eager_pre(qkv, cs, conv):
    S = qkv.shape[1]
    ci = mx.concatenate([cs, qkv], axis=1)
    co = nn.silu(conv(ci))
    q, k, v = [t.reshape(1, S, h, 128) for t, h in zip(mx.split(co, [2048, 4096], -1), [16, 16, 32])]
    s = k.shape[-1] ** -0.5
    return (s ** 2) * mx.fast.rms_norm(q, None, 1e-6), s * mx.fast.rms_norm(k, None, 1e-6), v, mx.contiguous(ci[:, -3:])

from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated
gn = Qwen3NextRMSNormGated(128, eps=1e-6)
SC = scales35()
say("Random data (bitwise mismatches q / k / v / state and gated norm; 0 = equal):")
bad = 0
for seed, (sx, sw) in enumerate(((1.0, 0.3), (3.0, 0.5), (0.2, 1.0))):
    mx.random.seed(seed)
    conv.weight = (mx.random.normal(conv.weight.shape) * sw).astype(mx.bfloat16)
    gn.weight = (1 + 0.2 * mx.random.normal((128,))).astype(mx.bfloat16)
    row = []
    for S in (1, 2, 3, 5, 8, 12, 16):
        qkv = (mx.random.normal((1, S, 8192)) * sx).astype(mx.bfloat16)
        cs = (mx.random.normal((1, 3, 8192)) * sx).astype(mx.bfloat16)
        r = eager_pre(qkv, cs, conv)
        f = prework35(qkv.reshape(S, 8192), cs.reshape(3, 8192), conv.weight.reshape(8192, 4), SC)
        y = (mx.random.normal((1, S, 32, 128)) * sx).astype(mx.bfloat16)
        z = (mx.random.normal((1, S, 32, 128)) * sx).astype(mx.bfloat16)
        m = [mism(a, b) for a, b in zip(r, f)] + [mism(gn(y, z), gated_norm35(y, z.reshape(S, 4096), gn.weight, 1e-6))]
        bad += sum(m); row.append(f"S{S} {m}")
    say(f"  scale x {sx} w {sw}: " + "  ".join(row))
say(f"  random data: {'ALL EQUAL' if bad == 0 else f'{bad} MISMATCHES'}")

# ---------------------------------------------------------------- real data
model, tok = dfm.load(MODEL)
mx.eval(model.parameters())
TM = model.language_model.model
for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
    r = json.loads(l)
    if r["key"].startswith("tf_code"):
        user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
        ids = list(ids if isinstance(ids[0], int) else ids["input_ids"]); key = r["key"]
        break
out = [int(r.token) for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=64, sampler=make_sampler(0.0))]
ctx, blk = ids + out[:48], out[48:64]
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
L = TM.layers
lin = [i for i, l in enumerate(L) if l.is_linear]
pre = {i: (cache[i][0], cache[i][1]) for i in lin}
U = {}
h = TM.embed_tokens(mx.array([[ctx[-1]] + blk[:15]]))
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
m0 = L[lin[0]].linear_attn
say(f"\nReal data: {key}, context {len(ctx)}, 30 GDN layers; norm weight {m0.norm.weight.dtype}, conv {m0.conv1d.weight.dtype}, "
    f"A_log {m0.A_log.dtype}, dt_bias {m0.dt_bias.dtype}")

FUSED = {}
for i in lin:
    m = L[i].linear_attn
    ws = [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a]
    FUSED[i] = tuple(mx.concatenate([getattr(w, n) for w in ws], axis=0) for n in ("weight", "scales", "biases"))
mx.eval(FUSED)
GS, BITS = m0.in_proj_qkv.group_size, m0.in_proj_qkv.bits

def eager(m, u, cs, st):
    """GatedDeltaNet.__call__ without the cache bookkeeping (B = 1, no mask)"""
    B, S, _ = u.shape
    qkv = m.in_proj_qkv(u)
    z = m.in_proj_z(u).reshape(B, S, m.num_v_heads, m.head_v_dim)
    b, a = m.in_proj_b(u), m.in_proj_a(u)
    ci = mx.concatenate([cs, qkv], axis=1)
    co = nn.silu(m.conv1d(ci))
    q, k, v = [t.reshape(B, S, hh, d) for t, hh, d in zip(mx.split(co, [m.key_dim, 2 * m.key_dim], -1),
                                                          [m.num_k_heads, m.num_k_heads, m.num_v_heads],
                                                          [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    s = k.shape[-1] ** -0.5
    q = (s ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = s * mx.fast.rms_norm(k, None, 1e-6)
    y, st2 = gd.gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, st, None, use_kernel=True)
    o = m.norm(y, z)
    return m.out_proj(o.reshape(B, S, -1)), st2, mx.contiguous(ci[:, -3:])

def gates(i, u, cs):
    m = L[i].linear_attn
    S = u.shape[1]
    w, sc, bi = FUSED[i]
    p = mx.quantized_matmul(u.reshape(S, -1), w, sc, bi, transpose=True, group_size=GS, bits=BITS)   # [S, 12352]
    q, k, v, ns = prework35(p, cs, m.conv1d.weight, SC)
    b, a = p[:, 12288:12320], p[:, 12320:12352]
    beta = mx.sigmoid(b)[None]; g = gd.compute_g(m.A_log, a, m.dt_bias)[None]
    return p, q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 32, 128), g, beta, ns

def fused(i, u, cs, st, rec=gd.gated_delta_kernel):
    m = L[i].linear_attn
    S = u.shape[1]
    p, q, k, v, g, beta, ns = gates(i, u, cs)
    y, st2 = rec(q, k, v, g, beta, st)
    o = gated_norm35(y, p, m.norm.weight, m.norm.eps, zoff=8192)
    return m.out_proj(o.reshape(1, S, -1)), st2, ns.reshape(1, 3, 8192)

say("Fused mixer vs eager mixer, same S (bitwise mismatches out / state / conv state over the 30 layers; 0 = equal), and time:")
for S in (1, 2, 4, 8, 12, 16):
    mo = ms = mc = 0
    for i in lin:
        u = U[i][:, :S]
        e = eager(L[i].linear_attn, u, *pre[i]); f = fused(i, u, *pre[i])
        mo += mism(e[0], f[0]); ms += mism(e[1], f[1]); mc += mism(e[2], f[2])
    te = timeit(lambda: [eager(L[i].linear_attn, U[i][:, :S], *pre[i])[0] for i in lin])
    tf = timeit(lambda: [fused(i, U[i][:, :S], *pre[i])[0] for i in lin])
    tt = timeit(lambda: [fused(i, U[i][:, :S], *pre[i], rec=recurrence_tg)[0] for i in lin])
    # the recurrence alone: threadgroup-preloaded vs mlx_lm's kernel on the same inputs
    G = {i: gates(i, U[i][:, :S], pre[i][0]) for i in lin}; mx.eval(G)
    my = mst = 0
    for i in lin:
        _, q, k, v, g, beta, _ = G[i]
        a1, b1 = gd.gated_delta_kernel(q, k, v, g, beta, pre[i][1]); a2, b2 = recurrence_tg(q, k, v, g, beta, pre[i][1])
        my += mism(a1, a2); mst += mism(b1, b2)
    r1 = timeit(lambda: [x for i in lin for x in gd.gated_delta_kernel(*G[i][1:6], pre[i][1])])
    r2 = timeit(lambda: [x for i in lin for x in recurrence_tg(*G[i][1:6], pre[i][1])])
    mo2 = sum(mism(eager(L[i].linear_attn, U[i][:, :S], *pre[i])[0], fused(i, U[i][:, :S], *pre[i], rec=recurrence_tg)[0]) for i in lin)
    say(f"  S = {S:2d}: mismatches {mo} / {ms} / {mc}{'  EQUAL' if mo + ms + mc == 0 else ''};  eager {te:6.2f} ms, fused {tf:6.2f} ms, "
        f"fused + tg recurrence {tt:6.2f} ms (out mismatches {mo2}) (30 layers)")
    say(f"           recurrence alone: mlx_lm {r1:5.2f} ms, tg {r2:5.2f} ms; tg vs mlx_lm mismatches y {my}, state {mst}")
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/test_gdn35.txt"), "w").write("\n".join(lines) + "\n")
