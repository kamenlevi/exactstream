"""L1-15 decisive test (Leap, Sat 2026-10-24, pre-registered in ~/Leap/GDN16_PREREG.md): the 35B's 30 GatedDeltaNet mixers
at 16 rows with the batch-invariant kernel program, ms, and spec == plain bitwise: every row of one 16-row call vs 16
sequential 1-row calls of the SAME kernels (plain decode threading the conv and recurrent state).
Kernel program (one set of kernels for every row count):
  qmm16(x, in_proj qkv | z | b | a)  -> prework35 (conv, silu, q/k rms_norm, scales, new conv state)
  -> MLX sigmoid(b) / compute_g(a) -> recurrence_r4 -> gated_norm35 -> qmm16(o, out_proj)
Real data: Qwen3.6-35B-A3B 4-bit, HumanEval/103 context + 48 greedy tokens, the next 16 greedy tokens as verify rows.
Pass: <= 4 ms with spec == plain bitwise. Kill: > 6 ms, or not batch-invariant.
Usage: [SpecDecode venv] test_mixer16.py  -> ~/Leap/data/test_mixer16.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import numpy as np
import mlx.core as mx
import mlx.nn as nn
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/sd35"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
import dflash35_model_mlx as dfm
from gdn35_exact import prework35, gated_norm35, scales35, recurrence_r4, recurrence_gates
from qmm16 import qmm16, qmm16c
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
def timeit(fn, reps=21):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

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
say(f"L1-15 decisive test: {key}, context {len(ctx)}, 30 GDN mixers, real verify rows")

FUSED = {}
for i in lin:
    m = L[i].linear_attn
    ws = [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a]
    FUSED[i] = tuple(mx.concatenate([getattr(w, n) for w in ws], axis=0) for n in ("weight", "scales", "biases"))
mx.eval(FUSED)
SC = scales35()

def mixer(i, u, cs, st, ks_in=1, ks_out=2, nt_in=1, nt_out=1, rec="r4", r=4, mm="b"):
    """u [1, S, 2048] -> out [1, S, 2048], new recurrent state, new conv state; the same kernels at every S.
    rec "r4": MLX's sigmoid / compute_g + recurrence_r4; "gates": recurrence_gates (beta / g inside, r rows per thread).
    mm "b": qmm16 (bf16-rounded dequantized weights); "c": qmm16c (code MMA, fp32-exact dequantization)."""
    m = L[i].linear_attn
    S = u.shape[1]
    qmm = qmm16c if mm == "c" else qmm16
    p = qmm(u.reshape(S, -1), *FUSED[i], ks=ks_in, nt=nt_in)            # [S, 12352]: qkv | z | b | a
    q, k, v, ns = prework35(p, cs, m.conv1d.weight, SC)
    q, k, v = q.reshape(1, S, 16, 128), k.reshape(1, S, 16, 128), v.reshape(1, S, 32, 128)
    if rec == "r4":
        beta = mx.sigmoid(p[:, 12288:12320])[None]; g = gd.compute_g(m.A_log, p[:, 12320:12352], m.dt_bias)[None]
        y, st2 = recurrence_r4(q, k, v, g, beta, st)
    else:
        y, st2 = recurrence_gates(q, k, v, p, m.A_log, m.dt_bias, st, r=r)
    o = gated_norm35(y, p, m.norm.weight, m.norm.eps, zoff=8192)
    wo = m.out_proj
    return qmm(o.reshape(S, -1), wo.weight, wo.scales, wo.biases, ks=ks_out, nt=nt_out)[None], st2, ns.reshape(1, 3, 8192)

CFGS = json.loads(os.environ.get("MIX_CFGS", "null")) or [
    dict(ks_in=1, ks_out=2), dict(ks_in=2, ks_out=2), dict(ks_in=1, ks_out=2, rec="gates", r=4),
    dict(ks_in=1, ks_out=2, rec="gates", r=8), dict(ks_in=1, ks_out=2, nt_in=4, nt_out=2, rec="gates", r=4),
    dict(ks_in=1, ks_out=8, nt_in=2, rec="gates", r=4)]

def eager(m, u, cs, st):
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
    return m.out_proj(m.norm(y, z).reshape(B, S, -1)), st2

# ---------------------------------------------------------------- MIX_SPLIT=1: Python build vs eval, and mx.compile
# The pre-registered number is the L1-13 method (build + eval, as above); this says what an engine without Python per-op
# overhead would see, for stock and the kernel program alike.
if os.environ.get("MIX_SPLIT"):
    cfg = CFGS[0]
    say(f"build vs eval split, 30 mixers, config {cfg} (median of 21)")
    def split(fn, reps=21):
        mx.eval(fn())
        b, e = [], []
        for _ in range(reps):
            mx.synchronize(); t0 = time.perf_counter(); o_ = fn(); t1 = time.perf_counter(); mx.eval(o_); t2 = time.perf_counter()
            b.append(t1 - t0); e.append(t2 - t1)
        return sorted(b)[reps // 2] * 1e3, sorted(e)[reps // 2] * 1e3
    for S in (1, 16):
        us = [U[i][:, :S] for i in lin]; cs_ = [pre[i][0] for i in lin]; sts = [pre[i][1] for i in lin]
        f_e = lambda us, cs_, sts: [eager(L[i].linear_attn, u, c, s)[0] for i, u, c, s in zip(lin, us, cs_, sts)]
        f_k = lambda us, cs_, sts: [mixer(i, u, c, s, **cfg)[0] for i, u, c, s in zip(lin, us, cs_, sts)]
        be, ee = split(lambda: f_e(us, cs_, sts)); bk, ek = split(lambda: f_k(us, cs_, sts))
        ce, ck = mx.compile(f_e), mx.compile(f_k)
        tce = timeit(lambda: ce(us, cs_, sts)); tck = timeit(lambda: ck(us, cs_, sts))
        same = sum(mism(a, b) for a, b in zip(ck(us, cs_, sts), f_k(us, cs_, sts)))
        say(f"  S = {S:2d}: stock: build {be:5.2f} + eval {ee:5.2f} = {be + ee:5.2f} ms, compiled {tce:5.2f} ms;  "
            f"kernel program: build {bk:5.2f} + eval {ek:5.2f} = {bk + ek:5.2f} ms, compiled {tck:5.2f} ms "
            f"(compiled vs eager bits: {same} differ)")
    open(os.path.expanduser("~/Leap/data/test_mixer16_split.txt"), "w").write("\n".join(lines) + "\n")
    sys.exit(0)

# ---------------------------------------------------------------- spec == plain, bitwise, S = 16 and S = 8
for cfg in (CFGS[0], CFGS[2]):
  for S in (16, 8):
    mo = ms_ = mc = 0; tot_o = tot_s = 0
    for i in lin:
        u = U[i][:, :S]
        o16, s16, c16 = mixer(i, u, *pre[i], **cfg)
        cs, st, outs = pre[i][0], pre[i][1], []
        for t in range(S):                                            # plain decode: 1 row at a time
            o1, st, cs = mixer(i, u[:, t:t + 1], cs, st, **cfg)
            outs.append(o1)
        mo += mism(o16, mx.concatenate(outs, 1)); ms_ += mism(s16, st); mc += mism(c16, cs)
        tot_o += o16.size; tot_s += s16.size
    say(f"{cfg} S = {S:2d}: spec (one {S}-row call) vs plain ({S} x 1-row calls): out {mo} / {tot_o} differ, final state "
        f"{ms_} / {tot_s}, conv state {mc}  -> {'BITWISE EQUAL' if mo + ms_ + mc == 0 else 'NOT EQUAL'}")
# the gate-fused recurrence vs MLX's gates: the same bits?
d = sum(mism(mixer(i, U[i], *pre[i], **CFGS[0])[0], mixer(i, U[i], *pre[i], **CFGS[2])[0]) for i in lin)
say(f"recurrence_gates vs MLX gates + recurrence_r4 (same matmul config), S = 16 outputs: {d} differ")

# ---------------------------------------------------------------- time, 30 mixers
say("Time, 30 mixers (median of 21):")
for S in (1, 8, 16):
    te = timeit(lambda: [eager(L[i].linear_attn, U[i][:, :S], *pre[i])[0] for i in lin])
    say(f"  S = {S:2d}: stock MLX eager {te:6.2f} ms")
    for cfg in CFGS:
        tm = timeit(lambda: [mixer(i, U[i][:, :S], *pre[i], **cfg)[0] for i in lin])
        say(f"          {cfg}: {tm:6.2f} ms")
# accuracy vs stock: the new numerics differ only by rounding order
d = []
for i in lin:
    e, _ = eager(L[i].linear_attn, U[i], *pre[i]); f, _, _ = mixer(i, U[i], *pre[i], **CFGS[2])
    d.append(float(mx.abs(e.astype(mx.float32) - f.astype(mx.float32)).max()) / max(1e-9, float(mx.abs(e.astype(mx.float32)).max())))
say(f"  vs stock MLX at S = 16: max |diff| / max |out| per layer: median {np.median(d):.2e}, max {max(d):.2e}")
# per-stage time of CFGS[2] at S = 16 (30 layers, stage inputs precomputed)
c = CFGS[2]; S = 16
P = {i: qmm16(U[i].reshape(S, -1), *FUSED[i], ks=c.get("ks_in", 1), nt=c.get("nt_in", 1)) for i in lin}; mx.eval(P)
Q = {i: prework35(P[i], pre[i][0], L[i].linear_attn.conv1d.weight, SC) for i in lin}; mx.eval(Q)
R4 = lambda i: [t.reshape(1, S, h, 128) for t, h in zip(Q[i][:3], (16, 16, 32))]
Y = {i: recurrence_gates(*R4(i), P[i], L[i].linear_attn.A_log, L[i].linear_attn.dt_bias, pre[i][1], r=c.get("r", 4)) for i in lin}
mx.eval(Y)
O = {i: gated_norm35(Y[i][0], P[i], L[i].linear_attn.norm.weight, L[i].linear_attn.norm.eps, zoff=8192) for i in lin}; mx.eval(O)
st_ = {"qmm16 in_proj": lambda: [qmm16(U[i].reshape(S, -1), *FUSED[i], ks=c.get("ks_in", 1), nt=c.get("nt_in", 1)) for i in lin],
       "prework35": lambda: [x for i in lin for x in prework35(P[i], pre[i][0], L[i].linear_attn.conv1d.weight, SC)],
       "recurrence_gates": lambda: [x for i in lin for x in recurrence_gates(*R4(i), P[i], L[i].linear_attn.A_log,
                                                                              L[i].linear_attn.dt_bias, pre[i][1], r=c.get("r", 4))],
       "gated_norm35": lambda: [gated_norm35(Y[i][0], P[i], L[i].linear_attn.norm.weight, L[i].linear_attn.norm.eps, zoff=8192) for i in lin],
       "qmm16 out_proj": lambda: [qmm16(O[i].reshape(S, -1), L[i].linear_attn.out_proj.weight, L[i].linear_attn.out_proj.scales,
                                        L[i].linear_attn.out_proj.biases, ks=c.get("ks_out", 2), nt=c.get("nt_out", 1)) for i in lin]}
say(f"  stages of {c} at S = 16: " + ", ".join(f"{k} {timeit(f):.2f} ms" for k, f in st_.items()))
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/test_mixer16.txt"), "w").write("\n".join(lines) + "\n")
