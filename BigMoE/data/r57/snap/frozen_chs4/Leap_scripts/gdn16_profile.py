"""L1-15 step 1 (Leap, Sat 2026-10-24): where the 35B's 16-row GDN mixer time goes, op by op, and which ops are
batch-invariant (row t of an S-row call == the same row run as plain decode, bitwise).
Qwen3.6-35B-A3B 4-bit (mlx_lm qwen3_5 GatedDeltaNet), one real code context (EvalSuite tf_code item 0 + 48 greedy
tokens), the next 16 greedy tokens as the verify rows. Each op group is timed over all 30 GDN layers in one eval (captured
mixer inputs and pre-call cache states, nothing mutated), median of 15, at S = 1, 2, 4, 8, 12, 16.
Invariance: each op on identical inputs, S = 16 (and 8) rows vs 1-row calls (sequential ones threading the conv and
recurrent state, as plain decode does).
Usage: [SpecDecode venv] gdn16_profile.py  -> ~/Leap/data/gdn16_profile.txt"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights only: no downloads
import numpy as np
import mlx.core as mx
import mlx.nn as nn
mx.set_cache_limit(4 << 30)
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_third_party/sd35"))
import dflash35_model_mlx as dfm
from mlx_lm import stream_generate as lm_stream
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import gated_delta as gd

MODEL = os.path.expanduser("~/.cache/huggingface/hub/models--Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed/snapshots/"
                           "95460516fbfe34c48f577f67c9ed191f5d61fbaf")
BW = 280e9
model, tok = dfm.load(MODEL)
mx.eval(model.parameters())
LM = model.language_model; TM = LM.model
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def code_prompt():
    for l in open(os.path.expanduser("~/EvalSuite/cache/ref/tf_gen.jsonl")):
        r = json.loads(l)
        if r["key"].startswith("tf_code"):
            user = tok.decode(r["prompt_ids"]).split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                          enable_thinking=False)
            return r["key"], list(ids if isinstance(ids[0], int) else ids["input_ids"])

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

def pbytes(mod):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(mod.parameters()))

# ---------------------------------------------------------------- inputs: mixer inputs + pre-call states
key, ids = code_prompt()
out = [int(r.token) for r in lm_stream(model, tok, prompt=mx.array(ids), max_tokens=64, sampler=make_sampler(0.0))]
ctx, blk = ids + out[:48], out[48:64]
cache = make_prompt_cache(model)
mx.eval(model(mx.array([ctx[:-1]]), cache=cache))
L = TM.layers
lin = [i for i, l in enumerate(L) if l.is_linear]
pre = {i: (cache[i][0], cache[i][1]) for i in lin}     # conv state [1, 3, 8192] bf16, recurrent state [1, 32, 128, 128]
U = {}
h = TM.embed_tokens(mx.array([[ctx[-1]] + blk[:15]]))
from mlx_lm.models.base import create_attention_mask
fa_mask = create_attention_mask(h, cache[TM.fa_idx])
for i, (layer, c) in enumerate(zip(L, cache)):
    u = layer.input_layernorm(h)
    if layer.is_linear:
        U[i] = u
        h = h + layer.linear_attn(u, None, c)
    else:
        h = h + layer.self_attn(u, fa_mask, c)
    h = h + layer.mlp(layer.post_attention_layernorm(h))
mx.eval(list(U.values()), [p for v in pre.values() for p in v])
m0 = L[lin[0]].linear_attn
say(f"L1-15 step 1: {key}, context {len(ctx)} tokens, {len(lin)} GDN layers; per layer: in_proj_qkv {m0.in_proj_qkv.weight.shape}, "
    f"z {m0.in_proj_z.weight.shape}, b/a {m0.in_proj_b.weight.shape}, out_proj {m0.out_proj.weight.shape} (packed uint32), "
    f"conv {m0.conv1d.weight.shape} {m0.conv1d.weight.dtype}, state {pre[lin[0]][1].shape} {pre[lin[0]][1].dtype}")
W_BYTES = {"in_proj_qkv": pbytes(m0.in_proj_qkv), "in_proj_z": pbytes(m0.in_proj_z),
           "in_proj_b + a": pbytes(m0.in_proj_b) + pbytes(m0.in_proj_a), "out_proj": pbytes(m0.out_proj)}
say("  weight bytes per layer: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in W_BYTES.items())
    + f"; mixer total {pbytes(m0) / 1e6:.2f} MB -> 30-layer floor {30 * pbytes(m0) / BW * 1e3:.2f} ms at 280 GB/s")

def ops(m, u, cs, st):
    B, S, _ = u.shape
    r = {}
    r["qkv"] = m.in_proj_qkv(u)
    r["z"] = m.in_proj_z(u).reshape(B, S, m.num_v_heads, m.head_v_dim)
    r["b"], r["a"] = m.in_proj_b(u), m.in_proj_a(u)
    r["conv_in"] = mx.concatenate([cs, r["qkv"]], axis=1)
    r["conv_out"] = nn.silu(m.conv1d(r["conv_in"]))
    q, k, v = [t.reshape(B, S, hh, d) for t, hh, d in zip(mx.split(r["conv_out"], [m.key_dim, 2 * m.key_dim], -1),
                                                          [m.num_k_heads, m.num_k_heads, m.num_v_heads],
                                                          [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    inv_scale = k.shape[-1] ** -0.5
    r["q"] = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    r["k"] = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    r["v"] = v
    r["beta"] = mx.sigmoid(r["b"]); r["g"] = gd.compute_g(m.A_log, r["a"], m.dt_bias)
    r["y"], r["st"] = gd.gated_delta_kernel(r["q"], r["k"], r["v"], r["g"], r["beta"], st)
    r["o"] = m.norm(r["y"], r["z"])
    r["out"] = m.out_proj(r["o"].reshape(B, S, -1))
    r["cs_new"] = r["conv_in"][:, -(m.conv_kernel_size - 1):]
    return r

def split_qk(m, co):
    B, S, _ = co.shape
    q, k, v = [t.reshape(B, S, hh, d) for t, hh, d in zip(mx.split(co, [m.key_dim, 2 * m.key_dim], -1),
                                                          [m.num_k_heads, m.num_k_heads, m.num_v_heads],
                                                          [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    s = k.shape[-1] ** -0.5
    return (s ** 2) * mx.fast.rms_norm(q, None, 1e-6), s * mx.fast.rms_norm(k, None, 1e-6), v

# fused in_proj: qkv | z | b | a concatenated along the outputs (lossless: per-output-row quantization)
FUSED = {}
for i in lin:
    m = L[i].linear_attn
    ws = [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a]
    FUSED[i] = tuple(mx.concatenate([getattr(w, n) for w in ws], axis=0) for n in ("weight", "scales", "biases"))
mx.eval(FUSED)
GS, BITS = m0.in_proj_qkv.group_size, m0.in_proj_qkv.bits
def fused_in(i, u):
    w, s, b = FUSED[i]
    return mx.quantized_matmul(u, w, s, b, transpose=True, group_size=GS, bits=BITS)

# ---------------------------------------------------------------- timing
say("\nTiming: ms for all 30 GDN layers (median of 15), and % of the bytes floor at 280 GB/s")
REF = {}
for S in (1, 2, 4, 8, 12, 16):
    R = {i: ops(L[i].linear_attn, U[i][:, :S], *pre[i]) for i in lin}
    mx.eval([list(r.values()) for r in R.values()])
    REF[S] = R
    A = lambda i: L[i].linear_attn
    groups = {
        "in_proj_qkv": lambda: [A(i).in_proj_qkv(U[i][:, :S]) for i in lin],
        "in_proj_z": lambda: [A(i).in_proj_z(U[i][:, :S]) for i in lin],
        "in_proj_b + a": lambda: [x for i in lin for x in (A(i).in_proj_b(U[i][:, :S]), A(i).in_proj_a(U[i][:, :S]))],
        "conv1d + silu (+ concat)": lambda: [nn.silu(A(i).conv1d(R[i]["conv_in"])) for i in lin],
        "split + q/k rms_norm + scale": lambda: [x for i in lin for x in split_qk(A(i), R[i]["conv_out"])],
        "beta + g": lambda: [x for i in lin for x in (mx.sigmoid(R[i]["b"]), gd.compute_g(A(i).A_log, R[i]["a"], A(i).dt_bias))],
        "recurrence (gated_delta kernel)": lambda: [x for i in lin for x in gd.gated_delta_kernel(R[i]["q"], R[i]["k"], R[i]["v"], R[i]["g"], R[i]["beta"], pre[i][1])],
        "gated norm": lambda: [A(i).norm(R[i]["y"], R[i]["z"]) for i in lin],
        "out_proj": lambda: [A(i).out_proj(R[i]["o"].reshape(1, S, -1)) for i in lin],
    }
    res = {k: timeit(f) for k, f in groups.items()}
    res["WHOLE mixer chain"] = timeit(lambda: [ops(A(i), U[i][:, :S], *pre[i])["out"] for i in lin])
    res["fused in_proj (qkv|z|b|a, one matmul)"] = timeit(lambda: [fused_in(i, U[i][:, :S]) for i in lin])
    res["in_proj_qkv as S x 1-row calls (qmv)"] = timeit(lambda: [A(i).in_proj_qkv(U[i][:, :S].reshape(S, 1, -1)) for i in lin])
    say(f"S = {S:2d}:")
    for k, v in res.items():
        fl = 30 * W_BYTES[k] / BW * 1e3 if k in W_BYTES else None
        say(f"  {k:42s} {v:7.3f} ms" + (f"   floor {fl:5.2f} ms ({fl / v:4.0%})" if fl else ""))
    say(f"  {'sum of the op groups':42s} {sum(v for k, v in res.items() if k in groups):7.3f} ms")

# ---------------------------------------------------------------- batch invariance
def bits_diff(x, y):
    x, y = np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
    return int((x.view(np.uint32) != y.view(np.uint32)).sum()), x.size

def seq1(i, upto):
    """plain decode: `upto` 1-row calls threading the conv state and the recurrent state"""
    m = L[i].linear_attn; cs, st = pre[i]; rs = []
    for t in range(upto):
        r = ops(m, U[i][:, t:t + 1], cs, st)
        cs, st = r["cs_new"], r["st"]
        rs.append(r)
    return rs

say("\nBatch invariance, all 30 GDN layers (mismatching elements / total; 0 = bitwise equal)")
for S in (16, 8):
    tot = {}
    def add(k, d):
        a, b = tot.get(k, (0, 0)); tot[k] = (a + d[0], b + d[1])
    for i in lin:
        m = L[i].linear_attn; R = REF[S][i]; u = U[i][:, :S]
        rs = seq1(i, S)
        cat = lambda key: mx.concatenate([r[key] for r in rs], axis=1)
        # whole chain: S-row call vs S sequential 1-row calls (plain decode)
        for k in ("qkv", "z", "conv_out", "q", "k", "y", "out"):
            add(f"chain vs plain decode: {k}", bits_diff(R[k], cat(k)))
        add("chain vs plain decode: final state", bits_diff(R["st"], rs[-1]["st"]))
        # op by op on identical inputs
        for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "out_proj"):
            mod = getattr(m, nm); x = R["o"].reshape(1, S, -1) if nm == "out_proj" else u
            full = mod(x)
            add(f"op {nm}: S rows vs 1-row (qmv)", bits_diff(full, mx.concatenate([mod(x[:, t:t + 1]) for t in range(S)], 1)))
            add(f"op {nm}: S rows vs 2-row (band 2-12)", bits_diff(full, mx.concatenate([mod(mx.concatenate([x[:, t:t + 1]] * 2, 1))[:, :1] for t in range(S)], 1)))
        add("op in_proj_qkv: (S, 1, K) batched vs 1-row", bits_diff(m.in_proj_qkv(u.reshape(S, 1, -1)).reshape(1, S, -1), mx.concatenate([m.in_proj_qkv(u[:, t:t + 1]) for t in range(S)], 1)))
        fi = fused_in(i, u)
        add("op fused in_proj vs separate (same S)", bits_diff(fi, mx.concatenate([R["qkv"], R["z"].reshape(1, S, -1), R["b"], R["a"]], -1)))
        if S == 16:      # padding independence of the 16-row kernel: row t alone at row 0, other rows zero
            for t in (0, 5, 15):
                xp = mx.concatenate([u[:, t:t + 1], mx.zeros((1, 15, u.shape[-1]), u.dtype)], 1)
                add("op in_proj_qkv: 16-row padded vs 16-row real", bits_diff(m.in_proj_qkv(xp)[:, :1], R["qkv"][:, t:t + 1]))
        # conv + silu on identical qkv rows, sequential 1-row windows
        cs = pre[i][0]; co = []
        for t in range(S):
            ci = mx.concatenate([cs, R["qkv"][:, t:t + 1]], 1); co.append(nn.silu(m.conv1d(ci))); cs = ci[:, -3:]
        add("op conv1d + silu: S rows vs sequential 1-row", bits_diff(R["conv_out"], mx.concatenate(co, 1)))
        q1, k1, _ = zip(*[split_qk(m, R["conv_out"][:, t:t + 1]) for t in range(S)])
        add("op q/k rms_norm: S rows vs 1-row", bits_diff(mx.concatenate([R["q"], R["k"]], -1), mx.concatenate([mx.concatenate(q1, 1), mx.concatenate(k1, 1)], -1)))
        st = pre[i][1]; ys = []
        for t in range(S):
            sl = slice(t, t + 1)
            y, st = gd.gated_delta_kernel(R["q"][:, sl], R["k"][:, sl], R["v"][:, sl], R["g"][:, sl], R["beta"][:, sl], st)
            ys.append(y)
        add("op recurrence: T = S vs S x T = 1 (outputs)", bits_diff(R["y"], mx.concatenate(ys, 1)))
        add("op recurrence: T = S vs S x T = 1 (final state)", bits_diff(R["st"], st))
        add("op gated norm: S rows vs 1-row", bits_diff(R["o"], mx.concatenate([m.norm(R["y"][:, t:t + 1], R["z"][:, t:t + 1]) for t in range(S)], 1)))
    say(f"S = {S}:")
    for k, (a, b) in tot.items():
        say(f"  {k:52s} {a:9d} / {b:9d}" + ("   EQUAL" if a == 0 else ""))
say(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
open(os.path.expanduser("~/Leap/data/gdn16_profile.txt"), "w").write("\n".join(lines) + "\n")
