"""L1-31 screen (Leap, Sun 2026-10-25): the 125B's GDN core per 16-row check (36 layers), as the engine runs it for
S > 6 rows (ref/qwen4_exp.py GatedDeltaNet stock branch): concat [conv_state; qkv] -> depthwise conv1d k 4 -> silu ->
split q | k | v -> fp32 l2norm (q also x 128^-0.5) -> mlx_lm gated_delta_update (gates inside) -> SigmoidRMSNormGated
(rms_norm with weight, then fp32 sigmoid(z) x). Plus capture-commit's replay of the recurrence over the accepted rows
(T 10) and MTPLX's fused conv+norm rows kernel at S 16 if it runs. Random inputs of the real shapes, a separate weight
and state set per layer, each piece over all 36 layers in one eval, median of 15.
Usage: [BigMoE venv] gdn31.py -> ~/Leap/data/gdn31.txt"""
import os, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update

NL, S, HK, HV, D, CK = 36, 16, 16, 48, 128, 4
KEY, VAL = HK * D, HV * D
CONV = 2 * KEY + VAL
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

mx.random.seed(31)
bf = mx.bfloat16
L = []
for _ in range(NL):
    conv = nn.Conv1d(CONV, CONV, kernel_size=CK, groups=CONV, bias=False)
    conv.weight = (mx.random.normal(conv.weight.shape) * 0.3).astype(bf)
    L.append(dict(conv=conv, A_log=mx.random.normal((HV,)) * 0.5, dt_bias=mx.random.normal((HV,)) * 0.5,
                  nw=(1 + 0.1 * mx.random.normal((D,))).astype(bf),
                  cs=(mx.random.normal((1, CK - 1, CONV)) * 0.5).astype(bf),
                  st=(mx.random.normal((1, HV, D, D)) * 0.05).astype(mx.float32),
                  qkv=(mx.random.normal((1, S, CONV)) * 0.5).astype(bf),
                  z=(mx.random.normal((1, S, HV, D)) * 0.5).astype(bf),
                  a=(mx.random.normal((1, S, HV)) * 0.5).astype(bf), b=(mx.random.normal((1, S, HV)) * 0.5).astype(bf)))
mx.eval([list(d.values())[1:] for d in L], [d["conv"].weight for d in L])

def prework(d, s=S):
    ci = mx.concatenate([d["cs"], d["qkv"][:, :s]], axis=1)
    keep = mx.contiguous(ci[:, -(CK - 1):, :])
    co = nn.silu(d["conv"](ci))
    q, k, v = [t.reshape(1, s, h, D) for t, h in zip(mx.split(co, [KEY, 2 * KEY], -1), [HK, HK, HV])]
    def l2(x):
        xf = x.astype(mx.float32)
        return (xf * mx.rsqrt((xf * xf).sum(-1, keepdims=True) + 1e-6)).astype(x.dtype)
    return (D ** -0.5) * l2(q), l2(k), v, keep

PW = [prework(d) for d in L]
mx.eval(PW)
def rec(i, t=S):
    q, k, v, _ = PW[i]
    d = L[i]
    return gated_delta_update(q[:, :t], k[:, :t], v[:, :t], d["a"][:, :t], d["b"][:, :t], d["A_log"], d["dt_bias"], d["st"], None)
OUT = [rec(i)[0] for i in range(NL)]
mx.eval(OUT)
def gnorm(i):
    x = mx.fast.rms_norm(OUT[i], L[i]["nw"], 1e-6)
    g = mx.sigmoid(L[i]["z"].astype(mx.float32))
    return (g * x.astype(mx.float32)).astype(bf)

t_pre = timeit(lambda: [prework(d) for d in L])
t_rec = timeit(lambda: [rec(i) for i in range(NL)])
t_rep = timeit(lambda: [rec(i, 10) for i in range(NL)])
t_gn = timeit(lambda: [gnorm(i) for i in range(NL)])
def chain(d):
    q, k, v, keep = prework(d)
    o, st = gated_delta_update(q, k, v, d["a"], d["b"], d["A_log"], d["dt_bias"], d["st"], None)
    x = mx.fast.rms_norm(o, d["nw"], 1e-6)
    return (mx.sigmoid(d["z"].astype(mx.float32)) * x.astype(mx.float32)).astype(bf), st, keep
t_all = timeit(lambda: [chain(d) for d in L])
t_rec1 = timeit(lambda: [gated_delta_update(PW[i][0][:, :1], PW[i][1][:, :1], PW[i][2][:, :1], L[i]["a"][:, :1],
                                             L[i]["b"][:, :1], L[i]["A_log"], L[i]["dt_bias"], L[i]["st"], None)
                         for i in range(NL)])
fl = NL * (2 * HV * D * D * 4 + 2 * S * CONV * 2 + 3 * S * VAL * 2) / 280e9 * 1e3
say(f"L1-31: 125B GDN core per 16-row check, {NL} layers (bytes floor {fl:.2f} ms)")
say(f"  prework (concat, conv1d, silu, split, l2norm)  {t_pre:.2f} ms")
say(f"  recurrence gated_delta_update T 16             {t_rec:.2f} ms   (T 1: {t_rec1:.2f}; replay T 10: {t_rep:.2f})")
say(f"  gated norm (rms_norm + fp32 sigmoid gate)      {t_gn:.2f} ms")
say(f"  all three in one eval                          {t_all:.2f} ms; + replay {t_all + t_rep:.2f} ms")
try:
    from mtplx.kernels.gdn_conv_norm import fused_gdn_conv_norm_rows
    t_f = timeit(lambda: [fused_gdn_conv_norm_rows(d["qkv"].reshape(S, -1), d["cs"].reshape(CK - 1, CONV),
                                                    d["conv"].weight) for d in L])
    q_f, k_f, v_f, _ = fused_gdn_conv_norm_rows(L[0]["qkv"].reshape(S, -1), L[0]["cs"].reshape(CK - 1, CONV), L[0]["conv"].weight)
    e = max(float(mx.abs(q_f.reshape(PW[0][0].shape).astype(mx.float32) - PW[0][0].astype(mx.float32)).max()),
            float(mx.abs(v_f.reshape(PW[0][2].shape).astype(mx.float32) - PW[0][2].astype(mx.float32)).max()))
    say(f"  MTPLX fused conv+norm rows at S 16             {t_f:.2f} ms (max |diff| vs stock {e:.3g})")
except Exception as ex:
    say(f"  MTPLX fused conv+norm rows at S 16: not runnable ({type(ex).__name__}: {str(ex)[:120]})")
open(os.path.expanduser("~/Leap/data/gdn31.txt"), "w").write("\n".join(lines) + "\n")
