"""R-61 GDN16-125 step 1 (Leap, Sun 2026-11-01 04:4x; pre-registered Sat 02:4x in ~/Leap/TABLE_350.md 'R-61 / R-62'; shared
GPU, rough): where do a 16-row check's GDN milliseconds go? Fit12's R-59 (C 224): GDN 24.1 ms per check for 36 layers,
5.7x its bytes floor. MTPLX's fused GDN kernels cover only 1 row (step, conv-norm, gated out) or 2-6 rows (conv-norm
rows), so a 16-row check runs the stock chain. One GatedDeltaNet built from the model's config (Hk 16, Hv 48, Dk = Dv =
128, conv 4, hidden 2560), random 4-bit g64 projections, REF2 on (16-row padding / qmm16c), no cache (zero conv / delta
state). Per piece, median ms of 50 (eval after each): P the 4 input projections; C the conv chain (concat, conv1d,
silu, split, l2norm x 2, scale); R gated_delta_update; O the gated norm + out_proj; W the whole mixer call. At S = 16
and S = 1, x 36 layers per check.
Usage: [BigMoE venv] r61_gdn_profile.py -> ~/Leap/data/r61_gdn_profile.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
import ref2
from mlx_lm.models.gated_delta import gated_delta_update
Q = ref2.Q

D = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
med = lambda v: float(np.median(v))

def timeit(f, n=50):
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    return 1e3 * med(ts)

cfg = json.load(open(os.path.expanduser("~/BigMoE/models/bare/config.json")))
args = Q.TextArgs.from_dict(cfg.get("text_config", cfg))
mx.random.seed(0)
g = Q.GatedDeltaNet(args)
g.set_dtype(mx.bfloat16)                                 # bf16 like the model: qmm16c needs bf16 scales / biases
nn.quantize(g, group_size=64, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
mx.eval(g.parameters())
ref2.apply()
say(f"R-61 step 1: MLX {mx.__version__}; GDN Hk {g.num_k_heads} Hv {g.num_v_heads} Dk {g.head_k_dim} Dv {g.head_v_dim} "
    f"conv {g.conv_kernel_size} key_dim {g.key_dim} conv_dim {g.conv_dim}; REF2 on")

for S in (16, 1):
    x = (mx.random.normal((1, S, args.hidden_size)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    def proj():
        return g.in_proj_qkv(x), g.in_proj_z(x).reshape(1, S, g.num_v_heads, g.head_v_dim), g.in_proj_b(x), g.in_proj_a(x)
    qkv, z, b, a = proj()
    mx.eval(qkv, z, b, a)
    conv_state = mx.zeros((1, g.conv_kernel_size - 1, g.conv_dim), dtype=x.dtype)
    def conv():
        ci = mx.concatenate([conv_state, qkv], axis=1)
        co = nn.silu(g.conv1d(ci))
        q_, k_, v_ = [t.reshape(1, S, h, d) for t, h, d in zip(mx.split(co, [g.key_dim, 2 * g.key_dim], -1),
                      [g.num_k_heads, g.num_k_heads, g.num_v_heads], [g.head_k_dim, g.head_k_dim, g.head_v_dim])]
        l2 = lambda t: (t.astype(mx.float32) * mx.rsqrt((t.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-6)).astype(t.dtype)
        return (g.head_k_dim ** -0.5) * l2(q_), l2(k_), v_
    q, k, v = conv()
    mx.eval(q, k, v)
    rec = lambda: gated_delta_update(q, k, v, a, b, g.A_log, g.dt_bias, None, None, use_kernel=True)
    out, _ = rec()
    mx.eval(out)
    outp = lambda: g.out_proj(g.norm(out, z).reshape(1, S, -1))
    whole = lambda: g(x, None, None)
    t = {n: timeit(f) for n, f in (("P", proj), ("C", conv), ("R", rec), ("O", outp), ("W", whole))}
    say(f"  S {S}: P projections {t['P']:.3f} | C conv chain {t['C']:.3f} | R recurrence {t['R']:.3f} | O norm + out_proj "
        f"{t['O']:.3f} | W whole {t['W']:.3f} ms per layer; x 36 layers: P {36 * t['P']:.1f}, C {36 * t['C']:.1f}, "
        f"R {36 * t['R']:.1f}, O {36 * t['O']:.1f}, W {36 * t['W']:.1f} ms per check")
say("R-61 step 1 done")
open(os.path.join(D, "r61_gdn_profile.txt"), "w").write("\n".join(lines) + "\n")
