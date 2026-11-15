"""R-21 GDNB step 1 (NewIdeas' QUEUE row R-21; Leap, pre-registered Thu 2026-10-29 17:35 EET in ~/Leap/TABLE_350.md
'R-21 GDNB step 1'): how much of the commit's per-layer GDN replay cost can batching remove at all?
The commit (MTPLX commit_verified_window) replays each of the 35 pure-GDN layers alone: sigmoid, compute_g, the
gated-delta kernel, the conv-state concat + contiguous. R-12 measured ~0.07 ms fixed per layer (launch-bound).
Arms, on the 125B's GDN shapes (Hk 16, Hv 48, Dk = Dv 128, conv 10240 x 3; bf16 rows, fp32 states), synthetic values:
  per-layer: MTPLX's own replay code, 35 layers
  ideal:     the same ops once over inputs already stacked [35, ...] (B = 35 in the same kernels): the floor of any
             batched replay
  stacked:   ideal + mx.stack of the 35 layers' inputs inside the timing (the naive way to get there)
Each arm's lazy graph is built, then mx.eval is timed (GPU + encode), interleaved, 40 reps, at keep 1 / 4 / 10 / 15.
Bitwise: every layer's new state and conv state from 'ideal' and 'stacked' vs per-layer.
Lines (pre-registered): ideal saves < 1 ms per check at keep 10 -> KILL R-21 (no batched kernel can pass);
>= 2 ms AND bitwise -> build the batched kernel (the grouped multi-pointer one if 'stacked' gives the gain away).
Usage: [BigMoE venv] r21_screen.py -> ~/Leap/data/r21_screen.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update, gated_delta_kernel, compute_g

D = os.path.expanduser("~/Leap/data")
NL, Hk, Hv, Dk, Dv, CD, KC, R = 35, 16, 48, 128, 128, 10240, 4, 16
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
eqb = lambda a, b: a.shape == b.shape and bool(mx.all(mx.view(a.reshape(-1), mx.uint8) == mx.view(b.reshape(-1), mx.uint8)).item())

mx.random.seed(21)
bf = mx.bfloat16
L = []
for _ in range(NL):                                      # one verify window's captured rows + the layer's state
    q = (mx.random.normal((1, R, Hk, Dk)) * 0.09).astype(bf)
    k = (mx.random.normal((1, R, Hk, Dk)) * 0.09).astype(bf)
    L.append(dict(qkv=mx.random.normal((1, R, CD)).astype(bf), q=q, k=k,
                  v=mx.random.normal((1, R, Hv, Dv)).astype(bf), a=mx.random.normal((1, R, Hv)).astype(bf),
                  b=mx.random.normal((1, R, Hv)).astype(bf), A_log=(mx.random.normal((Hv,)) * 0.5).astype(bf),
                  dt_bias=mx.random.normal((Hv,)).astype(bf), conv=mx.random.normal((1, KC - 1, CD)).astype(bf),
                  st=(mx.random.normal((1, Hv, Dv, Dk)) * 0.02).astype(mx.float32)))
mx.eval([list(d.values()) for d in L])

def per_layer(keep):
    out = []
    for d in L:                                          # = commit_verified_window's "gdn" branch
        _, ns = gated_delta_update(d["q"][:, :keep], d["k"][:, :keep], d["v"][:, :keep], d["a"][:, :keep],
                                   d["b"][:, :keep], d["A_log"], d["dt_bias"], d["st"], None, use_kernel=True)
        ci = mx.concatenate([d["conv"], d["qkv"][:, :keep]], axis=1)
        out.append((mx.contiguous(ci[:, -(KC - 1):, :]), ns))
    return out

def stack_inputs(keep):
    S = {n: mx.contiguous(mx.concatenate([d[n][:, :keep] for d in L])) for n in ("q", "k", "v", "a", "b", "qkv")}
    S["A_log"] = mx.stack([d["A_log"] for d in L])[:, None]
    S["dt_bias"] = mx.stack([d["dt_bias"] for d in L])[:, None]
    S["conv"] = mx.concatenate([d["conv"] for d in L])
    S["st"] = mx.concatenate([d["st"] for d in L])
    return S

def batched(S):
    beta = mx.sigmoid(S["b"])                            # = gated_delta_update's own steps, B = 35
    g = compute_g(S["A_log"], S["a"], S["dt_bias"])
    _, ns = gated_delta_kernel(S["q"], S["k"], S["v"], g, beta, S["st"])
    ci = mx.concatenate([S["conv"], S["qkv"]], axis=1)
    cs = mx.contiguous(ci[:, -(KC - 1):, :])
    return [(cs[i:i + 1], ns[i:i + 1]) for i in range(NL)]

def timed(build):
    mx.synchronize()
    t0 = time.perf_counter()
    out = build()
    t1 = time.perf_counter()
    mx.eval(out)
    t2 = time.perf_counter()
    return out, 1e3 * (t1 - t0), 1e3 * (t2 - t1)

say(f"R-21 GDNB step 1: {NL} pure-GDN layers, Hk {Hk} Hv {Hv} Dk {Dk} Dv {Dv}, conv {CD} x {KC - 1}; state bytes read + "
    f"written per replay {2 * NL * Hv * Dv * Dk * 4 / 1e6:.0f} MB")
res, ok = {}, True
for keep in (1, 4, 10, 15):
    pre = stack_inputs(keep)
    mx.eval(pre)
    ref = per_layer(keep); mx.eval(ref)
    ide = batched(pre); mx.eval(ide)
    stk = batched(stack_inputs(keep)); mx.eval(stk)
    ok_i = all(eqb(a[0], b[0]) and eqb(a[1], b[1]) for a, b in zip(ref, ide))
    ok_s = all(eqb(a[0], b[0]) and eqb(a[1], b[1]) for a, b in zip(ref, stk))
    ok = ok and ok_i
    del ref, ide, stk
    T = {"per-layer": ([], []), "ideal": ([], []), "stacked": ([], [])}
    arms = {"per-layer": lambda: per_layer(keep), "ideal": lambda: batched(pre), "stacked": lambda: batched(stack_inputs(keep))}
    for _ in range(3):                                   # warm-up
        for f in arms.values():
            timed(f)
    for r in range(40):
        for n in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            _, cb, ev = timed(arms[n])
            T[n][0].append(cb); T[n][1].append(ev)
    m = {n: (float(np.median(v[0])), float(np.median(v[1]))) for n, v in T.items()}
    res[keep] = m
    say(f"  keep {keep:2d}: eval ms per-layer {m['per-layer'][1]:.3f}, ideal {m['ideal'][1]:.3f}, stacked "
        f"{m['stacked'][1]:.3f}; graph build ms {m['per-layer'][0]:.3f} / {m['ideal'][0]:.3f} / {m['stacked'][0]:.3f}; "
        f"ideal saves {m['per-layer'][1] - m['ideal'][1]:.3f} ms (+ build {m['per-layer'][0] - m['ideal'][0]:.3f}), "
        f"stacked saves {m['per-layer'][1] - m['stacked'][1]:.3f}; bitwise ideal {ok_i}, stacked {ok_s}")
    del pre
s10 = res[10]["per-layer"][1] - res[10]["ideal"][1]
v = "KILL" if s10 < 1 else ("BUILD" if s10 >= 2 and ok else "BETWEEN")
say(f"R-21 GDNB step 1 (ideal batched replay at keep 10 saves < 1 ms -> KILL; >= 2 ms and bitwise -> build): {v} "
    f"({s10:.2f} ms GPU-side, bitwise {ok})")
open(os.path.join(D, "r21_screen.txt"), "w").write("\n".join(lines) + "\n")
