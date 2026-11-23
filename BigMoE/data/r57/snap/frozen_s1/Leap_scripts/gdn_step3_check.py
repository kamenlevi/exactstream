"""GDN-STEP3 GPU check (Leap, Fri 2026-11-06 09:1x): Fable B's ~/FableGod/fableb/kernels/gdn_step3.py (read-only use) on the
reference's grid. Arms: var 0 (the reference loop in Fable B's harness) with abl 0 / 1 (no simd_sum) / 2 (k, q constants);
var 1 / 2 (fused reductions) only if abl 1 drops the time by >= 25 %. Bits for var 1 / 2 against mlx_lm's kernel.
Line (Fable B's): PASS <= 46 us chained at S 16, KILL > 53. -> ~/Leap/data/gdn_step3_check.txt"""
import importlib.util, os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

_spec = importlib.util.spec_from_file_location("fb_gdn_step3", os.path.expanduser("~/FableGod/fableb/kernels/gdn_step3.py"))
FB = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FB)
OUT = os.path.expanduser("~/Leap/data/gdn_step3_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nGDN-STEP3 check {time.strftime('%a %H:%M:%S')} ({os.environ.get('GS_COND', 'share lock')}; MLX {mx.__version__})")
rng = np.random.default_rng(9)
clock = time.perf_counter
def inputs(S):
    k = rng.standard_normal((1, S, 16, 128)).astype(np.float32); k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q = rng.standard_normal((1, S, 16, 128)).astype(np.float32); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= 128 ** -0.5
    v = rng.standard_normal((1, S, 48, 128)).astype(np.float32)
    g = rng.uniform(0.5, 1.0, (1, S, 48)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, (1, S, 48)).astype(np.float32)
    st = (rng.standard_normal((1, 48, 128, 128)) * 0.3).astype(np.float32)
    a = [mx.array(q).astype(mx.bfloat16), mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16),
         mx.array(g), mx.array(b).astype(mx.bfloat16), mx.array(st)]
    mx.eval(a)
    return a


def ref(q, k, v, g, b, st):
    return gated_delta_kernel(q, k, v, g, b, st)


def step3(var, abl, pf=0):
    def f(q, k, v, g, b, st):
        S = q.shape[1]
        kq = None
        if var == 2:
            kq = (q.astype(mx.float32) * k.astype(mx.float32)).sum(-1).reshape(S, 16)
        y, so = FB.gdn_step3(q.reshape(S, 16, 128), k.reshape(S, 16, 128), v.reshape(S, 48, 128), g.reshape(S, 48),
                             b.reshape(S, 48), st.reshape(48, 128, 128), S, var=var, abl=abl, kq=kq,
                             **({"pf": pf} if pf else {}))
        return y.reshape(1, S, 48, 128), so.reshape(1, 48, 128, 128)
    return f


def chain(fn, A):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for a in A:
            q = a[0] + (prev * 0).astype(a[0].dtype)
            if calls:
                y, so = fn(q, *a[1:])
                prev = y.reshape(-1)[:1].astype(mx.float32) + so.reshape(-1)[:1] * 0
            else:
                prev = q.reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(A)


def timed(arms, S):
    A = [inputs(S) for _ in range(36)]
    for f in arms.values():
        chain(f, A)
    acc = {k: [] for k in arms}
    keys = list(arms)
    for r in range(7):
        for k in (keys if r % 2 == 0 else keys[::-1]):
            acc[k].append(chain(arms[k], A))
    return {k: float(np.median(v)) for k, v in acc.items()}


def bits(fn, S):
    a = inputs(S)
    y0, s0 = ref(*a); y1, s1 = fn(*a)
    mx.eval(y0, s0, y1, s1)
    f = lambda x: np.array(x.astype(mx.float32))
    Y0, Y1, S0_, S1_ = f(y0), f(y1), f(s0), f(s1)
    return (float(np.abs(Y1 - Y0).max() / max(1e-30, np.abs(Y0).max())), float((Y1 != Y0).mean() * 100),
            float(np.abs(S1_ - S0_).max() / max(1e-30, np.abs(S0_).max())), bool(np.array_equal(S1_, S0_)))


if os.environ.get("GS_PF") == "1":                       # Fable B's prefetch ablation (pf bit mask), var 0, abl 0
    for pf in (1, 4, 3, 7):
        e = bits(step3(0, 0, pf), 16)
        say(f"  pf {pf}: bits vs mlx_lm at S 16: y {e[0]:.1e}, state {e[2]:.1e} (bitwise y {e[1] == 0.0}, state {e[3]})")
    t = timed({"mlx_lm": ref, "pf0": step3(0, 0), "pf1": step3(0, 0, 1), "pf4": step3(0, 0, 4), "pf3": step3(0, 0, 3),
               "pf7": step3(0, 0, 7)}, 16)
    say("  S 16 chained (us per call): " + ", ".join(f"{k} {v:.1f}" for k, v in t.items()))
    best = min(("pf1", "pf4", "pf3", "pf7"), key=lambda k: t[k])
    drop = 1.0 - t[best] / t["pf0"]
    t2 = timed({"mlx_lm": ref, "pf0": step3(0, 0), best: step3(0, 0, int(best[2:]))}, 2)
    say("  S 2 chained: " + ", ".join(f"{k} {v:.1f}" for k, v in t2.items()))
    say(f"RESULT {'GO' if drop >= 0.25 else 'STOP'} (Fable B's line: >= 25 % off pf 0; best {best} {100 * drop:.0f} %, "
        f"{36 * (t['mlx_lm'] - t[best]) / 1e3:+.2f} ms per 16-row check vs mlx_lm)")
    sys.exit(0)
e = bits(step3(0, 0), 16)
say(f"  var 0 (Fable B's reference loop) vs mlx_lm at S 16: y {e[0]:.1e} ({e[1]:.2f} % differ), state {e[2]:.1e} (bitwise {e[3]})")
t = timed({"mlx_lm": ref, "v0a0": step3(0, 0), "v0a1": step3(0, 1), "v0a2": step3(0, 2), "v0a3": step3(0, 3)}, 16)
say("  S 16 chained (us per call): " + ", ".join(f"{k} {v:.1f}" for k, v in t.items()))
drop = 1.0 - t["v0a1"] / t["v0a0"]
say(f"  abl 1 (no simd_sum) drops var 0 by {100 * drop:.0f} % (Fable B's gate for var 1 / 2: >= 25 %); abl 2 (k, q constants) "
    f"by {100 * (1.0 - t['v0a2'] / t['v0a0']):.0f} %")
if drop >= 0.25:
    for var in (1, 2):
        e = bits(step3(var, 0), 16)
        say(f"  var {var} bits vs mlx_lm at S 16: y {e[0]:.1e} ({e[1]:.2f} % differ), state {e[2]:.1e} (bitwise {e[3]})")
    t2 = timed({"mlx_lm": ref, "v1": step3(1, 0), "v2": step3(2, 0)}, 16)
    t2s = timed({"mlx_lm": ref, "v1": step3(1, 0), "v2": step3(2, 0)}, 2)
    say("  S 16 chained: " + ", ".join(f"{k} {v:.1f}" for k, v in t2.items()) + "; S 2: "
        + ", ".join(f"{k} {v:.1f}" for k, v in t2s.items()))
    best = min(t2["v1"], t2["v2"])
    say(f"RESULT {'PASS' if best <= 46 else ('KILL' if best > 53 else 'BETWEEN')} (line PASS <= 46 us at S 16, KILL > 53; best {best:.1f};"
        f" mlx_lm here {t2['mlx_lm']:.1f})")
else:
    say("RESULT KILL by Fable B's own gate: the reductions are not the cost (var 1 / 2 not run)")
