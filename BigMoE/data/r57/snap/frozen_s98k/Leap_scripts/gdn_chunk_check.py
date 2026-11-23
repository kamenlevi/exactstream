"""GDN-CHUNK GPU check (Leap, Fri 2026-11-06 08:4x): Fable B's row-parallel gated-delta step (~/FableGod/fableb/kernels/
gdn_chunk.py, a DRAFT, read-only use: its SRC / IN / OUT are loaded, nothing there is edited) against the engine's
kernel (mlx_lm gated_delta_kernel, 'gated_delta_step'), on random inputs shaped as the check's (q / k [S, 16, 128]
l2-normed, q x 1/sqrt(128); v [S, 48, 128]; g float32 in (0.5, 1); beta bf16 in (0, 1); state fp32 [48, 128, 128]).
  bits: y max relative error and the share of bf16 outputs that differ; state max relative error; S 16 / 8 / 4 / 2 / 1
  time: dependent chains over 36 distinct layers (cold states), round-robin, median of 7, at S 16 and S 2
Lines (Fable B's, DENSE_REBUILD v2): PASS <= 40 us chained at S 16, KILL > 50; bits FAST (outputs within ~1 bf16 ulp,
state within 1e-4 relative). The orchestrator 08:4x: time S 2 too. -> ~/Leap/data/gdn_chunk_check.txt"""
import importlib.util, os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

_spec = importlib.util.spec_from_file_location("fb_gdn_chunk", os.path.expanduser("~/FableGod/fableb/kernels/gdn_chunk.py"))
FB = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FB)
OUT = os.path.expanduser("~/Leap/data/gdn_chunk_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nGDN-CHUNK check {time.strftime('%a %H:%M:%S')} ({os.environ.get('GC_COND', 'share lock')}; MLX {mx.__version__})")
KER = mx.fast.metal_kernel(name="fb_gdn_chunk", input_names=FB.IN, output_names=FB.OUT, source=FB.SRC, header=FB.HDR)
NSG = int(os.environ.get("GC_NSG", "16"))
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


V2 = os.environ.get("GC_V2") == "1"                     # Fable B's redraft (gdn_chunk2.py: TAB + MAIN, RB rows per threadgroup)
RB = int(os.environ.get("GC_RB", "8"))
if V2:
    _s2 = importlib.util.spec_from_file_location("fb_gdn_chunk2", os.path.expanduser("~/FableGod/fableb/kernels/gdn_chunk2.py"))
    FB2 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(FB2)
    say(f"  v2: gdn_chunk2.py, RB {RB} ({48 * 128 // RB} threadgroups)")


def chunk(q, k, v, g, b, st):
    S = q.shape[1]
    if V2:
        y, so = FB2.gdn_chunk2(q.reshape(S, 2048), k.reshape(S, 2048), v.reshape(S, 6144), g.reshape(S, 48), b.reshape(S, 48),
                               st.reshape(48, 128, 128), S, RB)
        return y.reshape(1, S, 48, 128), so.reshape(1, 48, 128, 128)
    y, so, _ = KER(inputs=[q.reshape(S, 2048), k.reshape(S, 2048), v.reshape(S, 6144), g.reshape(S, 48), b.reshape(S, 48),
                           st.reshape(48, 128, 128)],
                   template=[("T", q.dtype), ("S_C", S), ("NSG", NSG)], grid=(32 * NSG, 1, 48), threadgroup=(32 * NSG, 1, 1),
                   output_shapes=[(S, 6144), (48, 128, 128), (48, 32, 128)],
                   output_dtypes=[q.dtype, mx.float32, mx.float32])
    return y.reshape(1, S, 48, 128), so.reshape(1, 48, 128, 128)


ok = True
for S in (16, 8, 4, 2, 1):
    a = inputs(S)
    y0, s0 = ref(*a)
    y1, s1 = chunk(*a)
    mx.eval(y0, s0, y1, s1)
    f = lambda x: np.array(x.astype(mx.float32))
    Y0, Y1, S0_, S1_ = f(y0), f(y1), f(s0), f(s1)
    ey = float(np.abs(Y1 - Y0).max() / max(1e-30, np.abs(Y0).max()))
    es = float(np.abs(S1_ - S0_).max() / max(1e-30, np.abs(S0_).max()))
    fl = float((Y1 != Y0).mean() * 100)
    good = ey <= 1e-2 and es <= 1e-4
    ok &= good
    say(f"  S {S:2d}: y max rel err {ey:.1e} ({fl:.2f} % of bf16 outputs differ), state max rel err {es:.1e} "
        f"{'ok' if good else 'BAD'}")


def chain(fn, A, reps=1):
    tt = []
    for _ in range(reps):
        for calls in (False, True):
            prev = mx.zeros((1,), mx.float32)
            for a in A:
                q = a[0] + (prev * 0).astype(a[0].dtype)
                if calls:
                    y, so = fn(q, *a[1:])
                    prev = y.reshape(-1)[:1].astype(mx.float32) + so.reshape(-1)[:1] * 0
                else:
                    prev = q.reshape(-1)[:1].astype(mx.float32)
            t0 = clock(); mx.eval(prev); tt.append((calls, clock() - t0))
    tn = np.median([t for c, t in tt if not c]); tc = np.median([t for c, t in tt if c])
    return 1e6 * (tc - tn) / len(A)


res = {}
for S in (16, 2):
    A = [inputs(S) for _ in range(36)]
    chain(ref, A); chain(chunk, A)
    r0, r1 = [], []
    for r in range(7):
        for which in ((0, 1) if r % 2 == 0 else (1, 0)):
            (r0 if which == 0 else r1).append(chain(ref if which == 0 else chunk, A))
    res[S] = (float(np.median(r0)), float(np.median(r1)))
    say(f"  S {S:2d} chained over 36 layers: gated_delta_step {res[S][0]:6.1f} us, gdn_chunk {res[S][1]:6.1f} us "
        f"({36 * (res[S][0] - res[S][1]) / 1e3:+.2f} ms per check saved)")
    del A
    mx.clear_cache()
t16 = res[16][1]
v = "PASS" if (ok and t16 <= 40.0) else ("KILL" if t16 > 50.0 else "BETWEEN")
say(f"RESULT {v} (Fable B's line: PASS <= 40 us chained at S 16 and bits ok, KILL > 50; S 16 {t16:.1f} us, bits {'ok' if ok else 'BAD'})")
