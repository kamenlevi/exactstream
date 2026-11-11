"""Fit12 R-04 GAP (Mon 26 Oct 2026; PLAN.md 'Rapid screens, batch 2'; port of ~/MoonShots/scripts/gapclock.py): does the GPU
run the 12 GB engine's small expert kernels slower after a short idle gap, as after an SSD wait? Random 4-bit weights at the
engine's shapes: step 'layer' = one MoE layer's routed experts for one row (top-10 of a 16-slot pool, 2560 -> 640 -> 2560,
gs 64, gather_qmm as mlx_lm's SwitchGLU calls it); step 'token' = 48 such layers chained in one graph. Each is timed after a
spin gap (CPU busy) and a sleep gap (CPU idle) of 0 / 0.5 / 2 / 5 / 10 ms, 60 shuffled rounds per gap. F12_GAP_SMOKE=1: CPU,
2 layers, 3 rounds (a shape check, no timing). Usage: [BigMoE venv] fit_gap.py TAG"""
import os, random, sys, time
import mlx.core as mx

TAG = sys.argv[1] if len(sys.argv) > 1 else "gap"
SMOKE = os.environ.get("F12_GAP_SMOKE") == "1"
if SMOKE:
    mx.set_default_device(mx.cpu)
H, I, SLOTS, K, GS, NL, R = 2560, 640, 16, 10, 64, (2 if SMOKE else 48), (3 if SMOKE else 60)
mx.random.seed(0)


def pool(n_out, n_in):
    q = mx.quantize((mx.random.normal((SLOTS, n_out, n_in)) * 0.02).astype(mx.bfloat16), group_size=GS, bits=4)
    mx.eval(q)
    return q


LAY = [(pool(I, H), pool(I, H), pool(H, I)) for _ in range(NL)]
IDX = [mx.array([random.Random(i).sample(range(SLOTS), K)], dtype=mx.uint32).reshape(1, 1, K) for i in range(NL)]
W = mx.full((1, 1, K), 1.0 / K, dtype=mx.bfloat16)
x0 = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
mx.eval(x0, W, *IDX)
gb_layer = sum(a.nbytes for p in LAY[0] for a in p) * K / SLOTS / 1e9


def moe(x, p, ix):
    (gq, gs, gb), (uq, us, ub), (dq, ds, db) = p
    xe = mx.expand_dims(x, (-2, -3))                                   # (1, 1, 1, 1, H)
    g = mx.gather_qmm(xe, gq, gs, gb, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    u = mx.gather_qmm(xe, uq, us, ub, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    y = mx.gather_qmm(g * mx.sigmoid(g) * u, dq, ds, db, rhs_indices=ix, transpose=True, group_size=GS, bits=4)
    return (y.squeeze(-2) * W[..., None]).sum(axis=-2)                 # (1, 1, H)


def step_layer():
    mx.eval(moe(x0, LAY[0], IDX[0]))


def step_token():
    y = x0
    for p, ix in zip(LAY, IDX):
        y = y + moe(y, p, ix)
    mx.eval(y)


def spin(ms):
    t = time.perf_counter() + ms / 1e3
    while time.perf_counter() < t:
        pass


GAPS = [0, 0.5, 2, 5, 10]
print(f"[{TAG}] layer step = {gb_layer * 1e3:.1f} MB of routed 4-bit experts; token step = {NL} layers "
      f"({gb_layer * NL:.2f} GB); {R} rounds per gap{' (SMOKE: CPU, no timing)' if SMOKE else ''}", flush=True)
for name, fn in (("layer", step_layer), ("token", step_token)):
    for _ in range(20 if not SMOKE else 1):
        fn()
    order = [(m, g) for m in ("spin", "sleep") for g in GAPS] * R
    random.Random(1).shuffle(order)
    res = {}
    for m, g in order:
        if g:
            spin(g) if m == "spin" else time.sleep(g / 1e3)
        t = time.perf_counter(); fn(); res.setdefault((m, g), []).append(time.perf_counter() - t)
    base = sorted(res[("spin", 0)] + res[("sleep", 0)])[R]
    print(f"[{TAG}] {name}: back-to-back median {base * 1e3:.3f} ms", flush=True)
    for m in ("spin", "sleep"):
        for g in GAPS[1:]:
            v = sorted(res[(m, g)]); med = v[len(v) // 2]
            print(f"[{TAG}] {name} {m} gap {g:>4} ms: {med * 1e3:7.3f} ms ({med / base:5.3f}x), p10 {v[len(v) // 10] * 1e3:7.3f} "
                  f"p90 {v[(9 * len(v)) // 10] * 1e3:7.3f}", flush=True)
print(f"[{TAG}] done", flush=True)
