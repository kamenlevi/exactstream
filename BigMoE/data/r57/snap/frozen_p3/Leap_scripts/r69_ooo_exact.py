"""R-69 OOO-EXACT (Leap, Sun 2026-11-01; tensor-level, shared GPU): are BigMoE's TIER_COMBINE (3c with the block's combine
fused in dispatch B) and OOO (tier_glu_phase phase 1 / phase 2 + combine_phase) bit-identical to what REF2-T runs today
(tier_glu without scores, then MLX's (y * scores[..., None]).sum(-2) in the verify call), at S 1-16 rows?
Synthetic pools at the 125B's expert shapes (as K-61/K-62), plan 4433333333, routing from 100 slots, sidecar per expert,
bf16 scores = MLX's top-10 of a precise softmax, renormalized (the verify call's order). OOO phases: a random ~30 % of
pairs in phase 2. Frozen tierglu copy (/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn)."""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
import tierglu as TG
H, I, CM, CL, NR, PLAN = 2560, 640, 128, 64, 10, "4433333333"
mx.random.seed(0)
def ru32(*shape): return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01): return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k): return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64, s=0.03)}
def lsb_part(o, k): return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64, s=0.03)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
rng = np.random.default_rng(1)
side_of = rng.permutation(CM) % CL
eq = lambda a, b: bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
rows = []
for S in (1, 2, 4, 7, 10, 16):
    ok_c, ok_o = True, True
    for trial in range(3):
        ms_np = np.stack([rng.choice(100, NR, replace=False) for _ in range(S)]).astype(np.int32)
        ls_np = side_of[ms_np[:, :2]].astype(np.int32)
        ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
        x = (mx.random.normal((S, H)) * 0.5).astype(mx.bfloat16)
        g = mx.softmax(mx.random.normal((S, 512)).astype(mx.bfloat16), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-NR, axis=-1)[..., -NR:]
        sc = mx.take_along_axis(g, inds, axis=-1)
        sc = sc / sc.sum(axis=-1, keepdims=True)
        y = TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN)                       # REF2-T today: [S, K, D]
        ref = (y[None] * sc[None, ..., None]).sum(axis=-2)[0]                  # the verify call's combine
        fused = TG.tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN, scores=sc)        # TIER_COMBINE
        ph = mx.array((rng.random((S, NR)) < 0.3).astype(np.int32) + 1)         # 1 = resident, 2 = loaded later
        y1 = TG.tier_glu_phase(x, msb, lsb, ms_, ls_, ph, 1, plan=PLAN)
        y2 = TG.tier_glu_phase(x, msb, lsb, ms_, ls_, ph, 2, plan=PLAN)
        ooo = TG.combine_phase(y1, y2, ph, sc)
        mx.eval(ref, fused, ooo)
        ok_c &= eq(ref, fused.reshape(ref.shape)); ok_o &= eq(ref, ooo.reshape(ref.shape))
    rows.append((S, ok_c, ok_o))
print("S, TIER_COMBINE == REF2-T today, OOO (phases + combine_phase) == REF2-T today (3 trials each, bitwise):")
for r in rows: print("  ", r)
open(os.path.expanduser("~/Leap/data/r69_ooo_exact.txt"), "w").write("\n".join(str(r) for r in rows) + "\n")
