"""1c-tiers glue count (Leap, Mon 16 Nov 2026; dispatch_diet_price.md section 5): the non-view MLX ops one MoE layer of
a 16-row check adds for 1c: BigMoE r57_screen_fast _t1c (tiers_mx + the counter) + the km path, and ref2's 2-D dup pick
(b43efc4). Graph export on the CPU (mx.export_to_dot), tiny shapes, no model."""
import os, re, sys
import mlx.core as mx
mx.set_default_device(mx.cpu)
sys.path.insert(0, os.path.expanduser("~/BigMoE"))
import bigmoe.tiers1c as T1C
M, K, R4 = 16, 10, 2
ind = mx.random.randint(0, 512, (M, K)).astype(mx.uint32)
sc = mx.softmax(mx.random.normal((M, K)), axis=-1).astype(mx.bfloat16)
mx.eval(ind, sc)

def count(outs, tag):
    p = f"g_{tag}.dot"
    mx.export_to_dot(p, *outs)
    s = open(p).read()
    ops = re.findall(r'label\s*=\s*"([A-Za-z]+)', s)
    prim = [o for o in ops if o not in ("array",)]
    views = {"Reshape", "Slice", "Broadcast", "Squeeze", "ExpandDims", "Transpose", "AsStrided", "Flatten", "Unflatten"}
    nv = [o for o in prim if o not in views]
    print(f"{tag}: {len(prim)} primitives, {len(nv)} not views: {sorted(set(nv))}")
    return len(nv)

# BigMoE _t1c (r57_screen_fast.py) + the km path (_fast_call)
ids2 = ind.reshape(-1, K)
tier = T1C.tiers_mx(ids2, sc.reshape(-1, K), s=0.98, lam=0.016)
texp = mx.full((512,), 3, mx.int32).at[ids2.reshape(-1)].minimum(tier.reshape(-1).astype(mx.int32))
cnt = mx.stack([(texp == 0).sum(), (texp == 1).sum(), (texp == 2).sum()])
nl = tier[:, :2] != 0
km = (tier != 2).reshape(ind.shape)
scores = mx.where(km, sc, mx.zeros_like(sc))
indices = mx.where(km, ind, ind[..., :1])
a = count([nl, km, scores, indices, cnt], "bigmoe_t1c_km")
b = count([cnt], "counter_only")
# Leap's 2-D pick (ref2 b43efc4)
ar = mx.arange(M * R4, dtype=mx.int32); mx.eval(ar)
idr = indices.reshape(M, K)[:, :R4]
j_ = mx.min(mx.where((~nl).reshape(-1), ar, M * R4)) % (M * R4)
ids_s = mx.where(nl, mx.take(idr.reshape(-1), j_), idr)
mx.eval(nl, indices)
c = count([ids_s], "leap_2d_pick")
print(f"per MoE layer ~{a + c} small ops for 1c (counter {b}); x 48 layers = {48 * (a + c)} per check")
