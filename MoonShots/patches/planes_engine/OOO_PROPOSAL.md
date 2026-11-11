# Out-of-order experts in BigMoE's engine: integration proposal (MoonShots, 2026-10-21 ~00:45)

Kernels (bitwise, tested): `tierglu.py` `tier_glu_phase(x, msb, lsb, msb_slot, lsb_slot, phase, which, plan)` and
`combine_phase(y1, y2, phase, scores)`; `test_split.py` = fused 3c+combine for any phase mask, M = 1-8, both plans.

## What the ext already has
`bigmoe_post_ids` (ext/bigmoe_ext.cpp ~395) reads the residency table published a pass ahead and writes
`dep = [ok, slot_0 .. slot_{n-1}]`: `sl = tab[...]` for every routed id, `ok` = all found with the right tag. The spin
kernel copies `dep` on `ok` (fast path), else polls the loader's release lines.

## Changes (small)
1. **post_ids, one line:** `dep[1 + t] = (tag == m.y) ? sl : -1;` (today `dep[1 + t] = sl;`, which may hold a stale-tag slot when
   ok == 0). Then dep[1..n] is the partial-hit row: a slot for experts resident and published for this pass (the table
   only grows during a pass and entries never move, so the slot is safe to read before the spin), -1 otherwise.
2. **gated.py: split route_and_gate** into `post(L, seq, ids) -> dep` and `spin(L, seq, dep) -> slots`, so the caller can
   enqueue the hit work between them (MLX keeps submission order; the hit kernels don't touch the spin's buffers, so
   MLX's concurrent encoder lets them run while the 1-thread spin waits).
3. **tier.py tier_call (plan mode):**
   ```python
   dep_m = post(L, seq, ids)                      # base pool row;  dep_l = post(side, seq, ids[..., :R4]) for the sidecar
   pre_m = dep_m[1:1 + n].reshape(M, R); pre_l = dep_l[1:1 + n4].reshape(M, R4)
   ok_l = mx.concatenate([pre_l >= 0, mx.ones((M, R - R4), mx.bool_)], axis=-1)   # 3-bit ranks need no sidecar
   phase = mx.where((pre_m >= 0) & ok_l, 1, 2).astype(mx.int32)
   y1 = tier_glu_phase(x2d, base, sideP, pre_m, mx.maximum(pre_l, 0), phase, 1, plan)   # runs during the spin
   sm = spin(L, seq, dep_m); sl = spin(side, seq, dep_l)
   y2 = tier_glu_phase(x2d, base, sideP, sm, sl, phase, 2, plan)
   y = combine_phase(y1, y2, phase, scores2d)     # == (y * scores).sum(-2) bitwise; the block adds shared after
   ```
   All lazy: no host sync. A pass where every expert hits takes the fast path anyway (phase 2 has no pairs: its
   threadgroups exit at once; ~3 near-empty dispatches, cost to be measured).
4. Exactness: per-pair arithmetic is the fused kernel's; the combine is MLX's LY=8 order -> outputs identical to 3c+comb
   (and so to 3b + MLX's combine) for any hit pattern. v3 shas unchanged by construction.

## Expected value (data/ooo_bound125.txt, pass_profile125 at 0.50)
Loader stalls 31-34 ms/pass (per-layer p50 0.57-0.65 ms); the resident pairs' routed compute (~0.3 ms/layer with 3c)
can hide ~9.4-10.5 ms/pass -> ~+7-10 % net of the extra dispatches (to be measured: bench_split.py in day_runner14).
More at lower memory (more stalls per layer), bounded by the hit compute per layer.
