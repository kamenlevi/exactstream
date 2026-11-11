# A roofline engine for the non-exact line (NewIdeas 2026-10-21; DECIDED 16:30: build it)

## Why (the binding factor under any quality bar)
Once SSD reads are cut (re-routing, 3-bit cache) and hidden (prefetch), the GPU's own time per token binds:
- eager: ~28.4 ms/token with zero reads vs a byte roofline of ~17 ms (dense 3.42 GB + experts 1.33 GB at 280 GB/s);
- 4-row MTP verify pass: ~82 ms measured (frugal+MTP code 39.2 tok/s at 3.2 tok/pass) vs ~30 ms roofline (~107 tok/s).
So the engine sits 1.7x (eager) to 2.7x (MTP) above its byte floor. Causes (MoonShots FINDINGS 5/7/10, Breakthrough
speed_path): ~40 dependent kernels per layer on the critical chain (3.5-7.6 us each, ~7-15 ms per pass), small matvecs
at M=1-4 reading at ~45-70 % of peak, per-layer gate waits, Python graph construction, draft steps run separately.

## Shape (non-exact allowed: fp32 accumulation, any reduction order; deterministic run to run)
Per layer, at most ~5 dispatches (or one persistent kernel per pass with device-scope barriers, ~2 us each):
1. HC read (stream mix + norm) fused with ALL input projections of the attention block (q/k/v/z/a/b or QSA q/k/v):
   one pass over the layer's projection weights for all rows.
2. Attention core (GDN recurrence or QSA sparse attention) + gated norm + out_proj + HC write + MLP HC read + norm +
   router logits.
3. Routing on the GPU: top-10 (+ near-tie re-route), residency lookup, cold experts pushed to a ring the CPU loader drains
   at QD >= 16; per-slot ready flags (no host sync).
4. Expert GLU for all 10 ranks + shared expert with the gate-weighted combine inside (MoonShots' 3c lineage), spinning
   only on flags of experts that must be on time; HC write.
5. (optional) late-add of half-a-layer-late experts before the next layer's step 2.
MTP draft steps reuse the same fused blocks; the next pass's layer 0 starts as soon as the bonus token is known.
Validation: KLD vs the exact engine on the Eval suite's tf sets must stay at the numerics floor (~0.004); speed vs the
byte roofline per layer (target >= 80 %) and fixed overhead per pass (target <= 1.5 ms).

## Expected (pending the speed-of-light layer)
4-row pass ~82 -> ~35-40 ms (2-2.3x): frugal-class reads ~90-100 tok/s at ~14 GB; formal gate with re-routing + 3-bit
cache ~70-85 tok/s at ~12-14 GB (SSD and compute about equal); in RAM (0.50 setting) ~2x today's ~40 tok/s.
Decision rule: build it if one layer at M=4 runs at <= ~0.5 ms (today 0.89-0.93 ms, byte floor 0.47 ms).
Owners if built: MoonShots (kernels), Breakthrough (frugal/re-route integration and measurements); the main track keeps
BigMoE's exact engine (exactness is not this engine's contract).

## Decision (16:30, on MoonShots' speed-of-light layer, ~/MoonShots/patches/sol_layer/README.md, data/day14/sol_layer_*.txt)
Measured / composed per GDN layer: M=1 0.426 ms (216 GB/s; floor 0.306-0.34), M=4 ~0.61 ms at real expert overlap
(~0.70 at 37 distinct experts; floor 0.44-0.48) vs 0.89-0.93 today: 1.45-1.5x. Megakernels with grid barriers lose on
M5 (barrier 2.0-2.5 us vs 1.3-1.6 us per dispatch boundary); Python graph build is hidden by mx.compile. Remaining gap:
GEMVs at 250-270 of 300 GB/s, ~12 dependent hops x ~4.6 us, ~70 us of small latency-bound kernels. Per pass in RAM: a
4-row verify + 3 MTP drafts ~34-38 ms -> ~80-90 tok/s at 3.1 tokens/pass, if no per-layer CPU gates; small RAM M=1 ~21 ms
of compute per token vs ~30 today.
My rule asked for <= 0.5 ms; 0.61 misses it, but I approve the build: the engine is the binding factor under every
quality bar, no other lever offers 1.45-1.5x, and dropping the per-layer CPU gates is worth more in small RAM than the
kernels alone. Shape changes from the draft: no persistent megakernel; bandwidth-shaped GEMVs (MLX qmv-style row
mapping), compiled graphs, GPU-side gating and a loader ring. First milestone: the 4-row verify + MTP drafts for the
frugal / dh_w4 policies at 10 % residency with GPU-side gating; validation KLD vs the exact engine at the numerics floor.

Confirmed end to end 16:11 (MoonShots v3, data/day14/sol_layer_v3_mlx.txt): per layer GPU M=1 0.410 ms (224 GB/s), M=2 0.512,
M=4 0.720 at 38 distinct experts (~0.63 at real overlap), M=8 1.307; mx.compile wall = GPU time within 2-3 %.

## REVERSED 10-22 04:36 (MoonShots M1b, true all-hit passes, data/day15/m1b_fixed_*.txt): the BIG BUILD is STOPPED
The same 4 positions re-run from a restored state (identical routing, 0 of 36,864 lookups missing): BigMoE's own model 38.2 ms
per 4-row verify pass (p10 38.0, p90 38.4), the engine + BigMoE's gated MoE 38.5 ms, the engine's gate-free prototype 39.1 ms.
So BigMoE under the reference flags already runs near the speed-of-light layer at M=4 (~0.80 ms per layer incl. the MoE vs
SoL 0.65-0.72); the "0.89-0.93 ms today" baseline behind the 16:30 decision included stall effects, and the per-layer gate
costs ~nothing. The engine's compute prize is ~10-20 % per pass, not 1.45-1.5x. The gap from ~100 ms (24 GB streamed) to ~38 ms
is cold-read stalls. Decision (NewIdeas): stop the general engine; keep its code as a tested kernel library; park M3 wide
passes (no drafter fills them), frugal phase-B multi-row and in-RAM mode; MoonShots moves to the STALL ANATOMY (24 GB exact
and B12 at 10 %: compute vs waited reads vs overlap per pass, SSD bytes and GB/s, the floor max(compute, bytes/bandwidth)
and the cheapest lever: bytes per miss, overlap, residency, fewer misses). Lesson: a speed-of-light comparison must use a
baseline measured without stalls (true all-hit passes), or it credits the engine with the SSD's time.
