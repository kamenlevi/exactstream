# Persistent level kernel: design note (MoonShots, draft line; Tue 17 Nov 08:0x; desk, before NewIdeas' go)
Goal: f per level from fused2's 0.269 ms toward the measured no-dependency bound (~0.09-0.11), on P2's C0 level
(177.6 MB; the planned head's format), F = 1 first.

## What f is made of (measured, BET 7)
- fused2 = 17 dependent kernels per level. Its steps timed alone sum to 0.58 ms, but chained the level takes 0.88:
  ~0.29 ms per level is lost between dependent kernels (~16 us per boundary), i.e. a drain (the last threadgroups of a
  step) plus a ramp (the next step's first weight loads go to DRAM cold).
- Dispatch count is not the cost: <= 1.3 us per dependent dispatch [K M7a]; a grid barrier is ~2-2.5 us [K M5].
- With the dependencies removed (each step fed the reference inputs) a level takes 0.699 ms: the overlap bound.
- The system cache does not hold a step's weights ahead of its consumer (touch prefetch KILL), so overlap must come
  from loads issued in REGISTERS before the barrier, not from cache warming.

## The kernel
- One dispatch per level (v1), later per block. Grid = the GPU's resident capacity, e.g. 20 cores x 2 x 512 threads
  [to be sized by a probe: occupancy of the heaviest phase]; every threadgroup loops over the level's phases.
- **Phases** = fused2's steps with a dependency between them, ~15: S1G, HC attn read (2), S4 qkv, S5 attention
  (blocks + combine), S6 o_proj, HC mlp read (2), S9 router, S9b top-10, S10 experts gate|up, S11 down + combine,
  HC mixer (2), S14 screened head, S15P tail (top-K, next embedding, next input norms).
- **Work mapping:** each phase keeps fused2's per-row code (the bit-matched GEMV / reduction bodies), run by "virtual
  threadgroups": a phase whose fused2 kernel used (32, 2)-thread threadgroups runs 8 virtual groups per 512-thread
  threadgroup, looping over its row blocks. A row's arithmetic is unchanged, so its bits are too (the property every
  fused2 step was built on).
- **Barrier:** a generation counter in device memory (atomic add + spin on the generation, with a device-scope fence).
  Timeout -> a status flag, and the host re-runs that level with fused2's discrete chain (no hang).
- **Overlap (the point):** before arriving at a barrier, a threadgroup issues the loads of its FIRST weight tile of the
  next phase into registers. Its assignment in a dense phase is static (row blocks by threadgroup id), so it is known
  in advance. Expert phases know their weights only after S9b: prefetch the router's output-independent parts there
  (the shared expert, the next HC weights).
- **Latency-bound phases** (norms, top-10, the tail, ~0.07 ms per level in BET 7's profile) stay; one threadgroup does
  them while the others prefetch the next phase.

## Plan / lines
- Wed: build from fused2's sources, with bit tests per phase against fused2's outputs (share lock).
- Thu: quiet bench: PASS f <= 0.15 (<= 0.76 ms per level at 400 keys), KILL f >= 0.25 (fused2's 0.269).
- Fri: block form + the M3 kept read-out.

## Risks
- Co-residency: the grid must not exceed resident capacity, or a spin barrier deadlocks (timeout fallback above).
- Register pressure from the cross-phase prefetch: it costs occupancy. Last night's module 2 showed that in a
  compute-bound kernel, extra registers lose more than they save. The level's GEMVs are bandwidth-bound, not
  compute-bound, so this should matter less here. To check in Thursday's bench.
