# MISSION 26: THE HC READ. Break the one engine wall nobody has an idea for.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~20:0x.)

**Why.** Cloud Fable's exact-350 answer (~/FableGod/CLOUD_ANSWER_EXACT350.md) leaves four engine pieces that must all land for 350, even with the gate:
- the dense pass <= 17 ms per 16-row check (today ~36; realistic with known levers 22-27)
- the expert kernel within 15 % of its byte floor
- the SSD hidden
- the draft hidden

Three have owners and tests. The first has a hole in the middle that nobody has filled: the hyper-connection (HC) reads take 11.4 ms per check against a floor of ~2.3 ms (Leap's estimate; ~4.7 ms if all 1.28 GB of HC weights must be streamed). That gap is the single largest dense pool, and closing it is the difference between ~25 and ~17 ms. Leap has tried the obvious fusions. This needs an idea.

**What is known** (read the sources; don't trust this summary over them)
- **The model:** 48 layers, hidden 2,560, 4 hyper-connection streams with low-rank mixers (hc_count 4, hc_lowrank 320; config at ~/BigMoE/models/bare/config.json). Each layer reads the streams into the mixer input (attention side, hc_a ~5.6 ms per check) and again for the MLP side (hc_m ~5.9 ms), then writes back (hc_w).
- **Leap's work** (~/Leap/TABLE_350.md: R-09 HCF from line ~1664, R-09b, R-82 HC-GATE, the CA / C0 breakdowns; scripts r09b_screen.py, r82_hc_gate.py, r73_glue_anat.py, c0_timeline.py):
  - the exact fused X5 read is bitwise equal to the eager chain: 24.4 -> 14.7 ms per check in its harness
  - MTPLX's 2x8 read is ~-13.9 ms but not bitwise
  - eager at 16 rows used ~11 kernels per read, with bf16 mixer matmuls ~12.7 ms of the reads + writes
  - 8-bit HC weights pass quality with a lean (-0.6 GB per check), but change tokens
- **Exactness:** REF2-T is defined by its arithmetic. A new kernel either keeps the reference's expression order (bitwise), or is row-invariant (1-row == 16-row bits) and joins a new reference certified once (cloud's REF3-T proposal: full-length SD == AR bit-exact plus same scores; Kamen's decision later).

**Deliver**
1. **What the HC read really computes:** write the per-layer math from the code (the mixer weights' shapes, which parts are static and which depend on the input, the norms, the low-rank path, the reads and writes of the 4 streams), and count bytes and FLOPs per 16-row check, piece by piece.
2. **Why it costs 11.4 ms.** Account for the gap to the floor: weight bytes, activation traffic (the 4 x 2,560 streams x 16 rows, read how many times?), small-matrix efficiency, the number of passes, launches. Reconcile with Leap's measurements.
3. **At least three new ways to compute it near the floor.** Examples:
   - algebraic restructuring: fold the static mixers into neighbouring weights, precompute the low-rank products, merge the attention-side and MLP-side reads
   - reading each stream once per layer
   - a layout change of the streams
   - doing the mixing in the matmul epilogues of the previous layer

   For each give:
   - expected ms per check (an upper bound, then halved)
   - exact or row-invariant (and why)
   - the risk
   - what Leap would build
   - the cheapest check of the idea: a CPU numeric check of the algebra in fp32 / bf16 where possible, or a 10-minute GPU screen for Leap
4. **The bet:** which design to build first, and what it does to the dense pass and to 350.

**Rules**
- CPU only (numeric checks of the algebra with numpy / MLX on the CPU are fine); writes only in ~/FableGod; no downloads. GPU timing is Leap's.
- Check ~/TRIED/TRIED.md and Leap's TABLE_350 for every HC idea already tried (R-09 / R-09b / R-82 / A2 bmm16 / HC8 / the 2-bit HC kill / HC static) and say why yours differs.
- Mark every number as measured / arithmetic / judgement.
- This is a real wall: if the floor turns out unreachable for a structural reason, prove which one.
