# CLOUD TASK 6: the dense pass. From ~32 ms to ~18 ms per 16-row check, exactly.

A closed thinking task. Don't use the web; think and calculate. Everything you need is below.

**Why.** Your last answer said 350 tok/s needs ~20 kept per check, a dense pass of ~20 ms, and the SSD hidden: all exact engine and drafter work. A local session is designing the drafter. You take the dense pass: everything a check computes besides the routed experts, which is read in full every check.

**The model** (Qwen3.8-Flash-Next 125B MoE)
- 48 layers, hidden size 2,560, with hyper-connections: 4 residual streams and low-rank mixers (rank 320).
- 36 Gated-DeltaNet layers: 16 key heads and 48 value heads of dim 128, conv kernel 4.
- 12 full-attention layers, one every 4th: 24 query heads, 2 KV heads, head dim 256, partial RoPE (0.25).
- MoE: 512 experts, top-10, intermediate 640, plus a shared expert of 640.
- Vocab 248,320 (untied head).
- Weights: 4-bit groups (the dense part); experts 3-4 bit tiers.

**The machine:** M5 Pro, 64 GB unified memory, ~258-284 GB/s measured streaming bandwidth, Metal / MLX, GPU cores with neural accelerators used for 16-row matmuls.

**Measured: one warm 16-row check, by piece** (GPU timeline, ms per check, misses 0):
- hyper-connection reads: 5.6 (attention side) + 5.9 (MLP side) = 11.4
- Gated-DeltaNet: 12.4
- attention: 7.3-7.5
- MoE glue: pre 2.5-2.8, post 2.2-2.5
- top level (embedding, final norm, head): 2.4-2.6
- routed experts: ~3.0 ms per row (~53 at 16 rows; not your topic)

The check's warm fixed part is 36-38 ms plus 2.9-3.2 ms per row. The dense pieces barely change with rows (each <= 0.1 ms per added row).

**Dense bytes per check:** 3.44 GB (hyper-connections 1.28, GDN 1.18, LM head 0.36, attention 0.36). At ~270 GB/s that is a ~12.7 ms floor (HC ~4.7, GDN ~4.4, head ~1.3, attention ~1.3). So today GDN runs at ~2.8x its byte floor, HC ~2.4x, attention ~5.6x.

**Tried** (the local kernel session's results)
- an exact fused HC read (MLX GEMMs + custom norm / gate kernels): bitwise equal, 24.4 -> 14.7 ms per check in its harness (BETWEEN; partly in today's numbers)
- a batched HC mix-down (bmm16): KILL (tokens differed or < 1.5 ms)
- 16-row shape tuning (SPK16): KILL (< 2 ms)
- K-73: ~1 ms, adopted
- a cheaper attention path (REF2c-T): failed quality (not exact)
- 8-bit HC weights: passed quality with a lean, ~-0.6 GB (~-2 ms), but changes tokens, so it is not part of this task
- a one-kernel GDN body: estimated to take the constant toward ~13 ms, 1-2 weeks of work, not built

The best reachable with known exact levers was judged ~15-18 ms.

**Exactness.** The exact mode is defined by its arithmetic: the reference (REF2-T) must reproduce bit-for-bit, checked at full length over ~15,000 rows. A kernel that changes reduction order changes bits. So every design must either keep the reference's arithmetic order, or propose a new reference and say how it would be re-certified (a full-length bit-exact test against its own autoregressive run, plus same benchmark scores).

**Deliver, in this order**
1. **The gap per piece** (<= 200 words): floor, today, and the likely cause of each gap: launches, syncs, reads of activations, small-matrix efficiency, state handling in GDN, recurrent dependencies.
2. **At least five concrete designs** (<= 600 words). Examples: a fused per-layer GDN kernel, HC read and mix fused into the neighbouring matmuls, persistent / megakernel scheduling across layers, overlapping layer L+1's weight stream with layer L's compute, attention at 16 rows with GQA-shared K/V loads, moving glue onto the GPU without syncs. For each give:
   - ms saved per check (an upper bound, then halved)
   - effort in days
   - the exactness plan
   - the risk
   - the cheapest test that proves it, with a kill line
3. **The plan to ~18-20 ms** (<= 250 words): the order of work, the cumulative ms, and what bounds it below that (bytes, launches, the GDN recurrence).
4. **The first kernel to build tomorrow** (<= 100 words).

**Rules**
- This Mac only, the full 125B. Mark every number as given / arithmetic / judgement.
- Treat your own savings as upper bounds; real runs came out ~2x below estimates today.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within your first ~20 minutes, write a complete first draft of all four sections to ANSWER.md. Update it once or twice after that.
- If you're running long, cut depth, not sections: every section gets at least 3 lines.
- When done, print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository, and don't install or download anything.
