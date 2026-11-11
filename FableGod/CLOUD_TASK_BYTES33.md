# CLOUD TASK 18: the 33 MB problem. Halve the SSD bytes per kept token.

A closed thinking task. Don't use the web; think and calculate. Everything you need is below, measured unless marked.

**Why.** Your ladder (task 15) put the SSD at ~2 ms per check at target. A local audit refuted that on Monday night:
- On code the SSD is bandwidth-bound, not latency-bound. One HumanEval check moves 1,452 MB in 124 ms of demand reads, 11.7 GB/s.
- Today's default (C 256) reads 62.4 MB per kept token. That alone caps exact at ~187 tok/s on code, even if everything else were free.
- C 320 (+ per-layer pool sizes) still needs 37.9 MB per kept token: ~40 ms per check at kept 12.6, your whole target cycle.
- **350 on code needs <= ~33 MB per kept token** at ~13 GB/s, with zero margin. Hiding can't fix bytes.

**The facts**
- **Model:** 48 layers, 512 experts top-10 per layer (intermediate 640), hidden 2,560.
- **Expert records:** 3-bit tiers ~2.15 MB per expert, plus a 0.77 MB 1-bit sidecar for ranks 0-1. Dense trunk 3.44 GB, always resident.
- **RAM:** 64 GB total. C = cache slots per layer (of 512).
  - C 256 is today's 41.7 GB footprint.
  - C 320 needs ~50 GB; C 350 needs ~55 GB, about the most this Mac can hold beside macOS.
- **SSD:** 11.3-13 GB/s sequential reads. +50-65 us wake penalty per burst after 0.25-1 ms idle; 4 KB keepalive reads remove it.
- **Locality on real checks:**
  - Within a 16-row check, rows per expert group: 2.33 (code), 2.58 (ifeval), 3.05 (multilingual). At 32 / 48 / 64 rows (code): 3.12 / 3.75 / 4.29.
  - 51 % of the next check's experts are in the last check's set, 74-77 % in the last four checks' union.
  - A repeated bigram shares only 26 % of its top-10.
  - LRU with a popularity prior is within 1.6-2x of Belady. Belady's 56 % headroom needs 5-20 checks of foresight; a perfect next-check predictor buys only ~15 %.
  - The look-ahead predictor (L+1's router on L's input) recalls 0.665 of L+1's union at precision 0.68.
- **MB per kept token by cache size** (code, replay of 171 real chained checks, per-layer pools):
  - C 256: 68.6, C 278: 56.4, C 300: 46.1, C 320: 37.9, C 340: 32.0, C 360: 27.4
  - plain LRU at C 320: 49.0; Belady MIN at C 320: 27.2
  - RAM: 38.0 GB at C 224 + 0.13 GB per C unit
  - The 1-bit sidecar pool is fixed at 112 slots per layer and misses ~15.8 per kept token at every C (12 MB of the 37.9).
- **Byte levers measured so far** (code, per kept token, all on the union per block):
  - Per-layer pool sizes: -15.7 %.
  - C 224 -> 256: misses -31 %.
  - Q1 (top-2 experts on drafter-sure rows, FAST, 0.09 % of gated rows flip): SSD x0.60-0.68 at C 256.
  - Lossless entropy coding: the 3-bit codes carry 2.72 bits of entropy, the whole record ~2.98 of 3.5 bits per weight, at most -15 %. Never built; decode cost unmeasured on this Mac.
  - The reuse of the previous token's MoE output: only x0.91 (the union kills it).
  - The sidecar at 1.5x slots: -10..-25 % reads.
- **Compute that's idle:** the 18-core CPU (~1 TFLOPS) is mostly idle during checks. The Neural Engine runs a drafter at 12.4 ms per batched call.
- **Kept per 16-row check:** 10.5 code / 2.4 chat (exact); FAST tops ~13 on code. Bytes are per check, so kept divides them.
- **Quality contract:** EXACT = bitwise to the engine's own greedy output. FAST = same scores on math / code / instruction / multilingual tests.

**Deliver, in this order**
1. **The lower bound** (<= 250 words): the fewest SSD MB per kept token at C 256 / 320 / 350, EXACT and FAST, under ideal caching with the locality above. What it would take to approach it.
2. **The mechanisms** (<= 450 words): what beats today's LRU + prior + look-ahead toward that bound. Consider:
   - foresight from the drafter's own tokens, which are known before the check
   - fewer bytes per expert, exact and FAST
   - more kept per check
   - using the idle CPU or Neural Engine to decode or predict
   - anything new
   Rank them by MB per kept token saved, halved, with the RAM each needs.
3. **The path** (<= 200 words): the cheapest combination that reaches <= 33 MB on FAST, and EXACT's best, at C 256 / 320 / 350.
4. **One radical option** (<= 150 words) that changes the problem, without a smaller model or pruning (all 512 experts stay), plus its odds.
5. **The three biggest unknowns** (<= 150 words), each with a 10-minute test and a pre-registered line.

**Rules**
- The full 125B (all 512 experts), this MacBook only, no external hardware.
- Mark every number given / arithmetic / judgement. Halve your own gains.
- Price bytes on the union per check (an expert read serves every row that uses it), never per row.

**Budget, so the answer always arrives**
- No web. At most 8 short Python runs (each under a minute).
- Within ~20 minutes, write a complete first draft of all five sections to ANSWER.md. Update it once or twice after that.
- Cut depth, not sections.
- Print ANSWER.md as your final message, in <= 1,300 words.
- Don't ask questions or wait for approval. Don't change or commit anything in any repository, and don't install or download anything.
