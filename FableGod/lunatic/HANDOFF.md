# HANDOFF: incremental or published ideas noticed by the Lunatic (for NewIdeas' queue; one line each, gains unhalved unless marked)

Round 1 (Tue 3 Nov 2026, 00:5x EET):
- **Chat relaxed acceptance (FAST)**: tie/mid rows are 57 % of IFEval and 51 % of multilingual rows (DAY9_q1); a top-3 / p >= 0.1 rule takes code mid rows 66 -> 82 % (lu_alpha), probably more on chat. Chat kept x1.5-2. Published family (MARS, typical acceptance). Big but not new: check that RELAX covers chat, not only code.
- **Chat's wall is the trunk**: at kept 2.4 the 3.44 GB trunk is ~60 % of chat's RAM bytes per kept token. Trunk q4 (the hc_q4 / att_q4 / gdn_q4 maps exist) is chat's largest byte lever: ~x1.5 on chat's roofline (FAST).
- **The horizon fix alone on EXACT**: perfect drafts at 125B-sure rows lift agreement 0.909 -> 0.9485, x1.36 bytes per kept at 32 rows (lu_alpha). This is EXACT's biggest single lever; MTP Head's target is block positions 8-15 at sure rows (S2).
- **Style-contract system prompt** (the engine defines a fixed "conventional style" prompt, so EXACT stays bitwise to the engine's own output): measure kept and scores with / without it. Expected <= +15 % kept on code.
- **Monitor-guided drafting for code**: a CPU parser / scope checker vetoes drafter tokens that cannot parse or name nothing in scope (Agrawal et al. 2023, MGD). Sure-row misses -10-30 %, kept +2-5 %.
- **Order-invariant pre-rounded accumulation** (ReproBLAS-style: integer codes x 11-bit fixed-point activations stay below 2^24, so fp32 sums are exact in any order). EXACT could reuse FAST-K kernels: H-1i +8 %, key-split attention -10 ms per check at 10K. Per-group scales limit it to within-group sums; overlaps Fable B's F1 / REF3-T and cloud task 17's integer-exact option.
- **Night domain distillation of DFlash2** on the day's 125B outputs (online speculative decoding, 2023): kept +5-15 % on the user's own domain.
- **Lossless entropy-coded RAM tier** decoded on the GPU (rANS, DietGPU-style): codes at 2.72 bits give +9 % slots and -9 % RAM and SSD bytes.
- **Certified LM-head argmax** via norm bounds (MIPS clustering of the 248K vocab rows): head 1.53 -> ~0.5 ms per 16-row check, EXACT.
- **Bit-plane residency** (the MSB planes of every expert resident, the LSB plane only for hot ones; generalizes the 1-bit sidecar): +-10-20 % SSD bytes; a replay question for MoonShots.
- **SSD queue depth** to reach the 14.6 GB/s peak from 11.7 GB/s bursts: up to +20 % on SSD-bound lines (check against M-20).
- **Copy-heavy agentic edits as a benchmark**: copy spans run at agreement ~0.995, so 128-row checks give x2.6 bytes per kept vs 16 rows (lu_union C), ~600 tok/s EXACT on the copied part. HumanEval cannot see it (COPYGATE x1.04). Speculative edits are published; add an edit workload to the measured mix.
- **Shadow stream** (decode the agent's predictable next step in a free batch slot): x0.81 RAM and ~0 SSD per its token (lu_union B2). Worth it only where the next prompt is exactly predictable.
- **Cross-expert correlation check** (one minute of CPU): correlate two experts' codes within a layer. Delta coding needs rho >= ~0.87 for 2x; expected near 0 (MoBE parked).

Round 1, think-first additions (00:5x):
- **Re-base EXACT on the all-3-bit plan if it scores the same** (Kamen's call): EXACT SSD per kept on code 46.3 -> 24.5 MB at C 256 (M-26), about 125 -> 160 tok/s at 45 GB (x1.3). The strict KLD gate failed it (M-02), so it hinges on task scores (t3_all, EvalSuite).
- **Prefill is the agentic wall at 350**: a 3K-token tool output costs ~2-3 s at 45 GB (about half the experts stream from the SSD per chunk), against 0.86 s for 300 decoded tokens. Levers:
  - the persistent prefix-state cache (PARKED, TRIED:780)
  - bigger prefill chunks
  - int8 matrix units for compute-bound prefill (1.6x there, useless for decode)
  - one agent's prefill under another agent's decode (S1)
- **CPU attention and Neural Engine dense layers as co-processors** inside S1's multi-stream pipeline: they fill the GPU's latency gaps across streams. <= 30 %.
- **MTP head for the first 1-3 block positions, DFlash2 for depth**: the MTP head gets close calls 86 % right at depth 1 (KWALL b) vs DFlash2's 66 %. A hybrid draft aimed at both halves of S2 (MTP Head).
- **Footprint audit**: today's 8.5 GB non-pool lifetime peak (BigMoE HANDOVER:607) is ~17 % of the night budget; every GB returned is ~8 more cache units per layer.

Round 2 (Tue 3 Nov 2026, 17:3x; from LUNACY2.md, gains unhalved):
- **Int8 (or ANE) trunk for wide passes:** at 256-384 rows per pass the trunk's compute is 40-60 % of the pass (0.32 ms per row in bf16 at 60 %); int8 gives +18-22 % at 8-12 merged streams (lu_1000 D). Matters only once checks merge streams.
- **Batched drafting throughput:** the 1000 line needs >= ~2,000 draft tokens/s from the ANE for 8-12 streams, fully overlapped; today it is 16 tokens per 12 ms serial.
- **Schedule merged streams by domain:** mixed domains cost +20 % union at 8 streams (283.4 vs 236.9 per layer, lu_union B). Group same-domain streams per pass.
- **Speculative edits for file writes (published):** 37.3 % of the lines Kamen's agents write into files were already seen in the session (Write 35.3 %, Edit 49.0 %; lu_copyshare), x1.35 on file writes. It is only 6.2 % of all agent output, and copy tokens carry no new information.
- **Agent prefill is compute-bound:** 5.05 G MAC per prompt token, so <= 1,663 tok/s in bf16 at 60 % or 2,661 in int8. At 1000 tok/s decode a 3K-token tool output is ~80 % of a turn. The exact prefix-state cache (TRIED:780) is the only exact lever; position-independent reuse (CacheBlend) is FAST.
- **Exact references diverge on most chat prompts:** REF2-T vs REF2-CT differ on 200 of 220 IFEval prompts after a mean 39 shared tokens, and a single flip rewrites the rest (only 28 % re-draftable; lu_resync). Any EXACT re-base changes nearly every chat answer. EvalSuite must compare scores, not texts.
