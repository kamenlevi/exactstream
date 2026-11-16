# IDEAS_1116b: FIRST PRINCIPLES (fable-zero, Mon 16 Nov 17:4x-19:00, desk only)

Target (PUSH800): 775-800 tok/s lossy, measured, at quality, by Saturday. My lens: the per-kept-token cost priced from
zero on this Mac (18 CPU / 20 GPU cores, ~280-300 GB/s, 64 GB), then what a Saturday-sized engine keeps or cuts, then
whether any ONE-stream decode shape beats the frame. Sizes are the real ones (Leap 13:5x: 2.784 MB per 4-bit expert,
2.064 MB per MSB-3; NewIdeas 17:5x corrected PUSH800 §1 the same way). CPU calc: scratch price_zero.py; nothing ran
on the GPU. [K] measured, [P] priced from measured parts, [G] guess.

## 1. The price from zero: ONE stream is bounded by BYTES PER KEPT TOKEN, nothing else

The check is layer-sequential (router waits for the dense, experts wait for the router, the next layer waits for the
experts), so its floor is a sum, not a max. At 32 rows [P, real sizes, 280 GB/s]:

| term | per 32-row check | per kept token (kept 16.91) | what sets it |
|---|---:|---:|---|
| dense: 3.73 B params, 2.13 GB, 119 G MAC | 8.5 ms bf16 / 7.6 ms byte-bound (int8) / 6.4 (V4 3-bit) | 0.45-0.50 ms | compute at bf16, BYTES at int8: 4.9 ms of int8 math sits under 7.6 ms of weight bytes |
| experts: 46.5 distinct per layer (15.3 at 4-bit + 30.7 at MSB-3), C5 5.05 GB / C6 4.60 GB | 18.0 / 16.4 ms | 0.97-1.07 ms | DRAM only; 43 G MAC = 3 ms of ALU under 16-18 ms of bytes |
| cores (GDN state 226 MB, attention KV) | 1.2 ms | 0.07 ms | bytes |
| **bound at 100 % of every peak** | **C5 27.8 / C6 + int8 25.2 / C6 + V4 + int8 24.1 ms** | **1.64 / 1.49 / 1.42 ms** | so 609 / 670 / 703 tok/s with ZERO draft and overhead at today's kept |
| draft (target 3 ms) + overhead (1.5) | 4.5 ms | 0.27 ms | |

- **The theorem of one stream:** tok/s <= eff x 280 GB/s x kept / (dense GB + expert-union GB), minus draft and
  overhead. During the expert phase (65 % of the check) the matrix units idle at ~17 % load; during the dense phase the
  DRAM is already at ~250 GB/s. There is no idle unit to move work to: the CPU's matrix units and the ANE share the same
  DRAM bus, and the check is DRAM-bound as a whole (6.7-7.2 GB in 24-28 ms). Compression of expert bytes is <= 6 %
  (TRIED, entropy 3.70 of 4 bits). **So every lever is one of three: fewer bytes per check, more kept per check, or a
  higher fraction of the byte peak. Nothing else exists for one stream.**
- **Kept needed for 775 (draft 3, overhead 1.5), by engine efficiency** [P]:

  | engine | C6 + int8 (bound 25.2) | C6 + V4 + int8 (24.1) |
  |---|---|---|
  | 100 % | kept 23.0 | 22.1 |
  | 90 % | 25.2 | 24.2 |
  | **85 %** | **26.5** | **25.4** |
  | 70 % (check 34-36 ms) | 30.1-31.4 | |
  | 60 % (check 40-42 ms, Leap's Friday estimate for C6 + int8) | **34.6-36.1: above the 33 a 32-row check can keep** | |

- **So the frame's kept 22.3-24.0 at 85-90 % (PUSH800 §1, old sizes) is 25.4-26.5 at the real sizes: x1.50-1.57 vs
  today's 16.91, with the engine at 85 % AT THE SAME TIME.** At Leap's Friday estimate (36-42 ms measured) 775 is out of
  reach at 32 rows at ANY drafter: the check alone is 36-42 of the 41.3 ms that 32 kept tokens buy at 775.
- Bytes per kept token: 398 MB today (C6 + int8). 775 at 85 % allows 4.1 GB per check at kept 16.91, 6.0 at 23.15, 6.6
  at 25, 7.1 at 26.6 (C6 + int8 reads 6.73; C6 + V4 6.40). Read it either way: the bytes must fall ~40 % or kept rise
  ~55 %, or both halfway (bytes -20 % AND kept x1.25).

## 2. What a Saturday-sized engine keeps or cuts to reach >= 85 % of the bound (29.6 ms at C6 + int8)

Today's measured lossy 32-row check is 56.4 ms [K]; the gap to 29.6 is 26.8 ms. Priced term by term from Leap's
measured parts, ordered by what first principles says is the excess (floor = bytes at 280 GB/s):

| term | measured today | floor | excess | the cut |
|---|---:|---:|---:|---|
| the experts' lossy one-call | 4.60 GB at 171-184 GB/s quiet = 25-27 ms (the engine's 20.4 was at old bytes) | 16.4 | **~9-11** | stream the lossy call like the exact path's kernel (~290 GB/s on real sizes, CHECK55 pass K): one group per expert, all its rows in one tile, the MSB plane read once for both tiers. This is the single largest term and Leap has it as "Thursday, no promise": it should be first, since nothing else closes 85 % without it. |
| dense matmuls | 12.97 (dn32q 12.76) | 7.6 (int8 byte-bound) | 5.2 | int8 dense is not worth -0.44 ms as measured: it is what makes the dense phase BYTE-bound (4.9 ms of math under 7.6 ms of bytes). It pays only once the kernels run at the byte peak (qmm32n at 1.5x its floor today). |
| HC read (97 reads x 3 dependent kernels) | 6.3-8.1 | 2.25 | 4-6 | fold K2B into K1C's last threadgroup; one read per check for all 32 rows (hc8r32). |
| GDN recurrence at 32 rows | 4.3 (sequential) | 0.8 | 3.5 | a tree of depth ~21 has 21 sequential steps, not 32: batch the recurrence by DEPTH (siblings of one depth in one step); the chunked WY form for the rest. |
| 1c glue + router + MLX glue | 1.7 + 3 + (1.8-3.4) | ~1 | 5-7 | the tiers are a function of (row tier, rank): fold into router_fused (Wed). |
| two engine regressions | 1.6 | 0 | 1.6 | revert. |
| **sum** | **56.4** | **25.2** | **~29-34** | **-26 to -30 ms if every row above lands: 26-30 ms = 84-97 %. Leap's own Friday estimate is 36-42 because the first row is unpromised.** |

- The CPU: 27.9 ms of encode + 10.6 of glue per check today. Once the GPU is under 30 ms, the CPU is the wall unless
  the glue is fused (-10.6) and launches fall from 4,022 toward ~2,500. The Saturday engine must CUT launches, not
  optimise kernels: 4,022 dependent dispatches cost ~5-6 ms of GPU gaps alone (1.3-1.6 us each) and 28 ms of CPU.
- What a Saturday-sized engine should NOT do: anything that adds bytes per check for kept it cannot certify (the 2B audit
  at P = 2 is 2.12 GB per check, +32 % bytes for +8.9 % kept: a loss at any efficiency).

## 3. Rows per check: the optimum moves with the fixed cost (information, not a 775 lever)
At the floor, 32 rows beat 64 (the union grows ~x1.42 lossy, the dense compute x2, the GDN x2): 569 vs 455-526
tok/s at C6 + int8, zero excess [P]. At the Friday engine's 10-20 ms of fixed excess (launches, CPU, glue are per CHECK,
not per row), 64 rows with K128's kept (23.7, HE+ exact) and early death for the union reach 431 / 364 vs 426 / 340
for 32 rows: +1..7 %. Without early death they lose 10-12 %. So: no.

## 4. Ideas (not on the kill list)

### Idea 1 (best): CONFIDENCE-SPLIT LAMBDA: the sure rows' bytes are 60 % of expert bytes and close-call makes them safe to cut
- **Mechanism.** The lossy check's expert bytes split as 40 % for the unsure rows' full top-10 at 4-bit and **60 % for
  the sure rows' ranks 0-1 at MSB-3** (30.7 of the 46.5 distinct experts per layer; 3.0 GB per check). Those rows are
  where the drafter is right 99.5+ % (p1 >= 0.95) and where close-call d1 now accepts the draft at any near-tie. Give
  the tier rule two lambdas: lambda_unsure as C6 (1.8e-2) and lambda_sure 2-4x harsher (or rank 0 only), so the sure
  rows' second expert is dropped or kept at MSB-3 only where its importance is high. The safety net is close-call: a
  degraded sure row that flips becomes a near-tie in the degraded model and the draft (right 99.5 %) is accepted; the
  check still catches the drafter's rare confident errors with its remaining experts.
- **Number.** Sure-row rank-1 experts gone: -1.2..-1.5 GB per check (-18..-22 % of all bytes) = **-4.3..-5.4 ms at 100 %,
  -5.1..-6.4 at 85 %**: C6 + int8's bound 25.2 -> ~20 ms. In kept terms at 775 / 85 %: kept needed 26.5 -> ~22.5
  (x1.33 instead of x1.57 vs today). MSB-2 on sure rows instead: -1.0 GB, -3.6 ms.
- **Tag.** Adapted, re-opened: the lambda rule (C6) and the 1c confidence tiers exist; Q1 (top-2 on sure rows) and
  TOP1-QOM (rank 0 on sure rows, SSD era, PASS x3 with rank 1 at 4-bit) are the family. New: the split lambda keyed on
  the drafter's confidence with close-call as the net at 64 GB, priced per DISTINCT expert read, not per residency.
  Lossy (it is a check-bytes lever; EvalSuite gates it like G1).
- **Screen (<= 10 min CPU, NewIdeas' machinery):** ed_chain_bytes.py's loop on out_code24/scores with the tier rule's
  importance I = sum_d s^d g(cf_d) w^2, g = 1 for unsure rows and 1/2, 1/4 for sure rows (the row's cf from the aligned
  draft; if the scores files lack it, the position discount s^d with s 0.9 on rows 0-7 is the stand-in): expert GB per
  32-row check and the distinct count per layer vs C6's 4.61 GB. Line: >= -15 % bytes.
- **Quality risk and what kills it.** Sure-row flips (Breakthrough's map metric; DM's line 1.34 % code flips) and the
  kept loss when a flip is NOT a near-tie. Needs one map arm on the GPU (Breakthrough, ~1 h, the G1 screen's format) and
  the close-call replay with the degraded logits. KILL: code flips > 1.34 % at lambda_sure 2x, or kept -3 % in the
  replay, or (the honest one) the map showing the sure rows' rank-1 weight share above ~0.35 (then rank 1 is not minor).

### Idea 2: THE STREAMING-RATE CUT (the experts' call at the exact path's 290 GB/s)
- **Mechanism.** Section 2's first row. The lossy one-call reads 4.6-5.05 GB per check at 171-184 GB/s (quiet, exp32)
  while the exact path's expert kernel streams the same records at ~290 GB/s. The lossy call is slow because of its
  FORM (groups by (expert, tier), segments of <= 16 pairs, the MSB plane read twice for mixed-tier experts), not its
  bytes: all-MSB-3 and all-4-bit masks run at the same ~125-140 GB/s (ROUGH nlv). Make the lossy call the exact kernel
  with a per-expert LSB-skip flag: one group per expert, up to 32 rows in two MMA tiles from one read, the LSB plane
  fetched only for 4-bit rows.
- **Number.** 4.60 GB at 175 -> 290 GB/s: **26.3 -> 15.9 ms, -10.4 ms per check** [P]; it alone takes the measured
  56.4 to ~46 and is the difference between a 36-42 ms Friday check and one near 30. No kept or quality effect (exact
  per row by construction; bitwise vs today's lossy call).
- **Tag.** Engineering, Leap's (CHECK25 Thursday item, "no promise"; the merged-group kernel ON HOLD after nlv said
  re-reads mostly hit the SLC). My point is the ORDER: it is the largest excess by a factor of two and the only term
  whose floor is reachable (the exact path proves it on real sizes).
- **Screen (desk bound, no GPU):** CHECK55 pass K's 33.71 / 50.65 ms for 9.78 / 14.66 GB (290 GB/s) against exp32's
  571-614 us for 105 MB. The 10-min GPU microbench is Leap's Tue slot #1 (exp32 nlv at one group per expert).
- **Kills it:** the exact kernel's rate not holding at 46 small groups per layer (1.7 MB each, 6 us of streaming per
  group at 290 GB/s vs ~5 us of fixed cost per tiny dispatch: the kernel must run every group of a layer as one launch).

### Idea 3: K128 + EARLY DEATH + the lossy rules is the ONLY shape that reaches 775 on paper (a composition, priced)
- **Mechanism.** Not new (bet 11 + #8 / #10 / #13); priced here from zero because it is the one shape where the
  numbers close. A 128-node tree checked with early death: 128 rows through layers 1-16, the survivors (~32) through
  17-48; expert union ~x1.9 for a third of the layers then ~32-row; dense compute 2x (int8 9.9 ms); bytes 2.13 GB
  unchanged. Kept: K128 HE+ exact 23.7 x close-call d2 (~1.07) x span-conditional / inert comments (x1.17-1.30) = 29-33.
- **Number.** Check ~34 ms at 100 % (experts 6.0 GB 21.4 + dense 9.9 + cores 2 + the early-death taps), ~40 at 85 %;
  tok/s = 31.7 / (40 + 4.5) = **~710 on HE+** [P]; 775 needs the engine at ~92 % or kept 34. MBPP: K128 keeps 11.3,
  so ~350.
- **Tag.** Composition of ON items; exact in the tree, lossy in the rules. The honest read: **775 by Saturday on HE+
  needs all of: idea 2's rate, bet 11, #8, #10 and #13 (Kamen's pending comment decision), each measured.** Any one
  missing lands at 530-650.
- **Screen:** none of mine; the Wed early-death probe (PASS >= 90 % of off-spine children droppable at layer 16) is the
  gate. **Kills it:** that probe BETWEEN / KILL, or Kamen vetoing #13 (then kept ~25, ~560 tok/s at 85 %).

### Idea 4: A different one-stream shape? No. The bound says why.
- Checked and dropped by the arithmetic of section 1: draft / check overlap (the draft needs the check's last-layer
  state; an early-tap head is KILLed; hiding 3 ms is +10 % at most); two half-batches pipelining dense (ALU) against
  experts (DRAM) across layers (a second GPU stream is on the kill list, kappa 0.93; and the expert read is DRAM-bound
  for both halves, so the halves double the expert bytes unless fused, which breaks the overlap); the CPU's matrix
  units or the ANE taking a slice of the dense (they share the saturated DRAM bus; a 30 % slice on the CPU reads 0.64
  GB at ~120 GB/s = 5.3 ms, no faster than the GPU and competing for its bytes); skeleton-of-thought parallel bodies
  within one answer (multi-stream in spirit; code loses sequential context); accepting sure drafter tokens unchecked
  (every token must pass the 125B once to build state, so no bytes are saved; only the block's death moves, +5 % kept at
  p1 >= 0.95, and it is quality we cannot certify); 64-row checks (section 3).
- **What is left for one stream is exactly the three terms:** bytes per check (ideas 1-2, Breakthrough's G1 / G2),
  kept per check (the lossy rules and bigger trees with early death), the fraction of peak (section 2's cut list).

## 5. One line for the plan
At the real sizes, 775 needs kept 25.4-26.5 per 32-row check with the engine at 85 %, or kept ~22.5 with the sure
rows' second expert cut (idea 1) and the experts streaming at the exact path's rate (idea 2). The Friday 36-42 ms check
cannot reach 775 at any kept; the first engine item is the lossy one-call's 175 -> 290 GB/s, worth -10 ms, before any
other kernel.
