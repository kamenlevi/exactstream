# BREAKTHROUGH PUSH: ranked shortlist (NewIdeas, Tue 3 Nov 2026; completed 17:51 with all four thinker files in)

Inputs:
- Fable A push/SSD_WALL2.md (17:14), Fable C fablec/UNION_STACK.md (17:16), Fable B push/ROW_WALL.md (17:4x),
  Lunatic lunatic/LUNACY2.md (17:4x).
- Breakthrough's staged-check KILL (9b2cbdf), Leap's WHOLE-CHIP pre-registration (TABLE_350), MoonShots' M-27 /
  M-27b.
- My replays on our data: band45 stack prices (data/band45_oea/*stack*), calls_count.py, QOM masks.

Marks: [K] measured, [P] arithmetic on [K], [G] judgement. Ranked by halved gain x odds, cheapest verdict first when equal.

## The frame (what any design must beat)

**Code at 45 GB.** Fable C's cycle model (calibrated on HumanEval/68, 183 vs 187 ms):
- The best FAST union stack closes the expert term. What remains is the exposed SSD (6-7 ms at 60 % hiding) and
  kept per check.
- 350 needs kept >= 12.2 per check AND the SSD hidden [P].
- Decision-grade (halved): FAST ~210-225 at 45 GB, ~215-230 at 50 GB; EXACT ~130-150.
- Exact 350 single-stream is closed on this machine (Fable C, Lunatic round 1).

**Chat.** At kept ~3 per check the cycle is the dense pass + draft + chain; 350 needs kept >= 17 per check. No byte
or union cut moves chat toward 350; only the row wall (Fable B) and the drafter (MTP Head) do.

**RAM roofline [P].** Each pass reads the 3.44 GB dense part plus the check's expert union. Leap measures 258-287 GB/s
streaming; the expert kernel holds ~171-210 GB/s of it.
- Code: EXACT 6.04 GB union at 8.5 kept -> ~1.1 GB per kept token -> ~250 tok/s roofline at perfect kernels and
  zero SSD.
- Chat: ~2.6-3.0 GB per kept token -> ~90-110 tok/s.
- So for 350, what counts is fewer RAM bytes per kept token (union cuts, more kept), not only SSD time.

## The SSD side, priced on our data (calls_count.py; cap45 live checks, C 256, seed 0)

Fable A's fit: busy = 0.283 ms per loader call with misses + MB / 12.7 GB/s.

| rule (reads) | misses per check | calls (base + LSB rows) | JOINT calls | MB per check | busy: fixed + bytes (fixed share) | busy with JOINT |
|---|---|---|---|---|---|---|
| EXACT | 259.6 | 70.8 | 41.9 | 398 | 20.0 + 31.3 = 51.4 (39 %) | 43.2 |
| U3 (Q1 0.9 + 3-bit sure rows) | 136.9 | 60.8 | 39.0 | 213 | 17.2 + 16.8 = 34.0 (51 %) | 27.8 |
| HYB 0.9 u8 + t3 | 119.4 | 56.5 | 37.4 | 175 | 16.0 + 13.8 = 29.8 (54 %) | 24.4 |
| QOM theta 0 (Q1 0 reads) | 145.0 | 49.4 | 35.1 | 150 | 14.0 + 11.8 = 25.8 (54 %) | 21.8 |
| QOM theta 0 + LSB-on-miss (Q1 0 3b reads) | 34.6 | 18.7 | 15.8 | 65 | 5.3 + 5.1 = 10.4 (51 %) | 9.6 |

At C 320 (the volatile tier): EXACT 36.8 -> 30.9 with JOINT; HYB u8 + t3 23.5 -> 19.6; QOM-full 9.0 -> 8.3.

Reading [P]: the byte cuts shrink bytes faster than calls, so the per-call share of SSD time rises from 39 % to 51-56 %
under every FAST stack. Fable A's D1 is therefore worth MORE on top of the FAST stack, not less. JOINT alone is
-8 ms of busy per check on EXACT and -6 ms on U3.

## Ranked shortlist

### A. Single stream (the push's 350): ranked by halved gain x odds, cheapest verdict first

| # | design (thinker -> owner) | track | tag | halved gain at 45 GB | odds [G] | verdict from (pre-registered) | when |
|---|---|---|---|---|---|---|---|
| 1 | FAST union stack on code: HYB 0.9 u8 + t3 or REACH + t3 (Fable C); U3 is its first engine step | FAST | adapted | code ~210-225 tok/s decision-grade (from 63.6); expert groups x0.50 | high: map PASS on code (REACH live, HYB at 16 rows) | U3 smoke tonight (BigMoE); RELAX Thu (EvalSuite); HYB live-length map arm | tonight / Thu |
| 2 | D1 zero-latency bursts: JOINT one burst per layer + keepalive + >= 256 KB pieces + release pickup (Fable A -> BigMoE) | EXACT | adapted | +5 % EXACT by day. Worth more under the FAST stack: the per-call share of SSD time rises 39 % -> 51-56 % [P, calls_count.py], and JOINT alone cuts U3's busy 34.0 -> 27.8 ms. Chat +7-10 % [G] | high for JOINT (+7-8 % measured at 24 GB) | c45tj_224 vs c45t (<= -8 ms busy); ka1 vs ka0 (+2 %); a 10-min pieces microbench | tonight + Wed |
| 3 | Volatile tier at C 320 (PURGE-1 -> BigMoE) | EXACT | adapted | EXACT +17-21 % [P] (-35.5 % SSD bytes); FAST ~+4 % (the stack already removed those bytes) | high: mechanism [K]; the victim design matches fitted pools [P] | vt1_224 smoke; prediction 79.2 -> 52.1 MB per kept token (-34 %) | ~04:00 |
| 4 | D2 the queue never empties: layer-ahead forecast reads into the tier (Fable A) | EXACT | adapted | +9 % (exposed SSD 24 -> 6-8 ms); with D1, ~0 exposed on code | conditional on pre-route recall >= 0.85 | pr_256_preroute.txt tonight, then the CPU exposure replay (<= 8 ms) | Wed |
| 5 | M-27b native layout + 11-bit packed scales (MoonShots / Leap / BigMoE; Kamen-approved) | EXACT | adapted | -17.6 % code SSD bytes at C 256 [P]; NAX6-N's +21 % comes on top as a new reference | high: P11 bitwise [K] | M-27b live A/B (tokens bitwise, code MB per kept <= -10 %, steady >= +3 %) | Thu >= 15:00 |
| 6 | One-call OOO (Fable C; engine) | EXACT | engineering | -3.7 ms per check ~ +6-7 % [P] | medium | NOT pre-registered yet | - |
| 7 | COLD-CUT: unsure rows drop cold ranks >= 4 (Fable C) | FAST | adapted | bytes x0.35 vs u8 + t3's x0.46 (-22 % at C 320) -> +2.5-3 % halved | 50 % PASS (Fable C) | map arm hyb_cold4_t3 (UNION_STACK sec 5) | needs a map slot |
| 8 | Top-2 cheap chain as a route predictor for the next exact pass (Fable B D2) | EXACT | adapted | +7 % code halved, 0 chat | BETWEEN 50 % / PASS 35 % (Fable B) | S2, 5-min CPU replay (recall >= 0.45 at <= 1.3 requests per hit) | today, Fable B |
| 9 -> now first on the FAST track | Q1-ON-MISS / MISS-CUT (NewIdeas / Fable C). **Map PASS / PASS / PASS at cf 0.9 and 0.5 (Tue 22:59); Kamen's decision 2 = YES (23:1x)** | FAST, cache-dependent | adapted | on code dominated by the union cuts (no group cut). As a stack layer: HYB-MISS x0.35. QOM-full (theta 0 + LSB-on-miss) cuts SSD busy 51 -> 10 ms per check [P] | high on quality (94-98 % of experts kept) | map Wed after 15:00 (+ Breakthrough's res_c256) | Wed |
| 10 | WHOLE-CHIP: CPU / ANE computes part of each layer's experts beside the GPU (Leap) | FAST numerics | adapted | +20 % expert throughput if PASS; ~80-110 GB/s idle during the expert kernel [K] | 15 % (Leap) | TABLE_350 micro-screen | Leap's call |

### B. Step changes beyond one stream (Lunatic)

- **Route W, one user is a batch (FAST): ~880-1,270 tok/s aggregate at 8-12 streams [P].**
  - The user census PASSED: 63 % of Kamen's agent generation in the last week ran while >= 8 sessions were generating
    [K].
  - It is the only route to ~1000 the physics leaves. It is not "one chat", so whether it counts is Kamen's call.
  - CPU replay done 17:57 (scripts/route_w.py, data/ROUTEW_read.txt): KILL at today's drafter.
    - Groups per kept token at 8 streams: Q1U 0.9 162.5, EXACT 236.9. At 12 streams: 144.6.
    - The Lunatic's pass model then gives an aggregate of 391-515 tok/s FAST at 8-12 streams [P] (whole model in RAM, no
      SSD), vs 85 for one mixed stream.
    - ~1000 needs ~3x more kept per row (the horizon fix): the same drafter wall as single stream.
- **Route X, EXACT owns the Mac: EXACT ~540-670 at 8-12 streams [P].** It needs Kamen to re-base EXACT on the
  all-3-bit store (52.8 GB; t3_all is code BETWEEN, chat +4-5 points on the map) and a footprint diet to <= ~6 GB of
  non-pool + wired memory.

### C. Killed or closed today, with the reason

- **Staged check, both forms** (Breakthrough 9b2cbdf; Fable B D1). A perfect picker caps at x1.25 chat / x1.09 code. A
  cheap verdict is worth only its excess over the drafter's fresh accuracy: 0.04 (code) / 0.14 (chat) tokens per row,
  because the bonus token already cashes in the top-3.
- **Trees at any row price this machine can reach** (Fable B): a node is worth 0.135 / 0.05 tokens.
- **The RAM-only verifier as a selector, and the chat sibling fan** (+1 %).
- **Parareal decoding** (48.9 % drafted after one divergence), **2-bit residency** (TRIED), and **Route C copy spans**
  (6.2 % of Kamen's output).
- **MASS-CUT**: the router's top-10 mass is flat.
- **LOSSLESS45 entropy coding**: BETWEEN; its decode-free part lives on as M-27b.
- **Fable B's LSB-adjacency read merge**: overstated about 5x.
  - My replay (cap45, C 256): only 11.1 % of LSB misses coincide with an MSB miss of the same expert, so adjacent
    placement saves 5.0 % of reads (3.9 % under U3), not 25-30 %.
  - Sidecar misses are promotions of base-resident experts. That is a pool-size lever (Fable A's 1.5x sidecar) or a
    FAST one (LSB-on-miss), not a layout lever.

### D. What 350 means now (the physics verdict)

- **Single stream.** The bound with the whole model in RAM, the draft hidden and an ideal drafter is ~315 EXACT /
  ~350 FAST (Lunatic [P]).
  - Fable C: 350 on code needs kept >= 12.2 per check AND the SSD hidden.
  - Decision-grade by the end of this push: FAST code ~210-230 at 45-50 GB, EXACT ~130-150.
  - After the stack, the remaining gap is the drafter (kept per check) plus SSD hiding.
- **Chat.** 56-60 tok/s at kept 3; 350 needs kept >= 17 per check. Only the drafter moves chat.
- **The step change beyond 350 is width (Route W)**, not a faster single row.

**Decisions for Kamen:**
1. Does an aggregate over his parallel agent sessions count (Route W)?
> (a quote from the project owner, kept private)
3. Re-base EXACT on all-3-bit (Route X)? Only if t3_all passes chat same-scores.
