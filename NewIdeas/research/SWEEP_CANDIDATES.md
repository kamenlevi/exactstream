# THE SWEEP: candidates (Wed 4 Nov 2026 14:3x)

Kamen's ask, relayed by the orchestrator at 14:3x: re-check every KILL / PARKED from before the 11-02 two-track rule that
was judged on an exact or error bar instead of task results, and every published method in our notes set aside as
"lossy". Then price each under today's FAST stack (QOM + DRAFT17 + capacity counted twice).

This page is the extraction: a read-only sweep of TRIED, DEAD, QUEUE, FAST_TRACK, BYTES10, the ledger, MoonShots'
IDEAS / PRIOR_ART / HANDOVER and FableGod's FINDINGS / SSD_WALL / lunatic. It quotes the notes' own numbers. The pricing
and the ranking are in research/SWEEP.md.

Left out: anything already judged on map flips or task results. That covers the QOM / Q1 / Q1-UNION / OEA / HYB / COLD
arms, t3_all, 3-bit sure rows, lever 6, screens 2b and B', the gated bonus row, DRAFT17, POOLSPLIT, HC8 (D4), TK2 top-8
at 12 GB, the b16 / FM1 / B12 budgets, D1 frugal, and the decision-map P-frame and shared-expert arms. Pure kernel and
exact items are also left out.

Rnd19 is defined at ~/MoonShots/IDEAS.md:642 (Round 19, Oct 19): the candidates at :665-675, #5 (plane-aware prefetch)
at :671, the quality verdict at :680-687, prior art at :690-699.

Source key:
- T = ~/TRIED/TRIED.md (line numbers as of 11-03), TM = ~/TRIED/moonshots.md, N = ~/TRIED/NOVELTY.md
- D = ~/NewIdeas/DEAD.md, Q = ~/NewIdeas/QUEUE.md, FT = ~/NewIdeas/FAST_TRACK.md, BX = ~/NewIdeas/BYTES10.md,
  L = ~/NewIdeas/research/TEN_X_LEDGER.md, PP = ~/NewIdeas/research/PUSH_PRIOR_ART.md
- I = ~/MoonShots/IDEAS.md, PA = ~/MoonShots/PRIOR_ART.md, H = ~/MoonShots/HANDOVER.md
- F = ~/FableGod/FINDINGS.md, SW = ~/FableGod/SSD_WALL.md, LH = ~/FableGod/lunatic/HANDOFF.md,
  SR = ~/FableGod/decision_map/out/SHADOW_read.txt

| # | id / name | source | date | verdict | bar | mechanism | saves | pub / new | key numbers (as written) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B10 / B3 / B10a / B10-r: TRL-2 (2.06 b) and the 2.40 b trellis residency | T:328-333, D:232-239, Q:222 | 10-25 (reopened 11-04) | KILL, "data-aware 2-bit route CLOSED"; 11-04 bytes BETWEEN, screen 7 pre-registered | output error vs MSB-3 (line 1.5x, 1.8x unseen) | a second store of QTIP-style records | capacity + bytes per miss | published codes (QTIP); MoE residency use new | B10-r 1.854x MSB-3 (2.287x unseen prose); B10a 2.49x; LUT decode 2.62-2.88x 3c's 3-bit (T:329) |
| 2 | QTIP / AQLM / QuIP# codebook and trellis 2-bit formats | T:341, PA:59 | 10-19 | KILL (prior art) | "not nested, CUDA-centric" | a separate, non-nested 2-bit store | capacity + bytes per miss | published | AQLM 1.98-bit Mixtral Wiki2 4.61 vs 3.46 fp16; "QTIP 3-bit near fp" |
| 3 | RESID27 mixed-bit residency at 24 GB | T:216 | 10-24 | KILL "with today's codes" (unsure) | mass-weighted error vs MSB-3 | all 512 experts: hot resident at 2.5-3 b, cold stored at 2 b | misses + capacity | adapted (HOBBIT / DynaExq tiers) | "<= 16 misses per token needs ~288-320 of 512 resident"; 2.9-4.4x MSB-3 |
| 4 | T3R: TRL-3 base for every expert + a 1-bit trellis residual for ranks 0-1 | T:284, D:241-246, Q:235 | 10-25 | KILL at 12 GB (K1); quality K2 PASS; 24 GB not pursued | 12 GB bytes line (-20 %) | trellis base + top-rank residual | bytes per miss + capacity | adapted (QTIP + residual) | 12 GB -16.4 %; 24 GB 115.9 -> 82.0 MB/token = -29.2 %; ranks 2-9 at 0.5x MSB-3; D:246 "reopen only for a non-exact 24 GB mode with disk for a second format" |
| 5 | TRL-3 (3INST, 3.07 b) as the base; R-17 T3-45 | T:320, T:371 | 10-25 / 10-27 | PARKED (quality PASS; GPU encoder; Kamen's call); R-17 KILL at 45 GB | encoder cost; 45 GB speed | replace MSB-3 records with TRL-3 | capacity + bytes per miss + lower error | published (QTIP) | ~0.5x MSB-3's error at -13 % bytes (0.62x unseen prose); 3INST 1.257 vs 3c 1.240 ms per layer |
| 6 | HOBBIT / DyMoE separate low-bit copy on a miss (GPTQ-2 / GPTQ-3 copies) | PA:19, :49, :54, :97, :104; T:368 | 10-19 / 10-24 | set aside "lossy"; Q2 KILL | equal-accuracy bar; 1.5x MSB-3 line | a separately quantized copy read on a miss, or held warm | bytes per miss + capacity | published (HOBBIT 2411.01433, DyMoE 2603.19172, MC-MoE) | GPTQ-2 g64 3.65x MSB-3; GPTQ-3 g64 0.61x at 3.5 b; HOBBIT GSM8K 0.52 -> 0.51 |
| 7 | VD / M-04 POMVS: variable-size slots (2-plane residents; demote instead of evict) | T:321, T:884, Q:90, H:672-701 | 10-29 | M-04 BETWEEN; VD PARKED; VD + LP1 over the gate (R-51) | strict KLD at engine shares | a 2-plane resident uses 72.7 % of a slot | capacity | adapted (SliceMoE / D2MoE) | VD 73.4 vs POM 81.1 MB per token (-9.5 %); VD + LP1 65.6 (-34 % vs v3 99.1), then 109 / 112 / 107 % of the KLD target |
| 8 | B9 / B7: HOBBIT-style 2-bit warm tier for ranks 4-9 | Q:219, Q:221, T:334, H:826-828 | 10-25 | "KILLED by the 30 % line" | a bytes line (exact-rules replay); quality never measured | v3 hot experts + 2-bit copies of warm experts | capacity + misses | published (HOBBIT) | -18.0 % (16-row -21 %); "~40 % residency fits in 24 GB"; never re-priced under QOM |
| 9 | Rnd19#1 2+1+1 bit-plane cache (+ memory halving, records4) | I:667, I:680-687, T:933, T:934, T:380 | 10-19 / 10-21 | KILL | 35B perplexity; number of separate reads | 3 nested planes as LRU units | capacity | adapted (Any-Precision, DynaExq) | at 24 GB "-30 % memory at equal SSD traffic"; +2.46 % ppl (35B) |
| 10 | FAST 3-bit prefill | T:974, BX:29, T:968 | 10-20 | KILL (on paper) | exactness | prefill reads MSB-3 parts only | bytes per miss (prefill) | unchecked | not priced; at 24 GB process bytes 409-427 vs decode misses 226-247 MB per kept on code (W1B) |
| 11 | Rnd19#5 plane-aware prefetch | I:671, T:119 | 10-19 | PARKED (unsure) | never run; non-exact | prefetch only the MSB-2 plane of predicted experts | bytes (prefetch waste) | adapted (HOBBIT-style) | napkin "1.8x more predictions per byte"; the 24 GB engine reads 18-26 % more than it misses |
| 12 | Plane-granular cache (Rnd18#1 / Rnd19), SliceMoE unified LRU, bit-plane residency | T:224, T:96, I:606, I:720, LH:13 | 10-19 / 11-03 | KILL / BETWEEN | speed | MSB and LSB as separate cache units | capacity + bytes | published (SliceMoE 2512.12990, D2MoE) | sim -41 % demand MB per pass at 34 GB; prototype 14-26 % slower |
| 13 | 3B: no LSB sidecar + cold ranks 4-9 at 2 bits + a 2-bit cache | T:326, Q:219, H:832 | 10-25 | KILL | KLD (ranks 0-1 at 3 bits x2.1-3.7 of v3) | drop the sidecar; cold ranks at 2 bits | bytes per miss + capacity | adapted (HOBBIT) | -38.5..-39.8 % bytes; the no-sidecar half later judged on the map (t3_all) |
| 14 | Dense / trunk diet to free RAM (HC4, MSB-3 dense, 8-bit HC + PLE, trunk q4) | T:910, T:906, T:383, T:889, LH:5 | 10-19..11-03 | KILL / BETWEEN | hard KLD; ppl | a lower-bit trunk; freed RAM to expert slots | capacity | published | HC4 -0.95 GB, hard KLD +0.0243; 8-bit HC + PLE -627 MB "~+3.7 % expert slots at 17 GB"; MSB-3 dense +12.5 % ppl |
| 15 | 2-bit tails (TAILS2, last-rank-only, MSB-2 lowest ranks, plan4433332222) | T:393, :308, :311, :300, :292; D:135; FT:166-170 | 10-19..10-22 | KILL / PARKED / BETWEEN | 1.10x KLD gate | the lowest gate ranks at MSB-2 | bytes per miss | adapted (HOBBIT / Any-Precision) | TAILS2 code 0.00267 vs 0.00253; plan4433332222 "dropped 10-23 for priorities, not for a result" |
| 16 | Low-bit quality fixes: M-11 BIASCORR, M-11b LOWRANKCORR, activation-aware refit | T:798, :799, :795, :804; TM:309; Q:1308 | 10-19..11-01 | BETWEEN / PARKED / PASS | strict KLD gate | correct the low-bit tier's error | enables cheaper records | published (DFQ, LQER / QERA, DWQ) | M-11 block error -21.6 %; M-11b 0.67-0.78x; 2-bit refit x0.73-0.84 |
| 17 | POM / LP1 / LPOM / POM12: tail reads at fewer bits | T:377, :378, :297, :208, :103; Q:89; H:685, :693-700 | 10-21..10-29 | KILL (tok/s) / BETWEEN | engine speed; KLD at engine shares | a miss used only at rank >= 4 reads 2 planes | bytes per miss | adapted (HOBBIT, SliceMoE) | POM -19.0 % bytes per token, tok/s -1.9 %; "reads got smaller, not fewer" |
| 18 | LSB-on-miss / never-stall LSB fallback, D1j, margin-gated LSB skipping | T:95, :111, :315, :1030; N:56; PP:171 | 10-19..10-22 | BETWEEN / PARKED / KILL | nondeterminism + ppl | run MSB-only when the LSB part misses | bytes per miss | published (SliceMoE) | -62 % demand MB per pass at 34 GB; its bounds are on the map now (3-bit sure rows PASS) |
| 19 | Rnd19#2 gate-mass-adaptive tiers (and TIERBYW) | T:283, I:668, I:723-725, T:835, D:204-209 | 10-19 / 10-22 | PASS on 35B, never built / KILL | 35B ppl; error proxy | a token's gate mass sets each rank's bits | bytes per miss | published (SliceMoE, HOBBIT) | +0.17 % ppl at -22 % bytes vs fixed plans +0.6-0.8 % |
| 20 | B2 ungated cold-only 2-bit reads | T:290, D:224-229 | 10-25 | BETWEEN | code P99.9 | cold misses read at MSB-2 | bytes per miss | adapted (HOBBIT) | -18-20 % as screened; faithful replay -11.7-12.9 % |
| 21 | #28 layer-split fast mode / resident-only catch-up | T:108, T:109 | 10-18 / 10-19 | PARKED (design only) | non-exact | no waiting in layers 0-9 | misses | new | no numbers; layer sensitivity is uneven (L36-39 3.51 % vs L08-11 1.19 % code) |
| 22 | R-30 UNIQ / R-31 UNIQ-Q | T:314, T:115, Q:51, Q:53 | 10-27 | PARKED | no quality harness | drop a check's single-use low-rank experts | bytes / misses | adapted (MoE-Spec) | -17.6 % at 2.3 % gate mass; mostly subsumed by QOM at cf 0 |
| 23 | Dynamic / adaptive top-k (DYNK, L1-17b, TK3, ACE / Elbow / MoDES, top-6 / 4, S9, M-09d) | T:374, :201, :161, :800, :319, :162, :237, :824, :831; I:243 | 10-18..10-31 | KILL / BETWEEN / PARKED | strict KLD; error ratio | fewer experts per row | bytes / misses / compute | published | DYNK -13.2 % bytes, KLD x1.70 of v3; M-09d 124-145 % of the gate; on the map top_k8 0.65 / 5.24 / 4.31 % |
| 24 | M-12 RARE2 (+ static per-expert precision) | T:362, T:834, H:243-259 | 11-01 / 10-21 | KILL (bytes) / KILL (prior art) | 45 GB bytes line | rarely used experts stored at 2 b | bytes + capacity | published (MC-MoE, MxMoE) | 0.97x REF2-T's error with M-11; -4.3 % SSD MB per kept at C 256; not priced at 24 GB |
| 25 | PRUNE26 / REAP / PRN static pruning | T:198, :379, :826, :814 | 10-19..10-26 | KILL | policy (all 512 experts stay) + a task drop | keep a subset of experts | misses + capacity | published | out of bounds by Kamen's rule |
| 26 | Super-block scales (mxfp4 / Q4_K style) | T:392 | 10-18 | KILL (unsure) | napkin, prior art | lossy shared scales | bytes + capacity | published | ~5 %; the exact 11-bit pack is already x0.955 per part |
| 27 | Rnd19#4 salient-group refinement | T:385, I:670 | 10-19 | KILL (unsure) | napkin | refinement bits only for high-energy groups | bytes per miss | published (SpQR, AWQ) | never measured |
| 28 | Other low-bit formats (MSB-2 plane, RTN, rotated, C1 O2, KLT) | T:365, :389, :335, :338, :369 | 10-19 / 10-24 | KILL | ppl / error vs MSB-3 | all experts at 2-3 b | capacity | published | MSB-2 +11.3 % ppl; RTN-2 +18 / +15 %; all worse than TRL-2 |
| 29 | ASTC / BC texture-decoded experts | T:316, :324, :327, :382 | 10-17..10-19 | PARKED / KILL | napkin; a download needed | texture units decode 3.56 b blocks | bytes | published | -21 % vs 4.5 bits, but today's base is already 3.5 b |
| 30 | Expert substitution / cache-aware rerouting (SERE / SPICE, I6, SWITCHCAP) | T:244, :1025, :215, :348, :830, :245; PA:23, :28 | 10-17..10-30 | KILL | "not lossless"; KLD | replace a missing expert with a resident one | misses | published (2412.00099, SpecMD) | experts orthogonal (median abs cos 0.024); QOM is the adapted form |
| 31 | Sketch-on-miss / low-rank shadow tier | FT:101-103, SR, F:15, SW:26 | 11-03 | KILL | output error (GO <= 0.5) | SVD shadows replace missed experts | bytes per miss | published-adjacent (HOBBIT) | whitened r32 0.920 / r64 0.892 (a drop = 1.0): the expert spectrum is flat |
| 32 | Activation sparsity inside experts (TEAL / CATS, I3, B1, B1b, Rnd18#8) | I:209, T:347, :128, :130, :384 | 10-18..10-25 | KILL | energy line 0.90; IOPS | read only the active neurons' rows | bytes per miss | published | B1 0.769 energy at 30 % neurons |
| 33 | Expert merging / shared bases / delta coding | T:310, :339, :350, :352; PP:76; I:59 | 10-19..10-24 | PARKED / KILL | physics; prior art | rebuild experts from shared bases | capacity | published (MoBE, D2-MoE) | the mean expert holds 0.20-0.36 % of the energy |

Contradictions the extraction found:
- B7 / B9's kill reason: "KILL on quality" (SW:22) vs "KILLED by the 30 % line" (a bytes line; Q:219, H:828), with
  quality never measured.
- The 2.40 b mix's error: 1.87x MSB-3 (D:235) vs 1.62x (Q:1640). B10's headline is 2.49x (B10a) in some notes and
  1.854x (B10-r, the final test) in others.
- FAST_TRACK's 11-02 sweep took only kills "not on quality" (FT:6-7), so it filtered out every KLD or error-ratio kill
  by design. That is why B10 / T3R / TRL-3 / GPTQ-3 were never re-priced as FAST copies.
- Lever 6 tested only the crudest 2-bit form (the midpoint rebuild), not the bias-corrected one (M-11) or a trellis /
  GPTQ code.
