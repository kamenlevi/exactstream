# IDEAS_1117b: the cheapest extra row on the M5 Pro (fable-zero, Tue 17 Nov 08:2x-09:45, desk only)

The question (BREAK98b, lens a): ~96 more check rows for <= ~5 ms, i.e. <= ~0.05 ms per row; every compute unit and
every format; what a row costs from measured parts. CPU calc: price_1117b_row.py beside this file; nothing ran on the
GPU. [K] measured, [P] priced from measured parts, [G] guess. Records: check25_bound(_real).txt, treek_price.md /
treek_union.txt, WEEK98-M2/M2a/M2b (the expert one-call's profile), wc_screen.txt (WHOLE-CHIP), ANE-1/2/2b, M-19
CPUDRAFT, TRIED "Exact tree checks" (PARKED) and "Chunked (WY / UT) GDN verify" (KILL for EXACT modes), FAR's 0.28 ms
pricing, Fable B's union-constrained 9-of-10 finding.

## 1. The answer in four lines
1. **A row is dense MACs.** 3.61 G MAC per row (GDN projections 2.08, attention 0.60, lm_head 0.64, shared 0.24,
   router 0.06) [K from the config]. At the measured matrix peak (14.0 TMAC/s bf16, x1.55-1.73 at int8 activations,
   dn32q) that is **0.26 ms bf16 / 0.16 ms int8 per row, 0.13 ms for a leaf row (no lm_head)**. 96 rows = 12.4-15 ms at
   100 % of the matrix peak. **The brief's 0.05 ms per row needs ~3x this chip's measured int8 rate: not on this
   hardware.** No unit beside the GPU's matrix units is within 10x: the Neural Engine ran a 0.295 GB int8 package at
   ~0.5 TMAC/s effective (ANE-2b: 16 rows in 10 ms [K]) = 7 ms per 125B row; the CPU's fp32 path ran the same drafter
   call 5x slower than the ANE (M-19: 62 vs 12.4 ms) = ~35 ms per row; both share the DRAM bus the check saturates
   (WHOLE-CHIP: bw4 beside the GPU slowed it 22.8 %). So the cheapest row is a GPU int8 row, and only its NON-dense
   parts can be made free.
2. **Everything that is not dense MACs CAN be ~free for a row, and today is not:** (i) the experts' MMA tiles are 16
   rows wide and run at ~5.2 pairs per segment, 2/3 padding [K, M2a]: an extra row that routes inside the loaded union
   rides in the padding at zero MMA cost up to ~16 pairs per expert, and at K128 a row's experts average only 6.1 pairs
   per expert; (ii) the expert BYTES and the decode (the one-call is compute-bound on the nibble decode, 331 of 352 us
   without loads [K, M2]) grow with the UNION, x1.43 / x1.92 at 64 / 128 rows by the chain law, 15-25 % under it for
   trees: +4..+15 ms at the bound unless the off-spine rows are routed inside the union (Fable B: a constrained row
   keeps ~9 of its 10 experts); (iii) the GDN cores: Leap prices 35-46 ms at 128 rows because the recurrence is
   sequential per row and each fork copies ~50 MB of state per layer; the chunked (WY) tree form reads the 226 MB state
   once (0.81 ms [K]) and the intra-tree term is 3.6 G MAC = 0.26 ms, ~2 ms at any K <= 128 [P]. TRIED killed the chunk
   form "for EXACT modes" (reordered arithmetic): **in LOSSY mode it is admissible, like int8 and C6, and it is the one
   item that makes 64-128 rows possible at all.**
3. **The floor per marginal row on this chip: ~0.13-0.16 ms** (int8 dense, union-constrained experts, chunked cores, no
   lm_head on leaves) vs 0.28 today (FAR) and 0.34 if the union and the cores grow as priced. 96 rows = 12.5-15 ms, not 5.
4. **So "kept 24 for the price of a 32-row check" is not available by rows on the M5 Pro.** The cheapest-row design
   gives, at 100 % of every peak and a 5.2 ms draft [P]: K32 510 / K48 577 / **K64 606** / K96 579 / K128 545 tok/s;
   at 85 %: 447 / 506 / **530** / 505 / 474. The optimum is 64-96 rows, ~+15-20 % over K32, with 550 at the edge
   (K64 at >= ~90 %). Kept 24 at K128 costs 11-14 ms of dense MACs that nothing hides.

## 2. Idea 1 (best, the build that unlocks rows): LOSSY-MODE CHUNKED TREE GDN + ROWS IN THE PADDING
- **Mechanism.** Two kernel facts turn 64-96 rows from "35-46 ms of cores + x1.4-1.9 experts" into "+2 ms of cores +
  ~0 experts":
  (a) GDN over a tree, chunk form: for every layer, read the block-start state S once; for node j with path P(j) the
      output is q_j . S (decayed by the path's cumulative gate) + sum over ancestors i in P(j) of (q_j . k_i) (decay
      i->j) v_i, i.e. a 128 x 128 masked attention with the ANCESTOR mask (the same mask the tree attention already
      builds) and the gated-delta correction (the WY form, TreeWY's math). The next block's state is S' = decay S +
      the rank-r update along the ACCEPTED path only, written once. No per-node state, no fork copies. Bytes: 226 MB
      read + write per check (as at 32 rows). MACs: 36 layers x 48 heads x 128^2 x 128 = 3.6 G.
  (b) off-spine rows routed inside the spine's loaded union: zero expert bytes, zero decode; their pairs fill the
      16-row MMA tiles' padding (5.2 -> <= 16 pairs per segment at no tile cost); the spine's ~32 rows route freely
      (C6 tier rule as today). The 125B's token at an off-spine row then comes from 9 of its 10 experts on average
      [Fable B]; acceptance stays the 125B's own argmax under the lossy mode's gate.
- **Price [P, 100 % of peaks, C6 + int8, draft 5.2, overhead 1.5].** Per extra row: dense 0.13-0.16 ms, experts ~0,
  cores ~0.01, lm_head only on rows with children. K64 check 27.6 ms (kept 20.77 [K]) -> 606 tok/s; K96 32.2 (kept
  ~22.5 [P between K64 and K128]) -> 579; K128 36.8 (23.70 [K]) -> 545. At 85 %: 530 / 505 / 474.
- **What it reuses.** The tree attention's ancestor mask (built once per check, vectorized: treek_price's O(T^2) loop
  warning), the 32-row one-call kernel unchanged (its tiles are already 16 rows), the C6 tier rule for the spine.
- **Screens (<= 10 min each, Leap's 4-layer slice, microbench, no engine change):**
  - S1: chunked tree GDN at 64 / 128 rows vs the sequential recurrence at 32 rows: PASS <= 3 ms at 128 rows for 36
    layers (scaled from the slice), KILL > 8; plus max |delta| vs the sequential form on 24 real blocks (the lossy
    gate's input: expect ~1e-3 relative, like a chunk kernel).
  - S2: the one-call at 64 / 128 rows with the off-spine rows' routing clamped to the 32-row union: time per call vs
    32 rows. PASS <= 1.10x (the padding absorbs the rows), KILL > 1.4x (the segments spill into extra tiles).
  - S3 (CPU, the 127-node dumps + the decision map's scores, Fable B's frame): the share of off-spine rows whose 125B
    argmax changes when 1 of 10 experts is dropped; the lossy gate's F_line (0.00 per 1k at >= 2 nat flips) decides.
- **Quality risk.** Lossy only (fp reorder in the GDN chunk; 9-of-10 experts on off-spine rows). Exact mode keeps the
  sequential cores and so stays at 32 rows: this is a lossy-line lever, which is where 550 is asked.
- **Kills it.** S1 > 8 ms (the chunk form on this GPU is not matrix-unit friendly at d 128, like the parked 32-row
  chunked GDN that came out 3x slower as built); S3 flipping > the F_line (then off-spine rows need their own union
  and the experts grow x1.4-1.9 after all: K64 then ~550 at 100 %, ~480 at 85 %).

## 3. Idea 2: cut the dense MACs per row, the only term left (from 0.16 to ~0.10 ms) [P]
- **lm_head only where it is read:** 0.64 of 3.61 G MAC per row is the vocabulary projection; a tree row's logits are
  needed only to verify its CHILDREN. Leaves (40-50 % of a 128-node tree) and rows whose children are all pruned
  skip it: -8 % of dense MACs at K128, exact.
- **int8 everywhere the dense is:** today's dn32q runs int8 x int4 only on the FAST dense shapes; the GDN projections
  (2.08 G of the 3.61) are the bulk; at the measured 1.55-1.73x they are the whole int8 gain and are already in the
  C6 + int8 bound. Nothing cheaper than int8 is measured on this chip's matrix units; I do not price int4 activations.
- **Share the prefix, not the row:** sibling rows share their attention keys (already, via the mask) and nothing else;
  the GDN projections and the shared expert are per token. There is no structural sharing of dense MACs between
  rows that are different tokens. (Fable B's lens; the arithmetic closes it from my side.)
- **Drop rows by the head's own edge gap before the check** (EARLYDEATH stage 2: the head alone drops 31.4 % of wrong
  children at <= 2 % false drops): that is K* optimisation, not a cheaper row, but it is the one free -30 % of
  off-spine rows: K128's dense MACs at K~90's price with K128's kept [P, needs the dumps' replay].
- Together: the marginal row at ~0.10-0.11 ms; 96 rows ~10 ms; K128 at 100 %: ~565 tok/s; still not 5 ms.

## 4. The units, for the record (why not the ANE / CPU)
| unit | measured | a 125B dense row | verdict |
|---|---|---:|---|
| GPU matrix units | 14.0 TMAC/s bf16, x1.55-1.73 int8 [K] | 0.26 / 0.16 ms | the row |
| Neural Engine | 0.295 GB int8 package, 16 rows, 10 ms (ANE-2b); 4-bit LUT expert work +16-25 % beside the GPU only while the GPU is under its byte peak (WHOLE-CHIP), and the GPU->ANE->GPU handoff is 78-147 us per layer (WHOLE-CHIP-2 KILL) | ~7 ms | not a row engine; at most a drafter that waits for the check (MEND-b / Mirror-SD, KILL) |
| CPU (AMX via MLX fp32 / NEON) | 62 ms for the ANE's 12.4 ms call (M-19); NEON expert path BETWEEN +16 % only beside a non-saturated GPU; 70-150 us per mid-pass sync | ~35 ms | no |
| DRAM | 280 GB/s, shared by all three | | the check is at it; a second unit beside the GPU slows the GPU (bw4 +22.8 %) |

## 5. One line
On the M5 Pro a check row is 3.6 G dense MACs: 0.16 ms at the measured int8 matrix peak (0.13 for a leaf), 7 ms on
the Neural Engine, ~35 on the CPU, so 96 rows cost >= 12.5 ms, not 5, and kept 24 "for the price of 32 rows" is out
by arithmetic. What IS free for a row is everything else: the experts (off-spine rows routed inside the loaded union
ride in the 16-row MMA tiles' 2/3 padding, zero bytes) and the GDN cores (the chunked tree form, admissible in LOSSY
mode, ~2 ms at 128 rows vs Leap's 35-46). With both, the marginal row is ~0.13 ms and the optimum is K64-96 at ~580-610
tok/s at 100 % of the peaks (530 at 85 %); screens S1 (chunked tree GDN on the slice, PASS <= 3 ms) and S2 (the
one-call at 64-128 rows with clamped routing, PASS <= 1.10x) are 10-minute microbenches for Leap.
