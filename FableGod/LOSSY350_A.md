# LOSSY350-A: the cost of one expert group once lossy is allowed (Fable A, Thu 5 Nov 2026; v1 16:29, v2 16:45, v3 16:55, v4 17:0x, v5 18:4x EET)

Brief: the orchestrator's MISSION LOSSY350-A (Kamen ~15:55: 350 tok/s LOSSY at 45 GB; real thinking, never-tried ideas or a
broken assumption; kill fast; every idea gets a number or an honest [G]). My wall: the cost of each expert group.
Marks: [K] measured (source named), [P] arithmetic or replay on [K], [G] judgement. Gains are halved where it says so.
What ran here: eight CPU replays (numpy, `python -I`, one process, 4 threads, nice 15, < 1 GB, 10-99 s each), scripts and
outputs in `~/FableGod/lossy350a/`. Inputs: Fit12's Q2 capture (`~/Fit12/data/q2/x_L{2,12,24,36,46}.npy`, the real MoE
inputs, 8,336 tokens per layer) and the real 4-bit weights of Leap's L1-16 experts (8 per layer, seed 0) plus the five
routers, read by numpy mmap from `~/BigMoE/models/bare` (118 MB touched; the orchestrator's 16:1x ruling for this mission:
read-only, no MLX, no GPU, records3 untouched). Cross-check: my loader reproduces Leap's act16 MSB-3 errors to four digits
(gate 0.0168, up 0.0223). No GPU, no downloads, nothing written outside `~/FableGod/`.
Metric: whole-expert OUTPUT error, sum ||y~ - y||^2 / sum ||y||^2 against the 4-bit expert, on the rows the layer's own
router sends to that expert (8,337 routed pairs; everything that depends on a fitted basis on the 2,431 held-out pairs).
The unit is "x MSB-3": the same error for the expert at the 3-bit tier that REF2-T already serves on ranks 2-9
(0.064-0.066 whole expert [K here]; gate-only 0.017, up-only 0.017, down-only 0.031).

## 0. Verdict
1. **Under lossy, a group gets cheaper only by touching fewer of its weights.** Fewer bits do not buy time (2-bit x1.22
   [K L1-30]) because the op set and the bytes are each ~7 us and both are per weight. Touching half the weights halves both.
2. **Half of an expert's weights can go untouched at about one 3-bit tier of error [K].** Rule: per row, keep the largest
   inputs for gate / up and the largest units for down (magnitude sparsity, TEAL's rule), after removing the layer's 32
   dominant input directions, whose response is exact and precomputed. Measured frontier (held-out rows):

   | weights touched | 0.84 | 0.72 | 0.63 | 0.59 | 0.55 | 0.51 | 0.47 | 0.43 |
   |---|---|---|---|---|---|---|---|---|
   | whole-expert error, x MSB-3 | 0.11 | 0.23 | 0.48 | 0.65-0.71 | 0.92 | 1.14-1.22 | 1.64 | 2.10 |

   Per weight saved this is the cheapest approximation of an expert we have measured: ~2x less block error than
   dropping the low ranks and ~3x less than dropping a bit [P, section 1].
3. **The kernel can cash it per (row, expert) pair, not per group [P].** Each row keeps its own half, and the union
   of two rows' halves is 45 of 64, so the lever lives on the pair path (the hybrid's ALU side, bytes-bound at 7.1 us per
   pair [K 3c fit]) and on single-row groups (one matrix-unit op per quant group instead of two). It needs the members of
   a quant group stored K-major: a bit shuffle of today's record. Per group, against the hybrid's ~7.8 us:
   **x0.93 / 0.86 / 0.78 / 0.70 / 0.62 at 0.84 / 0.72 / 0.59 / 0.51 / 0.43 of the weights**.
4. **On code it fits, and it spends the same budget the split form spends [P].** Two readings built on measured map
   arms price it (section 1): SWEEP v1.1's KL model (all-row flips) and the map's uniform neighbour arms read by row
   class (non-tie flips, close to linear in the block's error energy). They agree within ~1.5x. On code, half experts
   on everything add +0.4-0.6 points (room 0.95 all-row, 0.66 non-tie) and the mild rank plan +0.1-0.2. On chat only
   the mildest point (0.84 of the weights) fits, thinly on multi; everything stronger is over both chat bars. Per % of
   expert time the mild forms cost about what cutting tiles costs on code (0.006-0.010 non-tie points per %, tiles
   0.007 [K WALL_1105]); the hard forms cost twice that.
5. **On the check: +2-8 % tok/s raw on code alone, ~+3 % on top of the split form, halved +1-4 % [P].** Experts are
   22-27 ms of today's 90-95 ms [K AUDIT_1105 F2]; alone this takes 1.7-9 ms. Next to split 0.9 (non-tie 0.78 of the
   0.87 bar [K WALL_1105]) nothing fits; next to split 0.99 (0.51) the mild rank plan fits (+0.11-0.16) and plan B
   thinly (+0.21-0.30).
6. **The ceiling of this wall is x1.35 on tok/s at zero expert cost,** so the 3x is not hiding in the cost of a group.
   For the 350 arithmetic: a 30 ms check can afford ~8 ms of experts; the split form alone lands at 10-13 ms, the split
   form and half experts together at 7-9 ms, if code's budget holds both (it holds the split form plus the mild plan).
7. **Twelve other forms are dead by a number** (section 3), including my first lead, a prefix of K in the layer's rotated
   input basis. Its support in TRIED (L1-16b, line 337) was measured in sample; held out it is 6x worse.

## 1. The single best bet: HALF-K, magnitude-gated half experts on the pair path
Mechanism, five lines:
- Per (row, layer), once: c = V^T x for the layer's 32 dominant input directions (V from a capture; 82 K MACs), x_perp = x - V c.
- Per 64-wide quant group of gate / up: keep the c_x members with the largest |x_perp| (48, 40 or 32 of 64). The row's ten
  experts share this list. Each expert adds its precomputed response to c (two [640 x 32] matrices, +0.8 % of its weights).
- Per 64-wide quant group of down: keep the c_h units with the largest |h| (40 or 32 of 64), known after gate / up.
- Level by rank: the router's weights (0.20, 0.15, 0.12 ... 0.054 [K `rank_energy.out`]) put 56 % of the block's energy in ranks 0-1,
  so ranks 0-1 take the mild points and ranks 5-9 the hard ones.
- Layout: members of a quant group stored K-major (12 B per member per 32 outputs), so a lane loads only kept members.
  The exact kernel sees the same two op operands per quant group, so the same store serves both tracks [P, to be built].

The numbers [K `arms.out` deflated, `arms_plain.out` plain; held-out pairs; ranks 0-1 and 2-9 agree within 0.03]:

| arm (members kept of 64: inputs x, units h) | weights touched | error x MSB-3, deflated | plain | deflated, on MSB-3 weights (total vs 4-bit) |
|---|---|---|---|---|
| x48 | 0.84 | 0.11 | 0.17 | |
| x48 h40 | 0.72 | 0.23 | 0.29 | |
| x40 h40 | 0.63 | 0.48 | 0.68 | |
| **x40 h32** (threshold form) | **0.59** | **0.71 (0.65)** | 0.91 (0.83) | 1.65 |
| x36 h32 | 0.55 | 0.92 | | |
| **x32 h32** (threshold form) | **0.51** | **1.22 (1.14)** | 1.70 (1.59) | 2.12 |
| x32 h24 | 0.47 | 1.69 | 2.15 | |
| x28 h24 | 0.43 | 2.10 | | |
- Down alone at half (h32): 0.36 for a sixth of the weights. The cheapest first step.
- Deflation: -30 % error at equal cost (x32: 1.38 -> 0.88); 256 directions 0.66 for +6.7 % weights. Errors add with the tier.
- By layer the same arm is not flat: x32 h32 costs 1.09 / 1.47 / 1.58 / 1.41 / 1.11 x that layer's MSB-3 at layers 2 / 12 / 24 /
  36 / 46 [K `own_dirs.out`, 15 experts]; the pooled 1.14-1.22 leans on layer 46's busy experts. The middle is ~30 % dearer.
- Against the other ways to save weights [P on K], in the routed block's relative error energy (ranks weighted by their
  measured energy shares): top-8 routing saves 20 % of the weights for 0.032; rank plan A saves 39 % for 0.024; plan B
  saves 48 % for 0.046; x32 h32 on everything saves 49 % for 0.078; one bit off every expert (MSB-3) saves 12.5 % of the
  bytes and no time for 0.066. Dropping a whole expert is 15.2 x MSB-3 on it (1 / 0.0657).

Cost model [P]: pair path = (weights touched) x 7.1 us + ~0.3 us of selection and gather [G]. Group mix in the live FAST
regime [G]: 65 % single, 22 % double, 13 % three or more; doubles run as two pairs when that beats 9.2 us; larger groups
stay on the matrix unit, exact. Saving against the hybrid's 7.84 us per group: 7 / 14 / 19 / 22 / 30 / 38 % at
0.84 / 0.72 / 0.63 / 0.59 / 0.51 / 0.43 of the weights = 1.7 / 3.4 / 4.7 / 5.4 / 7.4 / 9.3 ms of a 24.5 ms expert term.

Predicted quality [P], two models, each built on measured map arms; added flips over today's preset:
- **S = SWEEP v1.1's KL model** (`~/NewIdeas/research/SWEEP.md` 83-105; `data/ledger/map_kl_fit.txt`): a store at error ratio
  r vs MSB-3 costs KL r x 0.0020 / 0.0113 / 0.0128 on the sure rows' ranks 0-1 and (r - 1) x 0.0011 / 0.0065 / 0.0089 on
  ranks 2-9 (the 239-slot measurement; a floor at the 45 GB pool); base 0.0013 / 0.0093 / 0.0101; all-row flips =
  -0.078 + 15.68 sqrt(KL) / 0.420 + 39.04 sqrt(KL) / -0.173 + 36.17 sqrt(KL) (ten arms, rms 0.07 / 0.37 / 0.15).
- **C = the map's uniform neighbour arms, read by row class** (Fable C's convention, `fablec/WALL_1105.md`; computed in
  `lossy350a/build_1105/predict.py`): non-tie flips are close to linear in the routed block's relative error energy.
  Slope per unit, code / IFEval / multi: 4.6 / 49.5 / 43.3 (every expert at MSB-3, `t3_all - t4_all`), 6.6 / 59.3 / 48.2
  (`top_k8`), 7.3 / 54.3 / 58.4 (`topk5`) [K]. An arm at e x MSB-3 on everything therefore costs e x (0.30-0.43) /
  (3.3-3.9) / (2.9-3.2) non-tie points. (v4 anchored this column on oea_k8, whose error is below plain top-8's; that
  made the mild arms look 2-6x cheaper on chat than they are.)

| arm | error x MSB-3 | block relative error | S: code / IFEval / multi, all-row | C: code / IFEval / multi, non-tie | expert ms saved |
|---|---|---|---|---|---|
| x48 plain | 0.17 | 0.106 | +0.11 / +0.56 / +0.61 | +0.05-0.07 / +0.55-0.66 / +0.48-0.54 | 1.7 |
| x40 plain | 0.56 | 0.192 | +0.30 / +1.66 / +1.77 | +0.17-0.24 / +1.8-2.2 / +1.6-1.8 | 3.0 |
| x32 plain | 1.38 | 0.301 | +0.61 / +3.4 / +3.6 | +0.41-0.60 / +4.5-5.4 / +3.9-4.4 | 4.2 |
| x48 deflated | 0.11 | 0.085 | +0.07 / +0.39 / +0.41 | +0.03-0.05 / +0.36-0.43 / +0.31-0.35 | 1.7 |
| x40 h32 deflated | 0.68 | 0.211 | +0.35 / +1.95 / +2.07 | +0.20-0.29 / +2.2-2.6 / +1.9-2.2 | 5.4 |
| x32 h32 deflated | 1.18 | 0.278 | +0.54 / +3.0 / +3.2 | +0.35-0.51 / +3.8-4.6 / +3.4-3.7 | 7.4 |
| rank plan A (ranks 0-1 x48, 2-4 x40 h40, 5-9 x32 h32) | 0.11 / 0.48 / 1.18 | 0.156 | +0.19 / +1.03 / +1.19 | +0.11-0.16 / +1.2-1.4 / +1.1-1.2 | 4.9 |
| rank plan B (ranks 0-1 x48 h40, 2-4 x36 h32, 5-9 x28 h24) | 0.23 / 0.92 / 2.10 | 0.214 | +0.32 / +1.80 / +2.03 | +0.21-0.30 / +2.3-2.7 / +2.0-2.2 | 7.1 |
| room above today's preset | | | 0.95 / 1.33 / 0.43 | 0.66 / 1.03 / 0.72 [K WALL_1105: bar 0.87 / 2.22 / 1.63, preset 0.21 / 1.19 / 0.91] | |
- Code: every arm fits in both models; the split forms sit at +0.30 (0.99) and +0.57 (0.9) non-tie on the same scale.
- Chat: only x48 fits, thinly on multi (C: +0.5-0.7 non-tie against 1.03 / 0.72; S: +0.4-0.6 all-row against 1.33 /
  0.43). Plan A and everything stronger are over on chat in both readings.
- The rank weights are measured [K `rank_energy.out`]: ranks 0-1 carry 55.9 % of the routed block's energy, ranks 2-4
  30.0 %, ranks 5-9 14.1 % (router weights 0.198 ... 0.054; output norms fall 35-40 % from rank 0 to rank 9).

### The tests (each <= 10 min of GPU, after Fri 00:15; lines and predictions written now)
**T1, quality, the decisive unknown: decision-map arms on the 36 windows, stacked on today's preset (qom_0_live_bq).**
- The hook is a mask, not a store. In `decision_map/dm_common.py` `moe_inner` (line 310) the experts are called as
  `y = self.switch_mlp(x, inds)`; the arm calls `self.switch_mlp(x * m, inds)` with
  `s = abs(x).reshape(..., 40, 64); m = (s >= sort(s, -1)[..., 64 - c][..., None]).reshape(x.shape)`. The router and the
  shared expert keep the full x. The >= 32-row path needs the same mask where the prefill call reads x.
- Arms, first screen (x only, plain, every rank, no calibration): x48, x40, x32. Whole-expert error 0.17 / 0.56 / 1.38 [K].
- Predictions [P]: the table above, rows x48 / x40 / x32 plain: code +0.11 / +0.30 / +0.61 all-row (S), +0.05-0.07 /
  +0.17-0.24 / +0.42-0.60 non-tie (C); chat as tabled.
- **Built 18:4x:** `~/FableGod/lossy350a/build_1105/` holds the patch, a CPU self-test (68 checks PASS) and INSTALL.md
  with these predictions and lines. The mask sits on the switch's two entry points, not on line 310 alone: the map
  prefills in 512-row chunks, so nearly every row takes the >= 32-row path, which never reaches line 310.
- Read each arm by row class (`fablec/wall_class_flips.py`). PASS, in WALL_1105's convention: code all-row <= 1.34 % AND
  non-tie <= 0.87 %. KILL: over either. Chat is recorded as the control; x48 on chat is the arm that says which model
  holds (S: multi +0.6 all-row; C: +0.5 non-tie).
- The lever is dead if standalone x32 costs more than 1.03 % non-tie on code (topk5's own cost) or x48 more than 0.21 %
  (top_k8's own cost); stacked: x32 adds more than +1.2 all-row code points, or x48 more than +0.25. Then sparsity
  error is not priced like the errors both readings were built on.
- Second screen if x40 passes on code: the h mask (needs the GLU's three calls separately) and the rank plan
  (two `switch_mlp` calls on rank-sorted `inds`).
- My odds [G]: x40 code PASS 75 %; x32 code PASS 55 %; x48 inside both chat lines 40 %.

**T2, kernel, Leap: the pair path on layer 20's real records re-laid K-major, real routing.**
- Measure us per pair at 32 and 40 of 64 against 3c's 7.1 us per pair; bits against a numpy reference of the same masks.
  Two forms: ALU lanes over kept members, and one matmul2d op per quant group for single-row groups.
- PASS: <= 4.4 us at 32 and <= 5.3 at 40. KILL: > 6.0 at 32 (the gather ate the saving; each lane's code load is no
  longer amortized over 64 members, ~+27 % ALU work per weight [P]).
- Second line: NAX6 on the K-major layout bitwise == NAX6-N and within 3 % of its GB/s. If not, the layout is a FAST-only copy.

## 2. Ranked ideas (numbers first)
| # | idea | number | status |
|---|---|---|---|
| 1 | HALF-K at x40 h32 (deflated) | 0.59 of the weights at 0.65-0.71 x MSB-3 [K]; group cost x0.78 [P]; code +0.2-0.35 points, 5.4 ms [P] | best bet; T1 then T2 |
| 2 | HALF-K at x32 h32 (deflated) | 0.51 at 1.14-1.22 [K]; group cost x0.70; code +0.4-0.55 points, 7.4 ms [P] | same tests |
| 3 | the rank plan (mild on ranks 0-1, hard on 5-9) | plan A: 39 % of the weights saved for +0.11-0.19 code points, 4.9 ms, the cheapest point; plan B: 48 % for +0.21-0.32, 7.1 ms. Top-8 routing saves 20 % of the weights at a block error (0.18) between the two plans' (0.16, 0.21) [P on K] | second screen of T1; the form that fits beside the split form |
| 4 | x48 on every rank | 0.84 at 0.11-0.17 [K]; code +0.03-0.11; chat +0.3-0.7 in both readings [P]; 1.7 ms | first arm of T1 (the model check, both domains) |
| 5 | deflation of the dominant input directions | -30 % error at equal cost for +0.8 % weights [K]; the one "partial product reused across experts" that works | part of 1-3; needs V for 48 layers (one capture) |
| 6 | down at half alone (h32) | 0.83 at 0.36 [K] | the first kernel step of 1 |

## 3. Killed this hour, each by its number
| # | idea | number | source |
|---|---|---|---|
| K1 | K-prefix in the layer's rotated input basis (static order, per-row adaptive depth) | held out: 50 % of K costs 3.97 x MSB-3, 75 % costs 1.76; per matrix at MSB-3's own error only 2.5-4 % of K can go. In sample it looked like 67 % of K at 1.0 | `rotk_x.out`, `touch_less.out` |
| K2 | static unit order (prefix of the 640 units) | 75 % of the units costs 1.38; 50 % costs 3.85 | `touch_less.out` |
| K3 | gate-first unit selection (skip up and down where silu(g) is small; CATS's rule) | 50 % of the units costs 1.44 against the oracle's 0.33; the gate is soft: pre-activations have std 0.5 (74 % negative), so silu(g) never switches a unit off and \|h\| is set by u as much as by g | same |
| K4 | a cheap sketch picks the units (the response to 32-256 dominant directions) | 50 % of the units costs 3.2-4.9; the sketch sees only half the input energy. Hashing does no better: two near-orthogonal 2560-vectors need ~2560 measurements [P] | `sketch_units.out` |
| K5 | cascade and stacks built on K3 | 1.6-4.0 at 0.34-0.67 of the weights; dominated by HALF-K everywhere | `touch_less.out` |
| K6 | one shared member set for a 2-row group (to keep one op) | 2.98 against 1.56 for per-row sets | same |
| K7 | reuse an expert's last result and compute on the difference to that row | the difference has 1.35-1.66x the row's own energy after deflation (cos 0.20-0.34), adjacent rows the same | `delta_rows.out` |
| K8 | product-quantized inputs with per-expert lookup tables | after deflation, 1 bit per input dim (16 codewords per 4 dims) leaves 16 % error (10x MSB-3's gate error) at 0.25 of the weights touched and 4x the storage; 2 bits per dim leaves 4 % (2.4x) at 0.5 touched and 8x the storage, or 0.25 touched and 64x [P, rate-distortion bound] | arithmetic |
| K9 | Kronecker tile reuse; sampled columns; two codes in one B element; a low-rank term for the dropped inputs (R-Sparse's form) | 11x the MACs at 256 terms; error 113 % at half of K; the cross terms cannot be separated; rank 64 removes ~22 % of the dropped error for +12.5 % weights where 12.5 % more members remove 60 % (x32 0.88 -> x40 0.35 [K]) [P] | arithmetic |
| K10 | a 1-bit plane for the small inputs | 0.5 x MSB-3 at the bytes where plain sparsity gives 0.35 [P] | arithmetic |
| K11 | deflate each expert's OWN input directions (its routed rows' cluster) on top of the layer's 32 | its own top 4 / 16 / 48 directions remove 2.6 / 6.3 / 11.7 % of the remaining energy held out; x32 h32 1.19 -> 1.11 at +2.3 % weights. One expert's inputs are as high-dimensional as the layer's | `own_dirs.out` |
| K12 | hide the experts under the dense pass: low ranks half a layer late (B12LL's rule, a PASS at 12 GB) run beside the next mixer; or two half-batches staggered by half a layer (exact, by causality) | the hiding needs two GPU streams to overlap, and they do not: 0.06 ms per cycle (D-COMMIT D [K TRIED 831]), no overlap when busy (runahead [K TRIED 130]); the stagger pays the mixer twice, 1.4 ms per layer against 1.2 [P] | TRIED + arithmetic |
Already dead in TRIED and not re-opened: a shared base expert (L1-16a), per-expert low rank (L1-16d, 6.4x), a shared output
subspace for down (0.27-0.65), 2-bit experts (B10 family, LEVER6), fewer bits for time (L1-30).

## 4. TRIED and novelty positions
- **Correction to TRIED 337 (L1-16b) [K]:** "r 1536: gate / up 0.029 / 0.038" was computed with the subspace and the error on
  the same 8,336 tokens. With the basis from 5,836 tokens and the error on the other 2,500, the energy outside the top 1536
  directions is 0.13-0.19, not 0.01-0.03; outside the top 1024, 0.22-0.31, not 0.03-0.08 (five layers, `rotk_x.out`).
  Past the first ~100-250 directions the held-out spectrum is nearly flat and the coordinates are Gaussian-like (the top
  half by magnitude holds 93.3 % of the energy; a Gaussian gives 92.8 %).
- HALF-K's selection rule is published: TEAL (magnitude sparsity on the inputs of every matrix), CATS (the gate-first
  form, K3 here), R-Sparse (sparse inputs plus a low-rank term, K9 here). In TRIED, in-expert sparsity is "U" (AUDIT_1105
  item 10) and Fit12 is testing it now; my replay overlaps theirs on the plain rule, and the numbers are theirs to use.
- What may be new, for MoonShots to check: the deflated form with one basis per MoE layer shared by all experts; the
  K-major-within-quant-group record that serves the exact kernel and the sparse path from the same bits; one matrix-unit
  op per quant group as the sparsity level; the level keyed on router rank.
- Build cost, stated once [G]: a store re-layout (a bit shuffle, like this morning's conversion), two new kernels and a
  re-qualification of the exact kernel on the new layout, for +1-4 % halved on code. The split form is one condition for
  +7-9 % halved. Build HALF-K only if T1 shows code room beyond what the split form uses.
- For Fable B's wall, one line [P]: the verifier's lm_head is ~1 ms of bytes per check (248 K x 2560 at 4 bits); a head
  restricted to the draft token plus the drafter's own top candidates removes it, as a lossy acceptance rule.

## 5. Reproducibility
`~/FableGod/lossy350a/`: `rotk_x.py` (inputs only); `touch_less.py` (all routed pairs); `sketch_units.py`, `arms.py`,
`arms_plain.py`, `own_dirs.py` (held-out pairs); `rank_energy.py` (all routed pairs); `delta_rows.py` (inputs + routers); each writes `<name>.out`. Run:
`OMP_NUM_THREADS=4 nice -n 15 ~/BigMoE/.venv/bin/python -I <script>`. Limits: one capture corpus, 40 experts, five layers;
whole-expert error is a proxy for flips until T1; every kernel time and every flip number is [P].
