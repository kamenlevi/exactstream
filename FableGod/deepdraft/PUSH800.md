# PUSH800: 775-800 tok/s lossy, measured, at quality, by Sat 21 Nov (orchestrator, Mon 16 Nov 17:4x)

> (a quote from the project owner, kept private)
775, but not less, in lossy mode, which passes the quality and everything like that." Exact: "400 and something".
He also said: "if you want, don't run any big tests at all. Only go through ideas and then build on their projections"
(his guess, not a rule). The number must still be MEASURED (whole decode, known benchmarks) to count.

## 1. The frame (measured parts; HE+ code; one 32-row check per block)
tok/s = kept / (check + draft + overhead), ms.

> **CORRECTED by NewIdeas, Mon 17:5x (the number check asked for): this section used the OLD expert sizes (3.99 / 3.63
> GB). Since 13:5x the real pieces are 2.785 MB (4-bit) / 2.064 MB (MSB-3), so C5 experts are 5.05 GB and C6 4.60 GB
> per 32-row check (Leap's data/week98/check25_bound_real.txt).** The corrected numbers are below; the old ones are in
> brackets. Every conclusion gets harder, none gets easier.
- Check bound at 100 % of every peak (Leap, check25_bound_real.txt): dense 8.5 ms bf16 / 7.6 int8 (int8 is byte-bound
  on 2.13 GB, so not 4.9) / 6.4 with V4 + experts 18.0 ms (C5 5.05 GB at 280 GB/s) / 16.4 (C6 4.60 GB) + cores 1.2 =
  **27.7 (C5) / 25.2 (C6 + int8) / 24.0 (C6 + V4 + int8)** [were 23.9 / 21.7 / 20.6].
- Today: best measured lossy check 56.4 ms (c25 F32c6i) = ~45 % of its C6 + int8 bound [was ~42 %]; draft 19.1 ms
  (tree-32, 177.6 MB head per level); overhead ~4; kept 16.91 (tree-32 + close-call d1) -> ~215-227 tok/s. Exact ~144
  (tree-16, measured engine).
- **At today's kept (16.91), even 100 % of every peak with ZERO draft and overhead gives 610 (C5) / 671 (C6 + int8) /
  705 (C6 + V4 + int8)** [were 707 / 779 / 821]. **No byte plan reaches 775 at today's kept even at 100 %.** With
  realistic best engineering (90 %, draft 3, overhead 1.5): ~480-545 [was ~560-620].
- **So kept is the binding wall for 775.** Kept needed per 32-row check (overhead 1.5):

  | check bytes | 85 %, draft 3 | 90 %, draft 3 | 85 %, draft 7.4 (QAT head [P]) | 90 %, draft 7.4 |
  |---|---|---|---|---|
  | C6 + int8 (25.2) | 775: 26.5 / 800: 27.3 | 25.2 / 26.0 | 29.9 / 30.8 | 28.6 / 29.5 |
  | C6 + V4 + int8 (24.0) | 25.4 / 26.2 | 24.2 / 24.9 | 28.8 / 29.7 | 27.6 / 28.5 |

  [The old table read 21.2-24.0.] That is kept **x1.43-1.62 vs today with a 3 ms draft, x1.63-1.82 with the QAT head's
  projected 7.4 ms** [was x1.25-1.42], at the same time as the engine at 85-90 %. A 32-row chain's kept tops out near
  ~33, so 28-31 means ~99 %+ effective acceptance per position (lossy rules included).
- Exact 400: the exact 32-row floor is 57.9 ms (code, real sizes); at kept 16.26 that is 281 tok/s at 100 % with free
  drafting. Exact 400+ needs kept >= ~26 AND the floor. (The old "450 exact" is the ceiling math at a 97.8-98 %
  drafter, not a measurement.)

## 2. The four multipliers, all needed together
1. **ENGINE** (no quality risk; the biggest): whole-check efficiency ~42 % -> >= 85 % of the bound (x2.0). Parts:
   the lossy expert one-call at the exact path's ~280-290 GB/s (Leap's lead), qmm32n + int8 dense math, chunked GDN,
   router / 1c glue fusion (the 1c path's CPU encode ~28 ms, 4,022 dispatches per check, 52 % MLX glue).
2. **DRAFT** 19.1 -> <= 3 ms (x6): the QAT ~26 MB head (S13 tonight) + a fused level near f 0.056 (BET 7 got 0.269).
   [NewIdeas 17:5x: at bet 7's MEASURED fixed cost f 0.269 ms per level, the QAT head projects ~7.4 ms (ladder v3,
   PLAN900), not 3. A draft of <= 3 ms needs f <= ~0.1 ms per level, which no measured kernel reaches yet; see the
   draft-7.4 columns above.]
   [MTP Head 18:2x, from MoonShots' measured ~95 MB per QAT level: tree-32 draft ~10 ms with the QAT head alone (kept
   x0.97 today; >= 0.99 if S16 passes), and ~6.4 ms with FAR's 8K screen (kept x0.963). A CHAIN-32 draft costs 31 levels
   (~9-11 ms at the screened head), against ~21 for tree-32. S13 FAILED on its recipe (constant lr); S16, the annealed
   QAT, decides Wednesday's 125B recipe.]
3. **BYTES:** C6 + int8 dense (G1, Tue); a V4-class cut only if DENSE-3 passes; lambda 2e-2; RES; early death (Wed).
   [NewIdeas 18:2x, the CPU frontier (research/week98/tree_frontier.txt + tree_frontier_t16.txt, exact self-check):
   AT QUALITY the acceptance frontier is FLAT. #13's replay figures below are the WHOLESALE form (CD16), a recorded
   quality KILL (TRIED CD16-DROP, 11-06). The safe guarded form (T16) adds x1.00-1.02. #8 is already in the shipped
   base (comment rule C). R1(2/3) / TYP / TOP2 reach x1.02-1.05 on trees, with unknown quality on the 125B (d2 is
   gated tonight). So the kept multiplier has to come from the DRAFTER (Goal A), not from acceptance rules.]
4. **KEPT x1.43-1.62 in lossy mode (x1.63-1.82 at a 7.4 ms draft; corrected 17:5x, was x1.25-1.42):** close-call d2; span-conditional close-call (#8); inert comment / docstring
   acceptance (#13: replay x1.17-1.30 HE+, x1.63-1.79 MBPP, high quality risk); early death's bigger trees.
   ~~#8 / #13 are ON~~ **WITHDRAWN 18:2x (orchestrator): #13 wholesale = the recorded quality KILL CD16-DROP (11-06);
> (a quote from the project owner, kept private)
   are still written in full and code rows stay strictly checked; EvalSuite's gates decide, and any public number names
   the acceptance rule.

## 3. THINK_1116b (60-min desk sprint, Mon ~17:50 -> ~19:00): beat the frame
Question: find a lever NOT on the kill list below that gives one of:
- (a) kept x1.3+ per 32-row check in LOSSY mode at quality;
- (b) a x1.3+ cut in check cost per kept token beyond C6 + int8 at 85 % efficiency;
- (c) a different decode shape that beats this frame for ONE stream (multi-stream is information only).

Each idea: the mechanism; the number it moves (from our measured parts); a <= 10-min CPU screen on the existing dumps
(the real 127-node trees, the map windows, the bank) or a desk bound; the quality risk; what kills it.
Borrow freely from other fields (CPU runahead / value prediction, video codecs, compression, control, biology);
Kamen asked for that. Kamen's examples are options, not rules.

Kill list (do not re-propose without a NEW mechanism; records in ~/TRIED/TRIED.md, WEEK98.md, PLAN900.md):
SIB sibling states (KILL 15:2x); span tree / deep runs from copies; draft death / wide-shallow; keyed widening;
t30c (more budget / data); megakernel grid barriers; a second GPU stream (kappa 0.93); 3-bit dense by RTN / AWQ (V4
FAILED); comment-free decoding (Kamen: cheating); any smaller verifier model (never).

Lenses (one each; any can cross over):
- Lunatic: the limits. Where is the slack in kept (forced tokens, entropy of code positions, what bounds saturate)?
- ZERO: first principles. The per-kept-token cost from zero on the M5 Pro; what a Saturday-sized engine keeps / cuts.
- FAR: drafting. A drafting shape that lifts lossy kept x1.3 (not on the kill list).
- Fable C: audit + invent. Is any term of section 1 priced wrong (too pessimistic or too optimistic)? Then one invention.
- Fable B: the dense pass. 8.5 ms of dense compute at 32 rows: cut it (int8 / int4 activations, row pruning, sharing).

Write to your own folder as IDEAS_1116b_<topic>.md. Then send ONE line to NewIdeas (`peer send ojox-28`) and the
orchestrator: the best idea, its number, its screen. NewIdeas screens the best ones tonight (CPU, <= 10 min each).

## Saturday ladder v4 (NewIdeas, Mon 20:5x; ~/NewIdeas/research/week98/PLAN900.md, 6d9bee2)
- At quality, HE+ lossy: kept 16.3-16.6 (the frontier is flat; 8K-screen head loss x0.963-0.98). Check ~40 ms (Leap's
  modules PASS) / ~32 (stretch). Draft 8.8 ms (8-bit Linears free per S17 + the 8K screen, f 0.269) / 5.2 (f 0.10).
  Overhead 4 / 1.5.
- **Result: ~310-430 tok/s (today ~213).** 775 needs kept ~25-31 at these check times, which only a better drafter
  gives.
