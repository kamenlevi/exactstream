# Small-RAM verdicts (B12, B8, B8-fast), final for 2026-10-22/23

## Verdict, in one line each

- **B12 (12 GB, MTP): close but not clean.** Task scores are within 5 points of the reference in both runs (math +2,
  IFEval -3.5, HumanEval+ -4.2, MGSM 0), and agentic turns change no more than the floor's. But it truncates more
  answers (16-17 vs 13; the rule allows 14), and its multilingual continuations drift well beyond the floor's.
- **B8 (9 GB, eager): borderline.** IFEval sits exactly at the limit over two runs (-5.0: run 1 -6, run 2 -4); math,
  HumanEval+ and MGSM are fine. Run 2 passed all 5 rules, run 1 failed two: the run-to-run noise is as large as the
  margin.
- **B8-fast (9 GB, MTP): fails, in both runs (3/5 each).** Pooled over the two: IFEval -5.5 [-10.0, -1.0] (past the
  -5 limit), HumanEval+ -4.2 [-8.7, +0.3], math -1.0 [-6.0, +4.0]; truncations 15 / 19 vs 13. Run 2 (10-24 09:10)
  failed HumanEval+ (-5.0 [-9.7, -0.3]) and truncations where run 1 failed IFEval (-7.0) and truncations.

- **B12L (12 GB, MTP; B12 + cold ranks 2-3 half a layer late), 10-23: the best 12 GB candidate so far, but same truncation
  failure.** Two runs pooled: math 86.0 % (+0.0), IFEval 86.0 % (-4.0 [-7.6, -0.4]), HumanEval+ 90.8 % (-0.8 [-3.9, +2.3]):
  every task rule passes in both runs (4/5 each); truncations 17 / 17 vs 13 fail both. Its two runs agree closely (IFEval
  86 / 86), unlike B12's. MGSM / multilingual / agentic rows are queued.

- **B12LL (12 GB, MTP; all cold ranks 2-9 half a layer late), 10-23: passes everything (5/5) in both runs, at about
  half B12's speed.** Its two core runs are token-identical (210/210 rows): B12LL is deterministic. Math 86.0 % (+0.0), IFEval 87.0 % (-3.0 [-6.7, +0.7]), HumanEval+ 91.7 % (+0.0, every
  answer equal to the reference's); final-answer agreement 96.7 % = the floor's 97.1 %; truncations 14 vs 13 (the first
  frugal config inside the rule). ~13 tok/s in this harness (Breakthrough: 21.2 / 13.1 tok/s code / prose with MTP).
  Shipping env (Fit12, 10-23): + BIGMOE_PF_GROUP=64 (MTP-layer prefill scratch) and BIGMOE_ASYNC_EVERY=1, same tokens
  by sha, peak 11.28 GB instead of the 12.85 GB of run 1's env; the suite uses them from run 2's second chunk on (run 2
  stayed token-identical to run 1). Its MGSM / multilingual / agentic rows are decoded (table: results/report_b12ll.txt).
- **B12LL-b16 (Fit12's per-pass expert budget on B12LL): fails IFEval.** Against the 4-bit ref: math 88.0 % (+2.0),
  IFEval 83.0 % (-7.0 [-11.2, -2.8], McNemar p 0.02), HumanEval+ 91.7 % (+0.0), truncations 15 vs 13: 3/5. Against
  B12LL: IFEval -4.0 [-8.6, +0.6], 5/5. Deterministic (its second run reproduced run 1's first 15 rows, then was stopped).
- **B12LL-b20 (budget b20): fails on truncations alone.** Against the 4-bit ref: IFEval 88 % vs 90 % (-2.0, p 0.62), math
  and HumanEval+ equal, truncations 15 vs 13 (the rule allows 14): 4/5; 5/5 against B12LL. Of its IFEval rows capped
  where the reference ended (3; net +2), ifeval/1675 rambles (deliberates aloud; ref: 45 tokens) and ifeval/2275 falls
  into a list loop ("Go find your this. Go find your these. ..."): real defects; ifeval/3126 is a long answer at the cap.
- **B12LL-UGB-0.5 (b16 only on MTP-confident passes; Fit12's hook, ugb=0.5), 10-24 04:29: IFEval exactly at the limit,
  truncations fixed.** Against the 4-bit ref: math 86.0 % (+0.0), IFEval 85.0 % (-5.0 [-8.6, -1.4], McNemar p 0.06: FAIL
  at the boundary), HumanEval+ 93.3 % (+1.7), truncations 13 = 13 (PASS: the first budget variant inside the rule): 4/5.
  Against B12LL: 5/5 (IFEval -2.0 [-6.0, +2.0] with the seeded scorer, math identical). Checks: it loads; gate 1.01 (never open) = B12LL token
  for token (6/6 math rows). Footprint 11.91 GB on the 6-row check, 13.80 GB over the full run, both without MLX's
  buffer-cache cap (Breakthrough / Fit12 measure at B12_CACHE_MB=64: 11.35 GB), so not comparable.
- **U2 = B12LL + Fit12's PR1 (keep-1 commits replay the PLE layer on 2 rows: tokens independent of the draft policy),
  10-24 15:05: same task scores as B12LL, fails on truncations alone.** Against the 4-bit ref: math 86.0 % (+0.0),
  IFEval 87.0 % (-3.0 [-5.8, -0.2], PASS), HumanEval+ 91.7 % (+0.0, every answer equal), answer agreement 97.6 % vs
  floor 97.1 %, truncations 16 vs 13 (the rule allows 14): 4/5. Against B12LL: math answers 50/50 equal, IFEval 87 = 87
  (2 items flip each way), HumanEval+ 60/60 equal, answer agreement 98.1 % (floor 95.7 %), truncations 16 vs 14: 4/5.
  PR1 moves the token stream about as much as a summation-order change does: 80/210 rows token-identical to B12LL (the
  floor vs ref: 59/210); IFEval rows part as early as token 2. The two extra capped rows vs B12LL are real repetition
  loops: ifeval/2275 ("You can see mountain views. You can see city skylines. ...", the item B12LL-b20 also looped on;
  B12LL 608 tokens, ref 716) and ifeval/3326 (a two-speaker "With here." / "With there." dialogue; B12LL 101, ref 447;
  the floor flags this item as a loop too). Neither costs an IFEval point (2275 fails in both, 3326 passes in U2).
  Against the ref, U2 also caps ifeval/3126 and c&p/430 (long answers that B12LL caps too; the ref ended at 1,000 / 1,497)
  and ends algebra/1282, which the ref caps. The loop counter (a 20-token n-gram 3x in the last 400 tokens) misses
  both loops; only the truncation rule sees them. Checks: an 8-row replay of B12LL on today's bt_b12 (v3f, trims off)
  was token-identical (8/8), so U2-vs-B12LL differences are PR1's; PR1 replayed 3,478 keep-1 commits in the last chunk.
  Speed 13.9 tok/s core vs B12LL's 13.3 (different days, not an A/B). Footprint 13.86 GB without the MLX cache cap
  (comparable to UGB's 13.80, not to Fit12's 64 MB-cap numbers). Fit12: tau width policies give U2's tokens exactly,
  so this verdict covers them. One run (the B12LL family is deterministic).
- **B12LL + top-8 (b12llk8; Leap's L1-17, Fit12's hook: each verify row keeps its top-8 experts, renormalized; prefill,
  one-row calls and the MTP head keep top-10), 10-24 18:42: passes 5/5 against the 4-bit ref and 5/5 against B12LL.**
  Against the ref: math 84.0 % (-2.0 [-5.3, +1.3]), IFEval 86.0 % (-4.0 [-7.2, -0.8]), HumanEval+ 91.7 % (+0.0), answer
  agreement 95.2 % vs floor 97.1 %, truncations 14 vs 13 (at the limit, like B12LL), loops 2 = 2. Against B12LL: math
  -2.0 (one problem flips), IFEval -1.0 [-5.4, +3.4] (4 / 3 flips), HumanEval+ +0.0 (2 / 2), truncations 14 = 14.
  Tokens move far more than under PR1: 38/210 rows token-identical to B12LL; most rows part within the first tokens (the
  first answer token comes from the first verify pass). Capped rows: the same count as B12LL's; new ones are ifeval/1000
  (a run-on essay to the 1,024 cap; ref 508, B12LL 549, floor 927 tokens) and algebra/1282 (the ref caps it too), while
  B12LL's capped HumanEval/130 and ifeval/3126 end. Speed in this harness 16.0 tok/s core vs B12LL's 13.3 (+20 %, not
  an A/B; Fit12's ABAB at 12 GB: +17.4 %, MTP acceptance intact). Footprint 14.56 GB without the MLX cache cap (U2 13.86
  in the same harness; not a controlled number). Checks: the hook's no-ops (topk=10, topp=1.0) gave B12LL's tokens on
  6/6 rows each; both chunks ran the same hook files (fit_hook f1182fb0ca, fit_topk a919ca9c07). One run. This is top-8
  on U1; Fit12's stack TK2 is PR1 + top-8 (b12llk8u2, judged next against U2).
- **PR1 + top-8 (b12llk8u2 = the tokens of Fit12's TK2 stack; tau 0.7 moves none under PR1), 10-24 20:58: passes 5/5
  against the 4-bit ref, 5/5 against U2 and 5/5 against B12LL.** Against the ref: math 86.0 % (+0.0), IFEval 90.0 %
  (+0.0 [-3.3, +3.3]), HumanEval+ 93.3 % (+1.7 [-3.1, +6.4]), answer agreement 95.7 % vs floor 97.1 %, truncations
  13 = 13, loops 2 = 2. Against U2: math answers 50/50 equal, IFEval +3.0 [-1.3, +7.3], HumanEval+ +1.7, truncations 13
  vs 16 (U2's two loops, ifeval/2275 and 3326, are gone). New capped row: HumanEval/93 (it deliberates about the spec
  in code comments to the 768 cap; ref 340 tokens). 38/210 rows token-identical to U2, 70/210 to b12llk8 (top-8 on U1).
  READ WITH NOISE: the two top-8 streams differ only by PR1's arithmetic yet score IFEval 86 vs 90, so single-run
  numbers carry about +-2 points. Pooled over the two (tools/pool_runs.py b12llk8,b12llk8u2), top-8 vs the ref: math
  -1.0 [-4.7, +2.7], IFEval -2.0 [-4.3, +0.3], HumanEval+ +0.8 [-3.7, +5.4], truncations 14 / 13; B12LL's two streams
  (U1, U2) pool to math +0.0, IFEval -3.0 [-5.8, -0.2], HumanEval+ +0.0, truncations 14 / 16: top-8 costs nothing
  measurable on these sets. Checks: the no-op (pr1, topk=10) = U2 on 6/6 rows; PR1 on at every load (3,990 replays in
  the last chunk); same hook files as b12llk8. Speed in the harness 15.9 vs U2's 13.9 tok/s (not an A/B; Fit12's ABAB
  at 12 GB: TK2 = PR1 + top-8 + tau 0.7 is +19.8 % over B12LL at <= 11.72 GB). Footprint 14.26 GB without the MLX cache
  cap (not comparable to Fit12's capped numbers).
- **B12LL + adaptive k (b12llka80: each verify row keeps the smallest rank prefix holding >= 80 % of its top-10 router
  weight), 10-24 22:38: FAILS, 1/5 against the 4-bit ref and 2/5 against B12LL, from what looks like a numerical
  fault.** Against the ref: math 64.0 % (-22.0 [-31.7, -12.3], 11 / 0 flips), IFEval 85.0 % (-5.0 [-8.6, -1.4]), HumanEval+
  66.7 % (-25.0 [-35.1, -14.9]), length x1.19, truncations / loops 42 / 42 vs 13 / 2. 42 of the 210 core rows (math 16,
  HumanEval+ 19, IFEval 5) turn into a flood of token 0 ("!!!!...") from token 19-125 on, to the cap: the signature of
  NaN / inf logits, seen in no other run (ref, B12LL, U2, both top-8 runs: 0 rows). So this measures a fault in the
  top-p path more than the idea of adaptive k. Diagnostic only (biased to the easier rows): on the 168 rows without a
  flood it is math 32/32 = ref, HumanEval+ 97.6 % vs ref 95.1 %, IFEval 87.4 % vs 92.6 % (-5.2 on 95 rows). Sent to
  Fit12 to debug; a re-run makes sense only after the floods are gone (Fit12's speed side had already shown code
  acceptance collapsing, 2.06 / 1.29 tok per pass, which fits). Speed moot (16.3 tok/s core in the harness).
- **D1 at 24 GB, POOLED over two runs (10-25 01:22): fails IFEval, -6.0 [-9.6, -2.4] against the 4-bit ref (past the -5
  limit); math, HumanEval+ and truncations pass.** Runs: IFEval 82 / 86 % (ref 90), math 86 / 90 % -> +2.0 [-3.2, +7.2],
  HumanEval+ 90.0 / 91.7 % -> -0.8 [-3.2, +1.6], truncations 14 / 14 vs 13 (inside the rule in both runs, unlike B12 at
  C_m 51: 16 / 17), loops 2 / 3 vs 2. Run 2 alone: 5/5 vs the ref (IFEval -4.0 [-8.6, +0.6]), 4/5 vs B12LL (loops 3 vs
  1). B12 at C_m 51 pooled IFEval -3.5 [-8.5, +1.5], so more residency did not buy IFEval back (the CIs overlap). SSD
  reads reproduce: run 2 27.6 per generated token (22.8 waited), 40 MB / token.
- **D1 at 24 GB (b12c192: Breakthrough's frugal wait-top-2 = B12 with C_m 192 / C_l 77, 38 % resident, MTP), run 1,
  10-24 23:29: fails IFEval, 4/5 against the 4-bit ref and 4/5 against B12LL.** Against the ref: math 86.0 % (+0.0),
  IFEval 82.0 % (-8.0 [-12.5, -3.5], McNemar p 0.01: 8 items lost, none gained), HumanEval+ 90.0 % (-1.7), answer
  agreement 93.3 % vs floor 97.1 %, truncations 14 vs 13 (PASS), loops 2 = 2. Against B12LL: math answers 50/50 equal,
  IFEval -5.0 [-9.9, -0.1] (FAIL at the boundary), HumanEval+ -1.7, truncations 14 = 14. The 8 lost IFEval items all end
  at EOS (no loops, no caps); each misses one constraint (capital-word counts x2, letter frequency x2, sentence count,
  nth-paragraph word, forbidden words, placeholders): instruction-following precision, not degeneration. Speed 30.9
  tok/s core in the harness (2.3x B12LL's 13.3); footprint 31.3 GB without the MLX cache cap. SSD (decode only, new
  row counters): 27.8 expert reads per generated token (22.9 waited; humaneval 29.0, ifeval 31.2, math 24.3), 40 MB
  per token, flat across the run (22 / 25 / 31 / 29 by row block: steady state, not warm-up), vs Leap's replay of 5.5
  rank-0-1 misses per token on code at C 192 (5x). Not reconciled yet: the counters do not split base / LSB-sidecar /
  MTP-head reads, and passes per row were not logged in run 1 (logged from now on), so the pass shape behind Leap's
  number cannot be compared. One hint: ~1.44 MB per read suggests about half are 0.77 MB LSB-sidecar reads (ranks 0-1
  need both parts at 4 bits; C_l 77 = 15 % resident vs C_m 192 = 38 %), so C_l is a candidate lever. Run 2
  (b12c192.r2, [low]) is decoding for the pooled verdict; D1's runs depend on pool history.
- Scorer note (10-24 04:35): the vendored IFEval checker called langdetect unseeded, so re-scoring could move IFEval by one
  item (UGB's rows scored 85 and 84). Seeded now (DetectorFactory.seed = 0); every table was re-scored with it (10-24
  06:10-08:00): no verdict moved; the only change was w2f_r10's IFEval (89 -> 88, still PASS).

How judged: like-for-like, the repetition guard OFF on both sides (the shipped guard turns some correct long-math
answers into wrong, capped ones); two runs per config where they exist, pooled per item; reference = the exact 4-bit
model, plain AR greedy (reproducible); floor = the same weights in another summation order.

## Core sets (210 rows per run: math 50, IFEval 100, HumanEval+ 60)

| test | B12 (2 runs) | B8 (2 runs) | B8-fast (2 runs) | ref | floor |
|---|---|---|---|---|---|
| math | 88.0 / 88.0 -> **88.0 %** (+2.0) | 86.0 / 86.0 -> **86.0 %** (+0.0) | 86.0 / 84.0 -> **85.0 %** (-1.0) | 86.0 % | 88.0 % |
| IFEval | 86.0 / 87.0 -> **86.5 %** (-3.5) | 84.0 / 86.0 -> **85.0 %** (-5.0) | 83.0 / 86.0 -> **84.5 %** (-5.5) | 90.0 % | 88.0 % |
| HumanEval+ | 88.3 / 86.7 -> **87.5 %** (-4.2) | 88.3 / 91.7 -> **90.0 %** (-1.7) | 88.3 / 86.7 -> **87.5 %** (-4.2) | 91.7 % | 95.0 % |
| truncations (core) | 16 / 17 | 17 / 13 | 15 / 19 | 13 | 12 |
| pre-registered rules per run | 4/5, 3/5 | 3/5, **5/5** | 3/5, 3/5 | | |

Pooled over the two runs (per item: mean of the runs minus ref; 90 % CI; tools/pool_runs.py): B12 math +2.0 [-3.7, +7.7],
IFEval -3.5 [-8.5, +1.5], HumanEval+ -4.2 [-9.8, +1.5]; B8 math +0.0 [-4.7, +4.7], IFEval **-5.0 [-8.6, -1.4]** (a real
loss, at the limit), HumanEval+ -1.7 [-5.6, +2.2]; B8-fast math -1.0 [-6.0, +4.0], IFEval **-5.5 [-10.0, -1.0]**,
HumanEval+ -4.2 [-8.7, +0.3]. Loops (core): B12 2 / 3, B8 3 / 1, B8-fast 3 / 2 vs ref 2.

Per-run deltas with 90 % CIs (paired per item): B12 IFEval -4.0 [-9.2, +1.2] and -3.0 [-7.9, +1.9]; HumanEval+ -3.3
[-8.8, +2.1] and -5.0 [-11.1, +1.1] (run 2 fails at the boundary). B8 IFEval -6.0 [-11.1, -0.9] and -4.0 [-8.0, -0.0];
HumanEval+ -3.3 [-7.2, +0.5] and +0.0 [-5.5, +5.5]. B8-fast IFEval -7.0 [-12.4, -1.6], HumanEval+ -3.3 [-8.8, +2.1].

## New rows (one run each, guard off)

| test | B12 | B8 | B8-fast | ref | floor |
|---|---|---|---|---|---|
| MGSM, 80 problems in 10 languages | 87.5 % (+0.0) | 91.2 % (+3.8) | 90.0 % (+2.5) | 87.5 % | 91.2 % |
| multilingual (24 prompts): similarity to ref, minus the floor's | 0.41 (-0.22 [-0.28, -0.17]) | 0.44 (-0.20 [-0.27, -0.13]) | 0.45 (-0.18 [-0.25, -0.11]) | | 0.64 |
| multilingual: answer letters in the prompt's script | 98.4 % | 98.5 % | 98.4 % | 99.0 % | 98.3 % |
| agentic (29 Claude Code turns, next 128 tokens): similarity, minus the floor's | 0.56 (-0.03 [-0.10, +0.05]) | 0.53 (-0.06 [-0.11, +0.00]) | 0.56 (-0.02 [-0.09, +0.05]) | | 0.58 |

## Notes

- Noise: frugal outputs depend on which experts earlier rows left resident; the same config's two runs differ by up to
  4 points per set (B8 HumanEval+ 88.3 vs 91.7) and by 4 truncations. Single-run results at 3-5 points are not decisive.
- The guard-on B12 run (b12g) scored math 82 / IFEval 85 / HumanEval+ 90 with 18 truncations: the shipped guard
  (before R1) cost B12 accuracy on long math. BigMoE's R1 guard (ea6e139, LaTeX math masked) is the main-track default now.
- Open (BigMoE b17): MTP and plain decode can differ after ~1,000 tokens; long rows of the MTP configs (B12, B8-fast)
  may carry that extra noise. B8 (eager) and the reference (plain) do not.
- Configs: B12 = `load_b12(0.10)`, C_m 51 / C_l 20, MTP spec125 3:3:0, release prefill; B8 = `load_b12(0.05)` +
  `B12_LATE3=2,4`, C_m 26, eager; B8-fast = `load_b12(0.05)` + `B12_W=3`, MTP. All `B12_NORM=mass`, 8-bit HC mixers.

## Missing

- Second runs of the new rows (one run each so far). (B8-fast's second core run: done 10-24 09:10, above.)
- w2f_r10's new rows (64 of 80 MGSM, no multilingual / agentic), plan4433332222's decode rows (50 of 234): fillers.
- Breakthrough's next candidates (02:42, confirmed by NewIdeas): B12L (B12 + late cold ranks 2-3) and B12LL (all cold
  ranks late).

Data: ~/EvalSuite/results/{b12,b12.r2,b8,b8.r2,b8fast,b12g,ref,floor}/decode.jsonl; tables results/report_<v>.txt;
items data/items/dec_*.jsonl (MGSM: data/raw/mgsm/SOURCE.txt).
