# LEDGER — drafterwall (one line per idea: idea | tag | test | number | verdict)

Marks: [K] measured, [P] arithmetic on measured numbers, [G] judgment. Tag = published / adapted / new, after at least
5 queries in ~/TRIED (TRIED.md + the per-owner files). "TRIED:" names the entry that already decided an idea.
Started Wed 11 Nov 2026 20:4x EET; final state 21:2x. Tests T1-T5 are pre-registered in PREREG.md (commit 602d66d);
scripts in scripts/, outputs in out/. Verdict words: KILL, BETWEEN, PASS, or "door" (real, but outside the rules).

## A. Use what the check already computes
1. Second look: reuse the 125B's outputs on the rows after the first miss, wherever the new draft agrees with the old
   one | adapted (Token Recycling, Ouroboros, Carryover 2609.14717; TRIED: M-06 RECYCLE KILL for the plain tail) | T1 |
   after a miss the later drafts are the 125B's tokens on 13.2 % (HumanEval+), 11.9 % (MBPP), 7.0 % (IFEval), 5.9 %
   (multi) of positions [K] | KILL
2. Jacobi iteration started from the drafter's block | published (TRIED: Lookahead / Jacobi KILL <= 5 %) | T1 | the
   same number: one repair per sweep, the rest of the block is derailed [K] | KILL
3. The MTP head run on the rejected rows' trunk states, to bridge the position after a repair | new | T1 | those
   states were computed on wrong inputs 84-94 % of the time [K] | KILL
4. Tails drafted without the selector chain, so that a miss does not derail the tail | new | a2_tail_tf | even told
   the true previous token, the readers are right on 47.6 % (DFlash2) / 76.5 % (MTP) of the later positions on code,
   23.0 / 42.5 % on IFEval [K] | KILL
5. Runner-up cache: the 125B's rank 2-3 tokens at verified rows as tie-breakers later | new | T5 | held-out x0.973 /
   x0.978 / x0.970 the MTP chain [K] | KILL
6. Session taste table (the same confusion pair seen before) | adapted (TRIED: misprediction cache) | none | 0.2-0.9 %
   of misses repeat [K DRAFTER_WALL] | KILL on the existing number
7. A linear readout or kNN fitted on verified rows | adapted | none | a new trained drafter (rule 1); K-HEAD showed
   the drafter's state holds no more than its own confidence [K TrainLab] | not allowed, and KILL by K-HEAD

## B. Change the shape of the check
8. Leaf rows at the lowest-confidence depths | published (TRIED: D-24 leaves KILL, -7 .. +0.9 %) | T3 | part of the
   tree curve below | KILL
9. Best-first trees over the readers' top-3 with batched expansion | published (TRIED: TREES KILL, -10 .. +2 % at equal
   cost) | T3 | ceiling with unlimited rows 13.61 of 16 on code [K]; +0.69 kept per doubling of rows on code, +0.27 on
   IFEval; 12.37 at 72 rows, 13.09 at 621 [P rows] | KILL for the target; on chat the known small lever stands (3.07
   at 7 rows against 2.74 for the uncut chain), gated by a tree check and row cost
10. Both readers' chains verified as rows of one check | adapted (TRIED: D-17 two-branch PARKED) | T3, T4 | the
    either-right oracle is 11.04 against 9.50 on code at 16 rows (x1.16), x1.09-1.18 held-out [K] | KILL (a 2-leaf tree)
11. Root fan: three candidates at the first draft position only, where top-3 coverage is 99.6 % | new variant | [P]:
    the first position misses on 2.2-2.6 % of code checks, so the fan adds <= 0.2 tokens per check for 2 more chains |
    KILL
12. Two-tier exact: cheap provisional sweeps, then one wide exact sweep that keeps almost every row | adapted (M-16
    runahead, FLOOR's two-pass check) | [P] on BREAK_THE_WALL's bytes: (2.7 + 5.97) / 10.5 + (2.7 + 18.1) / 62 = 1.16 GB
    per token against 1.05 for the plain exact check; the provisional sweeps still read the trunk once per 10.5 tokens
    | KILL (a high "kept per exact check" here would be a vanity number)
13. Staggered wavefront: start row j+1 a few layers after row j so the trunk drafts for itself inside one sweep | new
    | [P]: with a stagger of s layers the sweep reads 48 / s + B trunks for B rows, the autoregressive cost; with
    groups of rows it reads one trunk per group and knows less than a plain sweep | KILL
14. Fail fast: detect the miss in the early layers and restart only those | adapted (TRIED: R-32 STOPPRED KILL, early
    exit 5.7 %, EARLYTAP x0.73 without tap 45) | none | KILL on existing numbers
15. Soft tokens: feed a probability mix of the candidates at an unsure position so later rows stay useful | new | T1,
    a2_tail_tf | the tokens AFTER the unsure position are the drafter's own and are wrong once the guess is wrong
    (13 % right), and no reader knows them (item 4) [K] | KILL without a 125B run
16. Block cut right before the next predicted close call | adapted (TRIED: the t 0.3 cut ADOPTED; CUTSWEEP within 2 %)
    | none | already at its optimum

## C. Combine the readers that exist
17. Max-confidence pick between DFlash2 and the MTP chain | adapted (TRIED: MTPPICK x1.073 on code, KILL on the MTP
    step's cost) | T4 baseline | held-out x0.981 / x0.972 / x0.944 the better single reader [K] | KILL
18. Log-linear fusion of the two readers with depth weights (4 numbers, held-out) | new | T4 | held-out x1.038 (code),
    x1.009 (IFEval), x1.016 (multi); the perfect pick x1.151 / x1.183 / x1.090 [K] | KILL
19. N-way ensemble of the block-drafter checkpoints on disk | adapted (TRIED: DUALDRAFT x1.026, DIVERSE2: a twin is
    wrong at 87-90 % of the first one's misses) | none | KILL on existing numbers
20. Ensemble of MTP heads (stock, b3, b4, b5) | adapted | none | [G] small: same experts, same data | not screened
21. The MTP chain with a screened output layer, to make its step cheap | adapted (TRIED: MTPSCREEN PASS, FR-Spec desk
    KILL) | none | a cost lever, known; it does not change kept
22. Copy override (MATCH) | published (TRIED: shipped on code) | none | known: +4-11 % on HumanEval, 0 on MBPP
23. Per-check choice of reader | adapted (TRIED: D-17 router KILL, D-19 bandits KILL) | none | KILL on existing numbers

## D. The trunk itself as the drafter
24. Trunk plus resident experts, one step per token | adapted (TRIED: D-07, D2) | [P]: 2.7 GB per token, 9.5 ms at the
    byte floor | KILL (the autoregressive ceiling)
25. The same on the CPU or the Neural Engine in idle memory bandwidth | new | [P]: 2.7 GB and about 10 GFLOP per
    token, 30-100 ms per token on the CPU [G] | KILL
26. A lower-precision or shorter trunk | adapted (TRIED: D2 2-bit 0.066 code, 3-bit 0.92; early exit 5.7 %) | none |
    KILL on existing numbers
27. Trunk-refreshed reader: give the reader an exact trunk state every k tokens | new as a measurement | a5_refresh |
    per 32 rows HumanEval+ 20.05 / 18.32 / 15.82 / 14.39 / 12.55 and MBPP 12.12 / 9.88 / 8.03 / 6.82 / 6.22 at k = 1 / 2 /
    4 / 8 / 31 [K]; a state costs a sweep, so k is the block length | KILL as a design, kept as the law's ruler

## E. Laws and doors
28. Lag law: reader accuracy as a function of the distance from the last exact trunk state | new as a statement | T2 |
    MTP 97.7 % at depth 0, 90.8 % at 7-10, 86.6 % at 23-30 on HumanEval+; 93.5 / 76.2 / 69.6 % on MBPP; mid rows 85 to
    48 % [K] | BETWEEN by the letter (2 of 3 predictions; my either-oracle ceiling was too high: 22.75, not 25); the
    law holds
29. Entropy law: a sweep is one round of R parallel next-token questions, so kept grows like ln R over the reader's
    uncertainty | new | T3 | slopes above [K / P] | holds
30. Close calls per sweep: a check passes 1.2-1.9 of the 125B's own close calls on every domain while kept varies
    fourfold | new | a1_decisions | code 1.16-1.26, IFEval 1.70-1.87, multi 1.15-1.67 [K] | holds (descriptive)
31. A reader trained to hold lag-2 accuracy at depth | published (EAGLE-3, D-22) | T2, a3 | ceiling per 32 rows:
    HumanEval+ 12.55 -> 20.05 (24.43 with a perfect pick of two readers), MBPP 6.23 -> 12.12 (16.78), IFEval 2.74 ->
    4.05 (5.47) [K] | door, outside rule 1; sized by the GPU test LAG2-AGENT
32. Lossy acceptance at the 125B's own near-ties | published (TRIED: R-61N BETWEEN, non-exact) | none | FAST track,
    does not count here
33. Several independent streams in one sweep | adapted (TRIED: R-63S BETWEEN, WL-WIDTH PASS as a measurement,
    ROUTE-W-REPLAY 391-515 tok/s at 8-12 streams) | none | door for total throughput only
34. One stream at 1,000 tok/s with a drafter at lag >= 2 | new as a bound | a5_refresh [P] | byte floor, lossy oracle
    gate, lag-2 ceiling: HumanEval+ 706 tok/s (964 with the perfect pick at 64 rows), MBPP 441 (591); exact 373 / 241 |
    does not exist on real code
