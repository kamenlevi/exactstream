# IDEAS_1115_limits: which walls are real? (Lunatic, problem 5, Sun 15 Nov 22:0x-22:4x)

Brief: ~/FableGod/deepdraft/THINK_1115.md, problem 5: QUESTION THE LIMITS.
Desk only: one tiny CPU script, `lim_kept.py` -> `lim_kept.out` (product rule over the THINK table's per-band
accuracies). Marks: [K] measured (file named), [P] arithmetic on [K], [G] judgement.

**The product rule and its calibration** [P]:
- kept/32 = 1 + sum over depths 0-30 of the product of per-depth accuracies (bands 0 | 1-2 | 3-6 | 7-10 | 11-14 |
  15-22 | 23-30).
- It reproduces the measured kept within +4-5 % (t30b chain 14.44 vs 13.77 [K]; true state 21.67 vs 20.77 [K]).
- Below, "cal." = x0.956.
- Cycle for EXACT = the 52.0 ms floor + 7.5 ms drafting and overhead = 59.5 ms [K WEEK98 drafter spec]. LOSSY uses the
  25 ms check.

## The verdicts first
| wall | real or assumption | binding for |
|---|---|---|
| 125B near-ties at check starts | **REAL for exact chains**, an assumption for lossy | 400 exact (the deciding wall) |
| expert union makes big trees expensive | **REAL** (independent experts; siblings route like new rows) | where tree rows are spent, not whether |
| drafter drift | real, partly closable (it is missing information, not a law) | both targets, about equally with the near-ties |
| 52 ms exact floor | **REAL** physics; the engine's 99-102 ms is not a wall but a 1.9x engineering gap | both |

## Idea 1: the after-miss region (bands 0-6) is the wall that decides 400 exact, and for chains it is real
**Mechanism (the finding):**
- Give the head the 125B's exact state at every depth (the oracle). It still holds only 93.2 / 95.3 / 96.4 % at bands
  0 / 1-2 / 3-6 [K THINK table].
- The 125B's choice there depends on its own processing of the previous token, which no chain head has.
- So a chain, however well-informed, stops at 20.77 kept/32 [K]: 349 tok/s at the 59.5 ms cycle, 399 only if the
  7.5 ms overhead were zero.
- The only exact way past it is computing the 125B on more than one candidate: branching (tree nodes ARE that
  processing).

**Numbers** [P, lim_kept.out, cal.]:

| profile | kept/32 (cal.) | tok/s at 59.5 ms | tok/s at 25 ms (lossy) |
|---|---|---|---|
| t30b chain (real) | 13.8 | 232 | 552 |
| true state (oracle chain) | 20.7 | 348 | 829 |
| oracle deep bands, chain bands 0-6 | 16.8 | 282 | 670 |
| oracle bands 0-6, chain deep bands | 16.9 | 284 | 675 |
| tree-16's bands 0-2 + oracle beyond | 22.3 | 375 | 893 |
| bands 0-6 at 97.8 + oracle beyond | **23.8** | **400** | 951 |
| bands 0-6 perfect + oracle beyond | 27.2 | 456 | 1,086 |

Readings:
- Fixing bands 0-6 is worth as much as closing ALL the deep drift: +3.1 vs +3.0 kept from the real chain.
- 400 exact with overhead needs bands 0-6 at ~97.8 across the WHOLE after-miss region, not only depths 0-2.
- Tree-16's depths 0-2 alone stop at ~375.
- Only m < 0.25 nats is strictly unfixable by any drafter: 9.4 / 11.1 % of the deep common misses [K LEVERS_A_MOONSHOTS
  F]. The rest is the 125B's own wording at 0.25-4 nats, which the 125B's processing of each candidate resolves.

**Tag:** a measurement and a decision rule. Branching after misses is published (EAGLE-2, SpecInfer), and its
composition with the audit is proposed (TRIED 1018 TREE-PLUS-AUDIT, NewIdeas). New here: the price shows a chain
cannot reach 400 with overhead even with oracle information. So trees over bands 0-6 are NECESSARY for 400 exact, not
an option.

**Exact:** yes.

**Kill:** fresh-data bands 0-6 stay < ~97 % under the best 32-node tree with the audited chain beyond. Then 400 exact
with overhead is out of reach by Saturday; certify ~350-390, and say "400" only at a 52 ms cycle.

**Monday test (15 min, CPU):** TRIED 1018's desk join (tree_t30b_heplus24.npz + rows2b_heplus.npz), extended to bands
3-6.
- Report bands 0-6 coverage and composed kept/32.
- PASS for 400 exact: composed kept/32 >= 23.8.
- BETWEEN: 22-23.8.
- KILL: < 22.

## Idea 2: for LOSSY the near-tie wall is an assumption; 1,000 is a check-time wall
**Mechanism:**
- 92 % (HE+) / 88 % (MBPP) of check-start misses sit where the 125B's margin is < 2 nats [K WEEK98 S6 CSM]. Close-call
  acceptance takes them directly.
- With bands 0-6 at ~99.5 under close calls [G], what is left is the deep bands and the check time.

**Numbers** [P, cal.]:

| close calls at bands 0-6, deep bands = | kept/32 | 25 ms | 20 ms |
|---|---|---|---|
| t30b chain | 21.3 | 851 | 1,064 |
| chain + 2B audit (98.3 at 15-30) | 22.2 | 888 | 1,110 |
| oracle state | 26.4 | 1,054 | 1,317 |

- So 1,000 lossy needs a <= 21-22 ms lossy check with today's deep bands, OR near-oracle deep bands at 25 ms.
- Close calls everywhere at 99 % gives the same 26.3 as the last row.

**Tag:** published (typical acceptance, Judge Decoding, lossy speculative decoding); in the pipeline (LOSSY_GATE.md's F_hi
and margin bins).

**Exact:** no.

**Kill:** EvalSuite's gate fails at delta 2 nats restricted to bands 0-6. (Check starts are where close calls
concentrate: 16.9 vs 5.4 % of anchors, HE+.)

**Monday test (15 min, CPU on the walk capture):**
- Apply close-call acceptance at bands 0-6 only (delta 1 and 2 nats).
- Read kept/32 and LOSSY_GATE's F_hi flips binned by margin.
- PASS: kept/32 >= 21 (cal.) with F_hi at the gate's line.

## Idea 3: "check starts are hard" is only true after a MISS, so let the miss pick the next check's shape
**Mechanism:**
- After a full accept, depth 0 is right 99-100 %; after a miss 89.2-89.3 % [K WEEK98 S6 CSM: "no dip after a full
  accept"]. Hard rows run in a row.
- The check that ends on a miss also knows the 125B's EXACT margin and entropy at the bonus row, free.
- Rule:
  - after a full accept, or a miss with margin >= 2 nats (8-12 % of check-start misses), build a pure 32-deep chain;
  - after a close-call miss (m < 2), build the tree that spends its branches over bands 0-6 (Idea 1).
- The signal is the target's own exact computation, not the drafter's confidence.

**Numbers:**
- Depth-0 accuracy 99-100 % vs 89.2 %.
- Check starts sit on close calls 3.1x more than other anchors (16.9 vs 5.4 % HE+) [K].
- Gain: modest [G], set by the share of after-accept and high-margin-miss checks, which takes no branches. Measure it
  before believing it.

**Tag:** adapted. Dynamic trees exist (EAGLE-2 by the drafter's confidence; Sequoia / OPT-Tree by size). Using the
previous check's exact 125B margin and the start type as the shape switch is not in TRIED (greps: start type,
after-miss, bonus margin, check-start). RECYCLE (TRIED 975) is a different lever (reusing rejected rows).

**Exact:** yes.

**Kill:** tree-vs-chain kept/32 does not differ by start class.

**Monday test (15 min, CPU on MTP Head's S11 per-check data):**
- Classify checks: after full accept / miss with m >= 2 / miss with m < 2.
- Compare chain-32 vs tree-32 kept in each class.
- PASS: the switch beats the best fixed shape by >= 3 % kept/32.
- KILL: < 1 %.

## Idea 4: the 52 ms floor and the union cost are real; the lever there is the engine's 1.9x, not physics
**Mechanism:**
- The exact check reads the 32-row union (108.5 experts per layer, 14.56 GB) at ~280 GB/s [K THINK].
- Nothing exact cuts those bytes:
  - the 512 experts are independent functions on the real weights: index-aligned mean expert 0.78-1.02 % of the energy
    (independent: 0.78 %), best neuron |cos| across experts median 0.086-0.090 (chance) [K lunatic/wl_redund.out];
  - lossless entropy coding saves 13 % of bytes but decodes slower than it saves (TRIED 93, M-27);
  - siblings route like new rows: 4 siblings ~21 experts vs ~10 [K THINK].
- So spending tree rows costs what any row costs. Idea 1 says spend them on bands 0-6 after close-call misses.

**Numbers:** engine today 99-102 ms vs the 52.0 floor at 32 rows [K Leap check55_plan] = x1.9 on every tok/s figure. It
is the largest single lever for both targets, and it is a build (Leap CHECK55), not a Saturday certification item.

**Tag:** measurement (show the wall is real).

**Exact:** yes.

**Kill / revive:** revive only if a bitwise kernel reads fewer bytes than the union. No path is known: T1 removes the
structural ones.

**Monday test:** none needed. Certify the floor's arithmetic (bytes / bandwidth) beside the engine's measured check.

## Idea 5: tie-tolerant exactness (a definition question, small)
**Mechanism:**
- Below some margin epsilon, two certified exact kernels (REF2-T, NAX6) can pick different tokens. There, the
  reference's choice is a rounding accident, not the 125B's preference.
- Define EXACT as "a greedy decode of the 125B under a certified kernel" and accept either side at m < epsilon, where
  epsilon is the measured cross-kernel logit spread.

**Numbers:** m < 0.25 nats is 9.4 / 11.1 % of the deep common misses [K]. The share below a realistic epsilon (<= 0.05
nats) is smaller. Gain ~+1-3 % kept [G].

**Tag:** new as an acceptance rule [G; unchecked]. Related: work on nondeterministic inference (batch-invariant
kernels).

**Exact:** bitwise to some certified kernel, not to REF2-T. Kamen's call. Not for Saturday.

**Kill:** < 3 % of misses sit inside the measured cross-kernel spread.

**Monday test (10 min, CPU):** the logit spread NAX6 vs REF2-T on shared rows, against the margins of the walk's misses.

## What this changes for Saturday (one paragraph)
- The certification should say per band what is physics and what is drafting.
  - Bands 0-6 after a miss are capped at 93-96 % for any chain, even with the 125B's exact state [K]. That is the
    125B's own wording uncertainty, and only branching (exact) or close-call acceptance (lossy) moves it.
  - Bands 7-30 are drafting: oracle state holds 97.8-99.3 %, real chains 94.3-97.7 %.
- "400 exact" holds only at a 52 ms cycle with oracle-level drafting, or at 59.5 ms with bands 0-6 lifted to ~97.8 by
  trees AND oracle-level deep bands [P].
- "1,000 lossy" is a check-time statement once close calls are accepted: ~21 ms with today's deep bands [P].
