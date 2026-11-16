# IDEAS_1116c_drift: the information limits of drift (Lunatic, THINK_1116c)

Mon 16 Nov 18:3x-19:1x. Brief: ~/FableGod/deepdraft/THINK_1116c.md; lens: the information limits of drift.

Desk + six tiny CPU screens on BigMoE's t30b dumps (`~/BigMoE/data/deepdraft/ceiling/`: cells + TOPK for chains,
tree_t30b_*_n127k8 for real 127-node trees; HE+ heplus24, MBPP mbppbank). Each takes < 1 GB and < 1 min; outputs are the
`.out` files beside each script:
- `drift_rankceil.py`: the own-information ceiling (ORACLE-31), K127, LAB, calibration by depth;
- `drift_bits.py`: same-target accuracy and bits lost by depth, anatomy of drift errors, free analogs;
- `drift_fork.py`: fork damage, controlled by the true-state guess;
- `drift_forkcal.py`: calibration after forks;
- `drift_labtree.py`: drift fixed + tie siblings, an upper bound;
- `drift_forksplit.py`: re-anchoring at the first fork, priced at the bound.

Marks: [K] measured on those dumps (dev, the same walk as every kept figure in WEEK98), [P] arithmetic on [K], [G]
judgement.

Terms:
- **fresh** = the true-state guess for target p: the depth-0 guess of anchor p-1. This is WEEK98's LAB line; it
  reproduces 20.77 / 12.72.
- **own** = the depth-d guess of anchor p-1-d (carried state; tokens teacher-forced, as in every walk).
- **fork** = a step on the own chain where the head's top-1 p < 0.5.

**CORRECTION to IDEAS_1116b (my 18:1x file):** its LAT stood on CC d1 + INERT wholesale. INERT wholesale is a recorded
quality KILL (TRIED CD16-DROP, 11-06), and the guarded T16 frontier is flat (FRONTIER#2). Under T16 a comment sibling is
not waste, so **LAT is void.** A banner is now at the top of that file.

## The answer in six lines
1. **775 at 32 rows needs the drift gap closed almost entirely (>= ~90 %), not half** [P on K].
   - Own chain 13.77; own tree-32 16.26; true-state chain 20.77 (HE+).
   - With the drift fully fixed, tree-32's x1.18-1.29 lifts that to ~24.5-26.8 (upper bound 26.79 [K]), just at the
     frame's 24.2-26.5.
   - Closing half the gap gives a ~17.3 chain and ~20-22 with a tree: ~600-700 tok/s at the frame's best.
2. **The head's own information is capped well below that** [K].
   - The best 31-node tree from its own 127 candidates (ORACLE-31) keeps 18.48 HE+ / 11.21 MBPP; EAGLE-2-31 keeps
     16.26 / 9.03.
   - The head is calibrated at every depth and after forks (p1 0.78 vs accuracy 0.77), so no re-scoring of its own
     candidates gets past that ceiling.
   - A sibling for every 2nd-choice miss, rows unbudgeted, still stops at 19.26.
3. **What costs kept is silent drift in confident chains, not forks** [K, same targets, true-state control].
   - HE+ chains that never passed a fork: own 96.3-96.5 % vs fresh 98.8-99.5 % per position at depths 2-30. A ~3x
     error rate that compounds over 20-30 positions.
   - After a fork the per-position gap is 10-19 pt, and it holds 10+ steps.
   - But re-anchoring at the first fork lifts kept only 13.77 -> 14.10 (HE+) and 6.92 -> 7.21 (MBPP). That is ~5 % of
     the gap, because the fork guess itself fails ~60 % of the time and ends the check anyway.
4. **In bits:** drift costs 0.08 -> 0.46 bits per position from depth 1-2 to 23-30 on HE+ (0.23 -> 1.25 MBPP). That
   is ~10 bits per 31-deep block on HE+ and ~28 on MBPP, spread over every position. The right token is still the
   head's 2nd choice in 55-75 % of HE+ drift errors [K].
5. **No free source carries those bits** [K].
   - An earlier matching context, whose 125B state is stored ("analog"), exists for 0.5-2.4 % of drift errors at
     >= 8 tokens (9-13 % at >= 4; its next token is right in only 1-3 %).
   - The state that fixes drift is the 125B's re-plan at every depth: the mid-depth computation that S12's tap carried
     and S14's shallow tap didn't.
6. **The price to beat** [P at Leap's C6 + int8 bound]: the cheapest drift information on record is the 125B's own
   rows.
   - K48 buys +2.6 kept for +5.5 ms per block; K64 buys +4.5 for +12.5 ms. That is **~2.1-2.8 ms per extra kept**,
     above tree-32's average of 1.77 ms per kept at the bound. That is why bigger trees are flat there.
   - The 2B audit pays ~5.2 ms per kept (P = 2: +1.45 for ~7.6 ms); the S12 taps far more.
   - A drift mechanism raises tok/s only below **~1.8 ms per extra kept**. Rows get there only with early death.

## 1. The information map [K, the .out files]
| HE+ / MBPP, kept per 32-row check | HE+ | MBPP |
|---|---|---|
| own chain-32 (t30b) | 13.77 | 6.92 |
| own tree-32 (EAGLE-2-31, real) | 16.26 | 9.03 |
| **ORACLE-31: best 31 of the head's own 127 candidates** | **18.48** | **11.21** |
| own chain + a sibling at every 2nd-choice miss (unbudgeted) | 19.26 | 11.68 |
| LAB: true-state chain-32 | 20.77 | 12.72 |
| K64 / K127, own information, 64 / 128 rows | 20.77 / 23.70 | 10.19 / 11.28 |
| **LAB + a sibling at every 2nd-choice tie (unbudgeted)** | **26.79** | **22.41** |
| LAB + top-4 siblings (unbudgeted) | 29.51 | 26.69 |

**Same-target accuracy, fresh vs own, by depth (HE+)** [drift_bits.out]:

| depth | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 |
|---|---|---|---|---|---|---|
| fresh | 97.93 | 97.91 | 97.88 | 97.84 | 97.79 | 97.72 |
| own | 96.30 | 94.41 | 93.06 | 92.16 | 90.94 | 89.65 |
| gap (pt) | 1.6 | 3.5 | 4.8 | 5.7 | 6.9 | 8.1 |
| bits lost per position | 0.08 | 0.19 | 0.27 | 0.32 | 0.39 | 0.46 |

- MBPP's gap: 4.8 / 10.4 / 14.2 / 16.7 / 19.4 / 21.5 pt; 0.23-1.25 bits.
- The 23-30 own figure (89.65) matches the brief's.

**Fork damage, controlled** [drift_fork.out, HE+; fresh / own on the same targets]:

| depths | no fork passed | 1 fork passed | 2+ forks passed |
|---|---|---|---|
| 2-6 | 98.8 / 96.5 (gap 2.3) | 89.7 / 79.3 (10.4) | 83.4 / 64.2 (19.2) |
| 7-14 | 99.4 / 96.4 (3.0) | 95.3 / 83.6 (11.7) | 87.9 / 69.9 (18.0) |
| 15-30 | 99.5 / 96.3 (3.2) | 97.8 / 86.1 (11.7) | 91.9 / 72.8 (19.1) |

- Since the last fork: 0-1 / 2-4 / 5-9 / 10+ steps -> gap 19.8 / 16.8 / 14.4 / 12.9 pt at depths 15-30. The damage
  persists.
- Fork chains hold 34 / 54 / 70 % of the per-position gap at depths 2-6 / 7-14 / 15-30.
- **Not a branch-choice problem:** these cells feed the head the 125B's own token at the fork, and it still drifts.
  What's missing is the re-plan after a branch it didn't expect, not which branch.

**Calibration** [drift_rankceil.out, drift_forkcal.out]:
- On HE+, the edge p of popped children tracks the on-true rate within 0.5 pt at every depth.
- After forks: mean p1 0.782 vs accuracy 0.769 (HE+), 0.742 vs 0.716 (MBPP).
- Drift shows up as honest blur. EAGLE-2's path_p already prices it, which is why ORACLE-31 is only x1.14 / x1.24 over
  it.

## 2. Idea 1 (best): BUY DRIFT WITH ROWS, and price every drift mechanism against rows
**Mechanism:**
- The information drift destroys is the 125B's own computation on the drafted tokens. The only source that delivers it
  at scale is a 125B row: a tree node the 125B processes.
- K64 with the head's OWN information equals the true-state chain (20.77 vs 20.77), and K127 beats it (23.70).
- So, for this week, the drift lever is cheaper rows: early death (EARLYDEATH_PREREG, Wed), plus the levels big trees
  need (the QAT head + fused level).
- **The number it moves:** kept per check, own information, at 32 / 48 / 64 / 128 rows: 16.26 / 18.86 / 20.77 / 23.70
  (HE+), 9.03 / 9.73 / 10.19 / 11.28 (MBPP) [K treek_final].

**tok/s at the bound** [P; C6 + int8, dense max(7.6, 4.9 R/32), experts by the code union law, draft 3 ms per 31
levels, overhead 1.5]. Cost per block without early death:

| K | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|
| HE+ tok/s | 515 | 561 | **566** | 551 | 504 | 418 | 356 |

- That is flat at K24-48 and falling after.
- **The price of a kept from rows is 2.1-2.8 ms per block** (K32 -> K48 / K64), vs the 1.77 ms average per kept at
  K32. Rows pay only once early death brings their marginal price below that.
- For K64 to beat K32 by >= 10 % at the bound, a 64-row check must cost <= ~1.15x a 32-row one. Early death must make
  rows 33-64 cost <= ~3.7 ms in all. That is the PASS line I'd put on Wednesday's probe for the drift purpose.

**Cost per block on the 125B:** the extra rows minus what dies early. No new quality risk: exact verification
unchanged.

**What kills it:** early death leaves rows 33-64 at > ~0.3x their full cost. Then no drift lever this week pays at the
bound, and 775 lossy via the drafter is out by Saturday.

**Screen:**
- CPU, done: drift_rankceil + the treek curve.
- The live one is Wednesday's pre-registered early-death probe; add the line above (64-row cost <= 1.15x a 32-row
  one).

**Tag:** published mechanism (dynamic trees; early-exit verification). New here: the exchange rate between rows and
drift information, and the bar it sets.

## 3. Idea 2: the bar for any new drift mechanism (an information budget, not a lever)
- **For 775:** close >= ~90 % of the own -> true gap (Section 0 line 1).
- **For a gain at all:** < ~1.8 ms of added per-block cost per extra kept at the C6 + int8 bound (HE+). That is
  tree-32's average: 28.7 ms / 16.26. Measured candidates:

| source | +kept per check (HE+) | added cost per block | ms per kept | verdict |
|---|---|---|---|---|
| 125B rows, K32 -> K48 | +2.6 | +5.5 ms | 2.1 | the cheapest; above 1.8 without early death |
| 2B audit, P = 2 (C2) | +1.45 (+8.9 %) | ~7.6 ms (2 x 1.06 GB) | ~5.2 | below the bar |
| 4B audit, P = 2 | +2.4 (+14.9 %) | ~16.9 ms (2 x 2.37 GB) | ~7 | below the bar |
| S12 tapmlp (mid-depth tap at every drafted position) | ~+0.8 (+5.1 %, sandbox) | a partial 125B pass per drafted token (~1.7 GB each) | >> 10 | far below |
| one true state at the first fork (CPU, drift_forksplit) | +0.33 (+2.4 %) | a partial or full pass per fork (1.7-3.5 GB) | >> 10 | far below |
| analog (stored) true states | <= coverage of 0.5-2.4 % of drift errors | ~0 | n/a | KILL on coverage |

- **The information limit:** fixing silent drift needs the 125B's re-plan at EVERY drafted depth (~10 bits per block on
  HE+, spread out). A source per depth on the 125B costs a pass per depth; a source per block (the auditors) carries a
  fraction of the bits. So no head-side mechanism on record closes half the gap inside 3-7 ms per block, and the
  bound says none will without a new source of the 125B's mid-depth computation.
- **What to stop:** further head-side state levers aimed at 775 this week. Capacity x2, t30c, carryac, SIB, taps, and
  now analogs and fork re-anchoring are all on the wrong side of the bar.

## 4. Idea 3 (a Sandbox98 screen worth 30 min, information only): is silent drift a READOUT or a STATE problem?
- **Why:** in confident chains the own head is 96.4 % vs fresh ~99 %, and the right token is usually its 2nd choice.
  If the carried state still contains the answer and only the readout is blurred, a cheap readout fix could recover
  part of it with zero 125B bytes. If the state has lost it, nothing head-side will.
- **Mechanism tested (the readout half):** a linear probe from the carried state at depth d to the true state's
  top-1 (or the true-state logits). It is trained on the sandbox's own / true pairs; the SIB harness already makes
  both, teacher-forced.
- **Screen:** Sandbox98, <= 5 GB, <= 30 min.
  - Fit the probe on 70 % of HE / MBPP sandbox rows at depths 3-14; test on 30 %.
  - Report walk-strict bands 3-14 with the probe's argmax vs the head's.
  - **PASS (readout):** the probe closes >= 30 % of the own -> true gap at bands 3-14 out of sample.
  - **KILL (state):** < 10 %, meaning the carried state no longer holds the 125B's choice.
- **Cost if it passed:** one small matmul per level (~MB), no 125B bytes. Quality: none (drafting only).
- **Bound from records:** the brief's direction error of 0.85-0.99 (carried vs true) predicts KILL. carryac (state
  supervision) gave +0.6-0.8 pt. So I expect < 10 % [G].
- Worth 30 min only because a PASS would be the one zero-byte drift lever left.

## 5. Killed today [K]
- **ANALOG STATES** (weather's method of analogues / SLAM loop closure): a stored true state from an earlier matching
  context. Coverage 0.5-2.4 % of drift errors at >= 8 tokens.
- **FORK RE-ANCHOR** (a true state right after the first fork, then a second wave):
  - kept 13.77 -> 14.10 HE+, 6.92 -> 7.21 MBPP (~5 % of the gap);
  - tok/s at the bound 542 / 309 vs tree-32's 566 / 321 (-4 % both) [P].
  - The +17-33 % over the plain chain is check-size savings, which trees already capture.
- **FORK-AWARE TREE RE-SCORING:** the head is calibrated after forks. The ceiling for any re-scoring is ORACLE-31 (x1.14
  HE+, x1.24 MBPP over EAGLE-2-31).

## What this changes
- **The drift wall is an information wall at this week's budget.** 775 lossy at 32 rows needs LAB-level drafting PLUS
  trees. The bits that LAB has are the 125B's per-depth re-plan. Only 125B rows deliver them at a near-paying price:
  2.1-2.8 ms per extra kept vs the 1.8 bar.
- **Put the drift budget on cheaper rows** (early death) and cheaper levels. Fund a new drift mechanism only if its
  screen shows < ~1.8 ms per extra kept.
- **Honest Saturday:** HE+ lossy at quality ~340-420 measured (NewIdeas FRONTIER#2). The projection to 775 needs a
  near-perfect drift fix that no record supports this week.

## Reproducibility
`cd ~/FableGod/lunatic && ~/BigMoE/.venv/bin/python3 -I drift_rankceil.py` (then `drift_bits.py`, `drift_fork.py`,
`drift_forkcal.py`, `drift_labtree.py`, `drift_forksplit.py`).
- Read-only on BigMoE's dumps and EvalSuite's certify98.
- Peak 0.7 GB; 15-60 s each.

## TRIED lines (ready to paste; Lunatic, 2026-11-16)
- DRIFT-INFO: drift anatomy on the t30b dumps (same targets, true-state control) | wall: drafter / drift | line: all | exact: n/a | verdict: MEASURED: own-information ceiling ORACLE-31 18.48 HE+ / 11.21 MBPP (EAGLE-2-31 16.26 / 9.03; LAB chain 20.77 / 12.72; LAB + 2nd-choice siblings, unbudgeted, 26.79 / 22.41); drift = 0.08 -> 0.46 bits per position HE+ (0.23 -> 1.25 MBPP), ~10 / ~28 bits per block; fork damage 10-19 pt per position (vs 2.3-3.2 without forks) but kept is lost to silent drift in confident chains; head calibrated after forks; 775 at 32 rows needs >= ~90 % of the gap closed | ~/FableGod/lunatic/drift_*.out, IDEAS_1116c_drift.md | 2026-11-16 | [lunatic]
- ANALOG-STATES (stored true state of an earlier matching context as the head's state) | wall: drift | line: all | exact: yes (drafting) | verdict: KILL on coverage: >= 8-token analogs exist for 0.5-2.4 % of drift errors (>= 4: 9-13 %, next token right 1-3 %) | ~/FableGod/lunatic/drift_bits.out | 2026-11-16 | [lunatic]
- FORK-REANCHOR (true state at the first doubtful step, second wave) | wall: drift | line: all | exact: yes | verdict: KILL: kept 13.77 -> 14.10 HE+ / 6.92 -> 7.21 MBPP (~5 % of the own -> true gap); tok/s at the C6 + int8 bound -4 % vs tree-32 | ~/FableGod/lunatic/drift_forksplit.out | 2026-11-16 | [lunatic]
- ROWS-FOR-BITS price: 125B rows buy kept at 2.1-2.8 ms per block per extra kept at the C6 + int8 bound (K32 -> K48 / K64, HE+); 2B audit P = 2 ~5.2, 4B ~7, taps >> 10 | wall: drift | line: lossy | exact: yes | verdict: PRICE; the bar for any drift mechanism is < 1.8 ms per extra kept (tree-32's average at the bound), so rows pass only with early death; early-death line for the drift purpose: a 64-row check <= 1.15x a 32-row one | ~/FableGod/lunatic/IDEAS_1116c_drift.md | 2026-11-16 | [lunatic]
