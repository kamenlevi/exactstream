# RESULT — the drafter wall, without retraining (drafterwall, Wed 11 Nov 2026, 20:10-21:2x EET)

Marks: [K] measured here on recorded data (script named), [P] arithmetic on measured numbers, [G] judgment.
Rules kept: no training of anything; exact (drafts only propose); CPU only, no GPU job was run; no downloads; writes
only in this folder; 34 ideas tagged against ~/TRIED (LEDGER.md); five tests pre-registered in PREREG.md (commit 602d66d) before
they ran, exploratory reads disclosed there. "Reader" below = a drafter (DFlash2 L1-20 or the MTP head b5).
"Sweep" = one check = one read of the 125B's trunk.

## 0. Verdict in seven lines
1. **The wall was not broken.** Nothing exact and training-free moves kept per check by 20 %. Of the five
   pre-registered tests, four are KILL and one is BETWEEN; the rest of the 34 ideas die on existing numbers or
   arithmetic, or are doors outside the rules.
2. **Law 1, causality** [K T1]: a sweep is one round of parallel next-token questions, and everything after the first
   wrong draft is about a different text. After a miss only 13.2 % (HumanEval+), 11.9 % (MBPP), 7.0 % (IFEval) and
   5.9 % (multilingual) of the later drafts are the 125B's tokens. Every "second look" at rejected rows dies here.
3. **Law 2, lag** [K T2]: both readers are right 97.7-97.8 % of the time two tokens past the last exact trunk state
   and lose it with distance: 90.8 % by depths 7-10 on HumanEval+, 76.2 % on MBPP. At the 125B's own close calls the
   fall is 85 % to 48 %. A trunk state that is even one token older costs 9-18 % of kept [K a5].
4. **Law 3, rows buy the logarithm** [K ceilings, P rows, T3]: with unlimited rows, every picker or tree over both
   readers' top-3 keeps at most 13.61 of 16 on code; a best-first tree gains +0.5-0.7 kept per doubling of rows on
   code and +0.19-0.27 on chat (MTP and fused readers); 1,000 rows keep 13.2 (code), 4.5 (IFEval), 4.0 (multilingual).
5. **Combining what exists is exhausted** [K T4, T5]: a log-linear fusion of the two readers, tuned on half the
   windows, keeps x1.04 / x1.01 / x1.02 on the held-out half; the perfect pick is only x1.09-1.18; the 125B's own
   second choices as tie-breakers lose 2-3 %.
6. **The door is lag, and it is outside this mission's rules.** A reader that held its lag-2 accuracy at every depth
   would keep 20.1 per 32-row check on HumanEval+ against 12.6 today, and 12.1 against 6.2 on MBPP; with a perfect
   pick between the two readers, 24.4 and 16.8 [K T2, a3]. Only training can give that, and even it stops short of
   25-30 on real code.
7. **So 1,000 tok/s on one stream needs more than a drafter** [P a5]: at the byte floor, with the lossy oracle gate
   and the lag-2 ceiling, HumanEval+ reaches 706 tok/s (964 with the perfect pick at 64 rows) and MBPP 441 (591).
   The one GPU test below sizes the door on agent turns, the workload none of today's tables cover.

## 1. What was measured
All numbers are kept tokens per check (accepted drafts + the bonus token), official loop, on recorded tables.

### 1.1 Law 1: after a miss, the rest of the block is about another text (T1, KILL)
`scripts/t1_secondlook.py` on the live HPD-0 walks of DFlash2 L1-20 (files the exploratory reads did not touch).

| set | checks with a miss | later drafts equal to the 125B's token | next 3 drafts all right |
|---|---|---|---|
| HumanEval+ (24) | 440 | 13.2 % | 10.9 % |
| MBPP (100) | 2,404 | 11.9 % | 8.4 % |
| GSM8K (60) | 1,775 | 16.3 % | 10.9 % |
| IFEval (30) | 2,808 | 7.0 % | 2.9 % |
| multilingual (24) | 3,270 | 5.9 % | 2.5 % |

Line: PASS at 40 % (code) or 30 % (chat), KILL under 25 % and 15 %. Result: KILL.
- The 125B's outputs on the rows after a miss were computed on inputs that are wrong 84-94 % of the time. That is
  why MoonShots' recycling test kept 0.6 tokens (M-06), and it kills the wider family in one number: reuse where
  the new draft agrees with the old one, Jacobi started from the drafter, the MTP head on rejected rows' states.
- It is not a hidden one-token shift: comparing the later drafts with the true tokens one or two positions earlier
  or later matches less often, 4.6-10.5 % against 13.6 % on the code map windows [K `scripts/s0_align.py`].
- A tail that does not derail cannot be built either [K `scripts/a2_tail_tf.py`]: told the true previous token at
  every step, the readers are right on only 47.6 % (DFlash2) and 76.5 % (MTP) of the later positions of the same
  block on code, 23.0 / 42.5 % on IFEval. Soft or per-position tails need tokens nobody has.

### 1.2 Law 2: accuracy is a function of the distance from the last exact trunk state (T2, BETWEEN)
`scripts/t2_laglaw.py`. Depth 0 is the first draft after the bonus token: the reader holds the exact trunk state two
tokens back. MTP head b5, teacher-forced, on files not seen before the predictions were written:

| set | depth 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 |
|---|---|---|---|---|---|---|---|
| HumanEval+ (24 answers) | 97.7 | 96.0 | 93.3 | 90.8 | 89.4 | 87.8 | 86.6 |
| MBPP (100 answers) | 93.5 | 87.6 | 81.4 | 76.2 | 73.2 | 70.9 | 69.6 |
| GSM8K | 96.1 | 92.1 | 87.3 | 83.6 | 80.7 | | |
| MGSM | 92.4 | 85.7 | 79.0 | 74.3 | 71.5 | | |

By the 125B's own class, 12 code map windows (confirmation; `scripts/s0_class.py`): sure rows DFlash2 99.9 to 74.2 %,
MTP 99.8 to 94.7 % from depth 0 to depths 11-14; mid rows 83.7 to 24.5 % and 84.7 to 47.8 %. IFEval mid rows: 65.2 to
12.7 % and 69.3 to 24.6 %.

The "lag-2 everywhere" drafter is a thought experiment: it is right at a position exactly when the reader is right
there at depth 0. It needs an exact trunk state for every token, which is the autoregressive cost.

| set | real chain, 32 rows | lag-2 everywhere, MTP | lag-2, perfect pick of the two readers | same at 64 rows |
|---|---|---|---|---|
| HumanEval+ (24) | 12.55 | 20.05 | 24.43 | 27.36 / 37.28 |
| MBPP (100) | 6.23 | 12.12 | 16.78 | 13.52 / 20.88 |
| IFEval (12 map windows) | 2.74 (16 rows) | 4.05 | 5.47 | |
| multilingual (12 map windows) | 2.80 (16 rows) | 4.37 | 5.07 | |

(`scripts/a3_lag2_either.py`, `gpu_test/g1_lag2_readout.py`; the read-out reproduces TrainLab's 9.58, 12.55, 5.94 and
6.23.) Predictions: P-a true (97.7 at depth 0, 90.8 at depths 7-10), P-b true (93.5 to 76.2), P-c true for the MTP
alone on HumanEval+ (20.12 against a line of 20) and for MBPP (12.20 under 16; t2 and a3 differ by 0.07-0.08 in how they end a window), false for the either-oracle on the 12
map windows (22.75 against 25). By the letter: BETWEEN. The law holds; its ceiling is lower than I predicted.

Window-level bootstrap, 90 % intervals [K `scripts/a6_ci.py`]: HumanEval+ chain 12.55 [10.62, 15.07], lag-2 MTP 20.05
[16.64, 24.20], ratio x1.60 [1.53, 1.68], perfect pick x1.95 [1.81, 2.08]; MBPP chain 6.23 [5.88, 6.61], lag-2 MTP 12.12
[11.26, 13.11], ratio x1.95 [1.88, 2.02], perfect pick x2.70 [2.61, 2.79]. Twenty-four answers make the HumanEval+ levels
loose; the ratios are tight.

How fresh the state must be [K `scripts/a5_refresh.py`]: the chain restarted from an exact trunk state every k positions
keeps, per 32 rows, HumanEval+ 20.05 / 18.32 / 16.95 / 15.82 / 14.39 / 12.55 and MBPP 12.12 / 9.88 / 8.68 / 8.03 / 6.82 /
6.22 at k = 1 / 2 / 3 / 4 / 8 / 31.

Where checks end [K `scripts/a1_decisions.py`, 12 map windows per domain, 16 rows]: the MTP chain's checks end at a mid
or tie row 43 % of the time on code (41 % reach the block end), 94 % on IFEval, 89 % on multilingual. A check passes
1.2-1.9 of the 125B's close calls on every domain, while kept varies fourfold: kept is close calls per sweep divided
by the text's density of close calls (0.135 code, 0.684 IFEval, 0.596 multilingual).

### 1.3 Law 3: rows buy the logarithm (T3, KILL)
`scripts/t3_rowslog.py`, 12 map windows per domain, 16-row checks.

| | DFlash2 top-1 | DFlash2 top-3 | MTP top-1 | MTP top-3 | either top-1 | union of both top-3 |
|---|---|---|---|---|---|---|
| code, unlimited rows | 8.66 | 11.41 | 9.50 | 12.55 | 11.04 | 13.61 |
| IFEval | 2.49 | 3.45 | 2.74 | 4.16 | 3.26 | 4.68 |
| multilingual | 1.93 | 2.33 | 2.80 | 3.82 | 3.08 | 4.15 |

(11.41, 12.55 and 13.61 equal TrainLab's STAGED oracles.) Best-first trees on the fused reader, rows estimated from the
reader's own probabilities [P; the MTP head is calibrated, 0.917 predicted against 0.910 right on code; DFlash2 is
overconfident, so its row counts are too low]: code 10.55 at 18 rows, 11.18 at 25, 12.37 at 72, 12.84 at 224, 13.09 at
621; IFEval 3.07 at 7, 3.54 at 16, 4.04 at 63, 4.51 at 958. Slope: +0.69 kept per doubling on code, +0.27 on IFEval.
Line: PASS needs a ceiling of 14.5 and 13.5 kept at 128 rows; result 13.61 and 12.37: KILL.
- This is why 25-30 cannot come from width: kept grows like ln(rows) over the reader's uncertainty, so each extra kept
  token on code costs about 3-4 times the rows, and on chat 13 times.
- On chat a small tree does keep more at equal rows (3.07 at 7 rows against 2.74 for the uncut 16-row chain). That is
  TrainLab's known tree lever. It needs a tree check for the linear-attention layers and cheaper rows, and it is +10-20 %
  in time at 1 ms per row [P], not a broken wall.

### 1.4 Combination (T4 KILL, T5 KILL)
`scripts/t45_fusion_runnerup.py`; tuned on 6 windows per domain, reported on the other 6.

| held-out, 16 rows | DFlash2 | MTP | max-confidence pick | fusion (4 numbers) | perfect pick |
|---|---|---|---|---|---|
| code | 7.58 | 8.29 | 8.13 | 8.61 (x1.038) | 9.54 (x1.151) |
| IFEval | 2.27 | 2.53 | 2.46 | 2.55 (x1.009) | 2.99 (x1.183) |
| multilingual | 1.86 | 2.76 | 2.60 | 2.80 (x1.016) | 3.01 (x1.090) |

The runner-up cache (the 125B's rank 2-3 tokens at verified rows as tie-breakers) keeps x0.973 / x0.978 / x0.970.

## 2. The law, stated once
1. A sweep reads the trunk once and answers R questions of the form "what is the next token after this prefix". The
   questions are fixed before the sweep starts: a layer-by-layer pass cannot feed a row's last layer into another
   row's first. Starting row j+1 a few layers after row j does not help: with a stagger of s layers the pass reads
   48 / s + B trunks for B rows [P], the one-token-at-a-time cost.
2. So kept per sweep is the length of the true path inside the question tree. For a chain that is the run up to the
   first wrong draft; for a tree it grows like the logarithm of the rows (1.3).
3. The readers' drafts are good only next to the frontier (1.2), because the 125B settles its close calls in its
   last layers at the last moment: the early taps carry little of it (TrainLab's EARLYTAP, x0.73 without the tap at
   layer 45), a state one token older already costs 9-18 % of kept, and after any wrong draft the block is lost (1.1).
4. Therefore sweeps per token = reader failures per token, and a reader failure at a close call can be repaired only
   by the model itself, one sweep each. The exact loop already does exactly this; its structure has no slack.
5. A 98 % drafter over 32-64 positions needs lag-1 information at every position. That is the trunk read once per
   token: 2.7 GB, 9.5 ms at the byte floor. TrainLab's and BigMoE's numbers close the other exits: a 3-bit trunk
   agrees 0.92 on code and 0.44 on chat, a 2-bit one 0.07 (D2).

## 3. The best design, with its numbers
- **Inside the rules there is none better than what ships.** The best exact, training-free arrangement on these tables
  is already known: the MTP chain at 32 rows keeps 12.55 on HumanEval+ and 6.23 on MBPP, and it costs 31 chain steps.
  Fusion would add 1-4 % (1.4). No arrangement of existing parts reaches +20 % at equal cost.
- **The design that the law points to** is a reader that carries a stand-in for the trunk state forward and holds its
  lag-2 accuracy at depth. Its measured ceiling is section 1.2's table: x1.60 on HumanEval+, x1.95 on MBPP, x1.5 on
  chat with one reader; x1.95, x2.7 and x1.8-2.0 with a perfect pick of two. It needs training, so it is not this
  mission's result. D-22 was a first step on this road (the stock chain fell to 0.70 by depth 15; b5 holds 0.89).
- Where that gain sits [K `scripts/a1_decisions.py`, 16-row checks on the map windows, MTP]: giving lag-2 accuracy only to
  the 125B's close calls moves code 9.50 to 10.80 and IFEval 2.74 to 3.67; only to its sure rows, 10.24 and 2.80; to
  all rows, 11.94 and 3.95. And half of it needs a state no older than four tokens (1.2). So the reader to train is
  one that predicts the model's close calls several tokens ahead, which is the hard half.
- **What even that ceiling buys** [P `scripts/a5_refresh.py`, byte floor 284 GB/s, trunk 2.7 GB, BREAK_THE_WALL's union
  bytes]: HumanEval+ exact 373 tok/s at 32 rows (today's kept: 269 at 16 rows); with the lossy oracle gate 706 (964
  with the perfect pick at 64 rows); MBPP 441 (591). One stream at 1,000 tok/s on real code is outside the reach of any
  reader that sits two tokens behind the trunk.
- Named doors that are not mine to open: lossy acceptance at the 125B's close calls (FAST track, EvalSuite's gates);
  several streams per sweep (total throughput only; the Lunatic's replay priced 391-515 tok/s at 8-12 streams).

## 4. The one GPU test: LAG2-AGENT (pre-registered Wed 11 Nov 2026 21:1x, not run)
Every lag-2 number above comes from HumanEval+, MBPP and the 36 map windows. Agent turns, the real workload, have no
such table: the stored agent windows give the readers a cut context (`scripts/a4_lag2_dflash_all.py`: DFlash2 is right
77.8 % at depth 0 there, chain 3.10, lag-2 4.34 per 32 rows; a weak number, not a ceiling).
- Question: on fresh agent turns and fresh code with the full context, how much kept would a depth-holding reader gain?
- Run (owner MTP Head; one share lock; the 45 GB loop for the capture, then about 12 GB as DEPTH32 was):
  1. Capture with TrainLab's `sticky/dlog_capture.py`: 24 agent turns (`~/Chat1000/data/agent125` prompts or newer
     ones) and 50 MBPP prompts not among the 376 in `data/dlog/d1102mbpp`; the files hold both readers' inputs.
  2. MTP head b5 at every answer position to depth 31: `gpu_test/g0_mtp31_set.py <set> <capture folder>` (a wrapper
     around TrainLab's `d23h_mtp31.py`; written here, NOT run: it needs the 125B loader).
  3. DFlash2 at every position: `TL_WALK_EVERY=1 TL_WALK_CF=1 TL_WALK_TAG=_lag2 TL_WALK_SETS="<set>=<folder>/*.npz"
     python sticky/d02b_walk.py` (TrainLab's own every-position walk).
  4. Read-out: `python -I gpu_test/g1_lag2_readout.py <set> <walk.npz> <table.npz>` (run here on the existing tables;
     it reproduces 9.58 / 12.55 / 5.93 / 6.23).
- Lines, per 32-row check, either-lag-2 against the better real chain:
  - PASS (the door is worth a training mission): >= x2.0 on agent AND >= x1.7 on fresh MBPP.
  - KILL (close the single-stream drafter track): < x1.4 on agent.
  - BETWEEN otherwise.
- Prediction: agent x1.6-2.0, MBPP x2.3-2.7 (the stored MBPP set gave x2.70): BETWEEN or PASS.

## 5. What I got wrong or could not do
- I predicted the either-of-two lag-2 drafter at 25 per 32 rows on the map windows; it is 22.75.
- The row counts of the trees are an estimate from the readers' own probabilities, not a count of real off-path nodes.
- The class split (sure / mid / tie) exists only for the 36 map windows; MBPP, GSM and agent have no classes here.
- All tables are teacher-forced along the 125B's own answers; no live engine run and no GPU job was made in this mission.
- One temporary copy of out/ was written to this session's scratch folder under /private/tmp for the re-run diff.
- The timestamps in PREREG.md were first written an hour late (21:4x for 20:4x) and corrected before T1 ran; the
  commit times are the record.

## 6. Reproduce
`~/BigMoE/.venv/bin/python -I scripts/<name>.py` (numpy only, each under 30 s): t1_secondlook, t2_laglaw, t3_rowslog,
t45_fusion_runnerup (pre-registered); s0_align, s0_class, s0_look (exploratory, before the lines); a1_decisions,
a2_tail_tf, a3_lag2_either, a4_lag2_dflash_all, a5_refresh, a6_ci (descriptive, after). Outputs are in `out/`.
Inputs, read-only: `~/TrainLab/data/` (d02b_walk_hpd0_*, d02b_walk_d23h_dfl_*, d02b_walk_d17e_*, d23h/, d24/, tree3_map/),
`~/BigMoE/models/bare/vocab.json`.
