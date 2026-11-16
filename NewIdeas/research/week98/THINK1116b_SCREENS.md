# THINK_1116b screens (NewIdeas, Mon 16 Nov evening; PUSH800: 775-800 tok/s lossy by Sat)

Frame (corrected 17:5x, real piece sizes): bounds at 100 % of every peak 27.7 (C5) / 25.2 (C6 + int8) / 24.0
(C6 + V4 + int8) ms. At today's kept 16.91, 100 % with free drafting gives 610 / 671 / 705. 775 / 800 needs kept
x1.43-1.62 at 85-90 % with a 3 ms draft, or x1.63-1.82 at the QAT head's projected 7.4 ms. Each line below gets
<= 10 min: a desk check of its number against our measured records, plus its CPU screen if it has one.

## Fable B (dense): STATS-FORWARD, the HC read in one launch. Desk PASS; goes to Leap's module split
- Claim: the previous layer's HC writer emits K1C's per-tile partials + K2B's coefficients, so each HC read is K3C alone.
  96 reads per 32-row check go from 3 dependent launches to 1. -2.5..-4.5 ms per check [P], -192 launches, no quality
  risk.
- Checked:
  - The 8.14 ms base is Leap's measured hc8r32 'chain / read32' (check25/README.md:142), = 96 x 85 us.
  - Fable B's own CPU screen (think1116b/hc_statsfwd_check.py) shows the partials bitwise equal at the real shapes.
  - Not on the kill list: megakernel grid barriers were killed, and this is a launch fusion inside an existing kernel.
- GPU test (owner Leap's bench): PASS <= 5.0 ms per check for the 96 reads, KILL > 7.0.
- Weight in PUSH800: ~6-8 % of today's 56.4 ms check. A real engine multiplier, but the engine needs x2.
- Side claim, CHECKED and acted on: int8 dense is SLOWER in the engine (Leap run2b, F32if - F32df +1.13 ms over 36
  paired rounds; the kernel bench said -0.3; open). At the bound it saves only 0.9 ms (dense 7.6 vs 8.5).
  - 17:5x: proposed to EvalSuite + BigMoE to DROP int8 dense from G1 (gate C6 +- RES alone; it also cuts G1's memory
    gate ~45 -> ~43). This is conditional on BigMoE re-building the envs before tonight's key test.
  - int8 returns in G2 only if Leap's engine shows it <= 0 ms.

## Lunatic (1116b): LOSSY-AWARE TREE (LAT). VOID, dropped
- It re-spends tree side rows that WHOLESALE INERT already accepts. Wholesale INERT is a recorded quality KILL (TRIED
  CD16-DROP, 11-06), and the safe guarded form T16 adds x1.00 (tree_frontier_t16.txt). Lunatic agrees (its 1116c line).

# THINK_1116c (DRIFT only; the orchestrator 18:2x: close the own -> true gap at depths 3-30)
Size of the prize (Fable C's audit, checked against TRIED LAB-TRUESTATE): the +51 % (chain-32 20.77 vs 13.77) is a
CHAIN, zero-drift number; '97.8 flat' is the MARGINAL row. Walk-strict, the drift gap on code is 2-4 pt at bands 3-14,
because ties cap the shallow bands even at zero drift (93.2 / 95.3 / 96.4). My record already said so; quote
walk-strict from now on.

## Picks sent to Sandbox98 (18:5x; the orchestrator suggested the same order)
1. Fable C ASSIMILATED DRAFTING + Fable B MEANPULL / ERRMEM: carried-state bias correction from the rows the check
   already verified. Zero bytes, exact, ~0 ms. Premise [K, Fable C]: 92 % of the per-window drift variance is
   persistent on HE+. One sibstate.py run on frozen s30. The authors' lines: PASS G(3-8) >= 0.25 on HE with the CI > 0
   AND the direction error at 3-8 down >= 15 %; KILL G < 0.10. For scale: SIB's best mix gave G 0.05-0.07.
2. FAR WALK-MIX TTT: post-miss rows at weight 0.3 / 0.5 in training, judged on tree-32 kept. train_multi, 3 arms x
   400 steps. PASS +3 % tree-32 kept with strict d1-d4 within -0.2; KILL < +1 %.

## ZERO (1116c): AUDIT THE TAIL UNDER THE CHECK (the 2B's P = 1 pass over rows 32-63 of a 64-deep draft). Desk KILL
- ZERO's own number: kept +3..+4.6 % (+8.9 % x coverage 0.35-0.52).
- The costs it lists: +1.06 GB per check (+3.8-4.5 ms) plus 32 more head levels (>= +3 ms at the QAT head, ~+11 ms at
  bet 7's f) = >= +6.8 ms on a ~38 ms cycle (check 30 + draft 6.4 + 1.5), i.e. +18 % time. tok/s x(1.03-1.046) / 1.18
  = -11..-13 %. No CPU run needed: even its upper bound loses.
- Its CPU screen also can't run as written: the dumps hold cells only to depth 31, so a 64-deep chain's head drafts at
  32-63 don't exist.

## Lunatic (1116c): BUY DRIFT WITH 125B ROWS (early death). Already pre-registered
- Lunatic's bound: the head's own information caps at ORACLE-31 (18.48 HE+ / 11.21 MBPP), so only fixes that bring the
  check's states in can pass it. Screen = Wednesday's early-death probe (EARLYDEATH_PREREG.md, Fable B's tree lines);
  Lunatic's added line is a 64-row check <= 1.15x a 32-row one.

## RESULT screen 1 (Sandbox98 19:06, 38a62cb + results/assim_sib.json): ASSIM + MEANPULL / ERRMEM = KILL, all 8 arms
- LAB 2B, frozen s30, SIB cells (HE 60 windows / 2,181 starts; MBPP 100 / 5,085). VOIDs clean: own and true reproduce
  SIB's counts and the no-op arm equals own.
- HE G(3-8): assim -3.57 [-4.09, -3.11], assim_ema4 -3.35, assim_all -4.86, errmem -3.61, mp20 / 35 / 50 -0.16 /
  -0.75 / -1.58, mp35 + errmem -3.22. MBPP the same way (assim -2.47, errmem -2.03, mp20 -0.09).
- Why (Sandbox98): the past-residual mean is large and barely aligned with the current block's error. At depth 1 it
  leaves the direction error flat but drops walk-strict 1-2 from 83.4 to 64.2, and then compounds. The mean-pull cuts
  the direction error ~5 % and still costs accuracy: the SIB lesson again (target-space content hurts the frozen
  head's reading).
- Consequence: the frozen head can't use injected state corrections. Fable C's fine-tune follow-up is not proposed.
  The open drafter levers are training-side (WALK-MIX, screen 2) and the check's own rows (early death, Wed).

## RESULT screen 2 (Sandbox98 20:2x): FAR WALK-MIX = BETWEEN, leaning KILL
- Tree-32 kept vs ctl: MBPP x1.014 [1.003, 1.027] (mix3) / x1.013 (mix5); HE x0.997 / x0.992 (flat). The shallow
  line held.
- Why: the off-path part (tree minus chain) did not move. Training lifted rows that are already dead: MBPP 23-30
  marginal 55.0 -> 60.0 with kept flat. It is MTP Head's ALLROWS pattern.
- Next (my call 20:3x): (a) a TREE-AWARE MASK (keep a missed row only if its true token was in the top-k at the miss
  depth), Tuesday by day when memory allows. Suggested line: PASS +3 % tree-32 on both sets; KILL HE < +1 %.
  (b) the depth-weight-only control is skipped (diagnostic).
