# Exact tree-16 under Leap's reference contract: a ready plan for a YES (NewIdeas, Wed 18 Nov 02:2x; NOTHING ARMED)

The orchestrator asked at 02:3x. The decision is Kamen's (it changes the exact reference). Until a YES, the exact headline is chain-16
on NAX6-N.

## Why
- M4 (BigMoE, Wed 02:10, scripts/m4_night_1117e.sh): EXACT FAIL by BATCH VARIANCE, not a bug (diagnosed 02:2x from the bank's tk / lp).
  - mbpp/16 token 57 is a 1-bf16-step near-tie (C -0.7495 vs T -0.8745). mbpp/30 splits on an exact tie (-0.9839 both).
  - 14 / 16 keys are identical, and each path is deterministic (C1 == C2, T1 == T2).
- Today's kernels do arithmetic that depends on the row count. So tree-16's verify can't be bitwise with the chain or AR at every near-tie.
- Leap's reference contract (CHECK55_LEVERS #6) fixes the K-chunks and summation order for every dense GEMM, the HC read and the head:
  - E14: A16c - A16 = -5.76 ms, C32c - C32one = -7.04 ms. C8 v2: SD == AR end to end under it.
  - Code: ~/Leap/scripts/week98/refc.py. install(ref2) runs after ref2.apply(); ON[0] toggles it.

## Scope (recommended): the EXACT headline's build only, this week
- The lossy stack, its gates (Thursday's tier 1, Friday night's card gate) and CERTIFY98 (Wed night's capture + Thursday's count
  check) stay on NAX6-N's build, untouched. The capture and the count check still share ONE build.
- So EvalSuite runs PART A of its RE-CERT PRICE (certify; ~4 h, + 1.6-3.3 h if stage 2 runs). It skips PART B (the ~6 h rebase of
  what is pinned to today's .so), which is only needed once the lossy stack moves.
- The contract's -5.76 ms on the LOSSY 16-row check is next week's lever, with Part B.

## 1. Who flips which flags
- **Leap, Wed by day (slice-level + small GPU):**
  a. Deliver refc as an opt-in installer for r57's exact paths (chain + tree) with a snapshot recipe:
     - which exact levers it composes with. REF2_BOF / BOF2-O / REF2_HC_B are candidates; hc_b and the contract both touch the HC read,
       so name the set and its bits.
     - the env name for BigMoE (off == today's build, bitwise).
  b. EvalSuite's VOID gate: the row-invariance self-test WITH TREE MASKS (tree-16 verify rows == AR rows, bitwise, 1-16 rows).
  c. refc patches mx.quantized_matmul globally. Confirm the drafter's kernels (t30b Q8H32 / c2) are untouched, or price the draft ms
     with ON.
- **BigMoE, Wed after the lossy r57 plumbing:** wire, snapshot and key-test only, as with G2's installers.
  a. One env in ref2's exact path and r57's exact tree path that calls refc.install + ON. Default off.
  b. One snapshot (.so / blob) for EvalSuite's pin: the contract + the levers Leap named.
  c. Thursday's key test adds M4C (~10 min): M4's 16 keys with the contract ON. Tree-16 == chain-16 == AR on every token,
     mbpp/16 + mbpp/30 included. If M4C FAILs, it's NO-GO this week and the headline stays chain-16 on NAX6-N.
- **EvalSuite, Wed by day (its ~4-5 h):** the PREREG on NAX6's template once the snapshot + flags are named, then the flagcheck head
  tests and the chain dry run.

## 2. Placement this week (nothing moves for the lossy stack)
- **Wed day:** Leap 1a-c; BigMoE 1a-b after the lossy plumbing; EvalSuite's PREREG.
- **Wed night (as booked):** CERTIFY98 (b) capture ~23:30 -> ~04:30 on NAX6-N; then MTP Head's 16 GB job.
- **Thu midday key test:** + M4C (~10 min).
- **Thu night, from ~23:35, ONE chain:**
  - (i) lossy timing, runner 69, ~45 min (as booked).
  - (ii) exact timing UNDER THE CONTRACT, quiet:
    - chain-16 (C), ~45 min: the reference rows for the 264, plus a chain headline.
    - then tree-16 (C), the Q8H32 vs c2 A/B in order-balanced halves, ~90 min. Token identity 264 / 264 vs (ii)'s chain-16 rows.
  - (ii-b) insurance: chain-16 on NAX6-N, ~45 min. It is the exact headline if the re-cert fails or (ii) breaks.
  - (iii) the CERTIFY98 count check, ~30 min, on NAX6-N (the capture's build).
  - (iv) EvalSuite PART A on the contract snapshot, ~4 h:
    - VOID gates;
    - same scores vs NAX6 (decision 4 on the core 210, AR at C 192, ~2 h, ~34 GB);
    - exactness (the loop == its own AR for 16- / 32-row chains AND TREES on R-56d's 36 prompts, ~1.5 h);
    - runtime tier 0 at C 256 (~20 min).
  - Total ~8 h 15 -> ends ~07:50 Fri. If the night starts late, (iv) moves to Friday's day window (big tests to 15:00 while away).
- **Fri day:** stage 2 (IFX 220, ~1.6 h per arm) only if IFEval alone fails: ~08:00-11:20.
- **Fri night (as booked):** the card gate (lossy). Sat ~07:00: the lossy verdict.

## 3. When the exact tree-16 headline is measured
- MEASURED Thursday night in (ii), ~01:00-02:45 Fri.
- VALID as exact once Part A passes: ~08:00 Fri (~11:30 if stage 2 runs).
- Quote it as "exact vs the contract reference, certified same scores vs NAX6".
- Expected [P]: ~144 -> ~150-165 tok/s with the contract + exact levers (E14 pricing, before the c2 A/B).
- Fallbacks: M4C FAIL (Thu midday) -> no contract runs, no re-cert. Part A FAIL (Fri morning) -> the headline = (ii-b).

## 4. Risks
- A switch moves same-scores comparators: quote every "vs exact" with its reference. REF2-T -> NAX6 moved HE+ by -3.33 pt and
  flipped a verdict.
- Saturday has two builds: lossy on NAX6-N, exact on the contract. Each claim stands on its own build; the write-up says so.
- BigMoE's Wednesday: the lossy plumbing comes first. If the contract snapshot misses Thursday's key test, the YES moves to next week
  whole (no partial runs).
- Night length: (i)-(iv) is ~8 h 15. A late start pushes (iv), not (i)-(iii).
